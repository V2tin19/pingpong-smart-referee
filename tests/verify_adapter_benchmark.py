"""
tests/verify_adapter_benchmark.py
=================================

用**真实视频 + 官方标注**验证 :class:`adapters.yolo_adapter.YOLOAdapter`。

这个脚本回答三个问题：

1. **坐标还原对不对？** 独立地"手工裁剪 + 直接推理 + 加偏移"，逐帧比对
   适配器输出，确保它真的把裁剪坐标还原成了原图坐标。
2. **召回对不对？** 在 OpenTTGames ``test_game_2.avi`` 的无泄漏测试集
   （390~778 帧）上算 Recall@15px / Recall@25px / Precision / MAE，
   与 Mode A 基线（83.55%）对照。
3. **契约守不守？** 逐帧检查每条观测都能通过 §7.1/§7.3 校验并 JSON 往返。

用法::

    python tests/verify_adapter_benchmark.py \
        --video  E:/2026Autumn/YOLO/table-tennis-detection/videos/test_game_2.avi \
        --gt     E:/2026Autumn/YOLO/table-tennis-detection/data_annotations/test_dataset_info.csv \
        --weights weights/yolo11n_clean_best.pt \
        --preset game2

注意
----
适配器是**无状态**的：每帧独立取置信度最高的检测，不做卡尔曼预测、
不做 180px 位移门限、不做双帧确认。因此本脚本测的是"适配器本身的感知上限"，
而不是最终系统指标——那要等 ``TrajectoryEventProcessor`` 落地（阶段三）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.yolo_adapter import YOLOAdapter  # noqa: E402
from core.geometry import preset  # noqa: E402
from core.schemas import FrameObservation  # noqa: E402

# Mode A（仅 ROI + YOLO11n）基线，来自 yolo_core/configs_and_data/comparison_results.csv
BASELINE = {
    "name": "微调 YOLO11n + 模式 A (仅ROI)",
    "recall_15px": 83.55,
    "recall_25px": 83.55,
    "precision": 99.69,
    "mae_px": 4.78,
}


def load_ground_truth(gt_csv: str, game_id: int = 2) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    return df[df["game_id"] == game_id].sort_values("frame_id").reset_index(drop=True)


def verify_coordinate_restoration(
    adapter: YOLOAdapter,
    frames: List[Tuple[int, np.ndarray]],
    imgsz: int,
    tolerance_px: float = 1e-6,
) -> Dict[str, object]:
    """独立重建"裁剪 → 推理 → 加偏移"，与适配器输出逐帧比对。

    这里刻意**绕过适配器**自己裁一次、推一次，用来证明适配器的偏移量加法
    没有写错（这是最容易悄悄错 150px 的地方）。
    """
    from ultralytics import YOLO

    raw_model = YOLO(adapter.weights_path)
    calib = adapter.calibration or preset("game2")
    checked = 0
    max_delta = 0.0

    for frame_id, frame in frames:
        h, w = frame.shape[:2]
        scaled = calib.scaled_to(w, h)
        box = scaled.roi_box.clip(w, h)
        crop = frame[box.y_min:box.y_max, box.x_min:box.x_max]

        res = raw_model(crop, imgsz=imgsz, conf=adapter.conf_threshold, verbose=False)[0]
        if len(res.boxes) == 0:
            continue

        boxes = res.boxes.xyxy.detach().cpu().numpy().astype(float)
        confs = res.boxes.conf.detach().cpu().numpy().astype(float)
        best = int(np.argmax(confs))
        manual_cx = (boxes[best][0] + boxes[best][2]) / 2.0 + box.x_min
        manual_cy = (boxes[best][1] + boxes[best][3]) / 2.0 + box.y_min

        obs = adapter.predict_frame(frame, frame_id, frame_id / 25.0)
        if not obs.ball.visible:
            raise AssertionError(f"frame {frame_id}: adapter missed a ball the manual pass found")

        dx = abs(obs.ball.center_px[0] - manual_cx)
        dy = abs(obs.ball.center_px[1] - manual_cy)
        max_delta = max(max_delta, dx, dy)
        if dx > tolerance_px or dy > tolerance_px:
            raise AssertionError(
                f"frame {frame_id}: coordinate restoration mismatch, "
                f"adapter={obs.ball.center_px} manual=({manual_cx:.2f}, {manual_cy:.2f})"
            )
        checked += 1

    return {"frames_checked": checked, "max_delta_px": max_delta}


def compute_metrics(log: List[Dict], name: str) -> Dict[str, float]:
    """与 evaluate_pipeline.py 完全一致的 TP/FP/FN 口径，便于直接对照。"""

    def prf1(tol: float) -> Tuple[int, int, int]:
        tp = sum(
            1
            for r in log
            if r["gt_has_ball"] and r["dist"] is not None and r["dist"] <= tol
        )
        fp = sum(
            1
            for r in log
            if r["has_pred"] and (not r["gt_has_ball"] or r["dist"] is None or r["dist"] > tol)
        )
        fn = sum(
            1
            for r in log
            if r["gt_has_ball"] and (not r["has_pred"] or r["dist"] is None or r["dist"] > tol)
        )
        return tp, fp, fn

    tp15, _, _ = prf1(15.0)
    tp25, fp25, fn25 = prf1(25.0)
    precision = tp25 / (tp25 + fp25) * 100.0 if (tp25 + fp25) else 0.0
    recall25 = tp25 / (tp25 + fn25) * 100.0 if (tp25 + fn25) else 0.0
    recall15 = tp15 / (tp15 + fn25) * 100.0 if (tp15 + fn25) else 0.0
    f1 = 2 * precision * recall25 / (precision + recall25) if (precision + recall25) else 0.0

    errs = [r["dist"] for r in log if r["dist"] is not None]
    mae = float(np.mean(errs)) if errs else float("nan")
    # RMSE 一并给出：TTNet 分支报的是 RMSE，两个指标不可混用，都算出来便于横向对齐
    rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")

    return {
        "scope": name,
        "frames": len(log),
        "gt_ball_frames": sum(1 for r in log if r["gt_has_ball"]),
        "recall_15px": round(recall15, 2),
        "recall_25px": round(recall25, 2),
        "precision_25px": round(precision, 2),
        "f1_25px": round(f1, 2),
        "mae_px": round(mae, 2),
        "rmse_px": round(rmse, 2),
        "detected_frames": sum(1 for r in log if r["has_pred"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify YOLOAdapter against ground truth")
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--preset", default=None, help="标定预设名（game2/rally/plos）")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15, help="与 Mode A 基线对齐用 0.15")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--game-id", type=int, default=2)
    ap.add_argument("--test-split-start", type=int, default=390)
    ap.add_argument("--skip-restoration", action="store_true")
    args = ap.parse_args()

    gt = load_ground_truth(args.gt, args.game_id)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video {args.video}")
        return 2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(len(gt), total, args.max_frames or total)

    calib = None
    if args.preset:
        calib = preset(args.preset)

    adapter = YOLOAdapter(
        args.weights,
        calibration=calib,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        strict_time=True,
    )

    print("=" * 78)
    print(f"Video      : {os.path.basename(args.video)} ({w}x{h} @ {fps:.1f}fps, {n_frames} frames)")
    print(f"Calibration: {args.preset or '(full frame, no ROI)'}")
    print(f"Adapter    : imgsz={args.imgsz}, conf={args.conf}")
    print("=" * 78)

    log: List[Dict] = []
    sample_frames: List[Tuple[int, np.ndarray]] = []
    roundtrip_failures = 0
    invisible_violations = 0

    for frame_id in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        obs: FrameObservation = adapter.predict_frame(frame, frame_id, frame_id / fps)

        # --- 契约检查（真实数据上逐帧跑一遍） ---
        d = obs.to_dict()
        if FrameObservation.from_dict(d).to_dict() != d:
            roundtrip_failures += 1
        if not d["ball"]["visible"]:
            if any(d["ball"][k] is not None for k in ("center_px", "bbox_xyxy", "confidence")):
                invisible_violations += 1

        row = gt.iloc[frame_id]
        gt_has_ball = bool(pd.notna(row["a"]) and pd.notna(row["b"]))
        gt_cx = float(row["a"]) * 1920.0 if gt_has_ball else None
        gt_cy = float(row["b"]) * 1080.0 if gt_has_ball else None

        has_pred = obs.ball.visible
        dist: Optional[float] = None
        if has_pred:
            if gt_has_ball:
                dist = float(np.hypot(obs.ball.center_px[0] - gt_cx, obs.ball.center_px[1] - gt_cy))
            else:
                dist = float("inf")

        log.append({"frame_id": frame_id, "gt_has_ball": gt_has_ball, "has_pred": has_pred, "dist": dist})

        # 留几帧给"坐标还原"独立校验
        if not args.skip_restoration and frame_id % 60 == 0 and len(sample_frames) < 6:
            sample_frames.append((frame_id, frame.copy()))

    cap.release()

    # --- 1. 坐标还原独立性校验 ---
    restoration: Optional[Dict[str, object]] = None
    if sample_frames:
        print("\n[1] 坐标还原独立校验（绕过适配器手工裁剪+推理，逐帧比对）")
        # 主循环已推进到末帧，这里要回到第 0 帧重放，
        # 正是 BaseAdapter.reset() 文档里说的"跳转回放"场景。
        adapter.reset()
        restoration = verify_coordinate_restoration(adapter, sample_frames, args.imgsz)
        print(
            f"    OK: {restoration['frames_checked']} 帧全部一致，"
            f"最大偏差 {restoration['max_delta_px']:.2e} px"
        )

    # --- 2. 契约检查汇总 ---
    print("\n[2] 契约检查（真实数据逐帧）")
    print(f"    JSON 往返失败: {roundtrip_failures}    §7.3 漏检违规: {invisible_violations}")
    print(f"    适配器自述: {adapter.describe()}")

    # --- 3. 指标 ---
    full = compute_metrics(log, f"全视频 (0~{n_frames - 1})")
    test = compute_metrics(log[args.test_split_start:], f"测试集 ({args.test_split_start}~{n_frames - 1})")

    print("\n[3] 感知指标（无状态适配器，未接跟踪门限）")
    header = (f"{'Scope':<22}| {'R@15px':>8} | {'R@25px':>8} | {'Prec@25':>8} | "
              f"{'F1@25':>7} | {'MAE':>8} | {'RMSE':>8}")
    print(header)
    print("-" * len(header))
    for m in (full, test):
        print(
            f"{m['scope']:<22}| {m['recall_15px']:7.2f}% | {m['recall_25px']:7.2f}% | "
            f"{m['precision_25px']:7.2f}% | {m['f1_25px']:6.2f}% | {m['mae_px']:7.2f}px | "
            f"{m['rmse_px']:7.2f}px"
        )

    print("\n[4] 与 Mode A 基线对照（测试集 390~778）")
    print(
        f"    基线 Recall@25px={BASELINE['recall_25px']}%  Prec={BASELINE['precision']}%  "
        f"MAE={BASELINE['mae_px']}px"
    )
    print(
        f"    适配器 Recall@25px={test['recall_25px']}%  Prec={test['precision_25px']}%  "
        f"MAE={test['mae_px']}px"
    )
    delta_r = test["recall_25px"] - BASELINE["recall_25px"]
    delta_p = test["precision_25px"] - BASELINE["precision"]
    print(f"    差值: Recall {delta_r:+.2f}pp, Precision {delta_p:+.2f}pp")
    print(
        "    说明: 适配器无状态、无 180px 位移门限，因此召回可能略高、"
        "查准可能略低——这正是阶段三要补回来的部分。"
    )

    # --- 退出码：契约必须零违规 ---
    ok = roundtrip_failures == 0 and invisible_violations == 0
    if restoration is not None:
        ok = ok and restoration["max_delta_px"] < 1e-6
    print("\n" + ("PASS" if ok else "FAIL") + ": 契约与坐标还原校验")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
