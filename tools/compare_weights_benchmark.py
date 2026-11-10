"""
tools/compare_weights_benchmark.py
==================================

在 test_4 数据集上对比两个 YOLO 权重的检测精度与假阳性：
1. 权重 A: weights/yolo11n_clean_best.pt (当前仓库在用权重)
2. 权重 B: E:/2026Autumn/YOLO/table-tennis-ball-detection/Models/detect/ball_detector/weights/best.pt (用户指定权重)

统计指标：
- Recall@5px, @10px, @20px, @25px, @50px
- MAE (Mean Absolute Error) / RMSE
- Precision（仅覆盖有明确标注的帧；缺项 UNKNOWN 不计入）
- False Positives（明确不可见标注上的检出，不等于全片真实无球虚警率）
- 解码与有标注帧推理的混合吞吐（不是纯推理 FPS）

输入尺寸优化请用 benchmark_input_sizes.py：包含独立耗时、预热与基准筛选。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.geometry import preset


def load_ground_truth(zip_path: str, max_frame: int = 6002) -> Dict[int, Optional[Tuple[float, float]]]:
    with zipfile.ZipFile(zip_path) as z:
        raw = json.loads(z.read("ball_markup.json").decode("utf-8"))
    gt = {}
    for k, v in raw.items():
        f = int(k)
        if f < max_frame:
            gt[f] = ((float(v["x"]), float(v["y"]))
                     if v and v.get("x", -1) >= 0 and v.get("y", -1) >= 0 else None)
    return gt


def evaluate_model(
    model_path: str,
    video_path: str,
    gt_dict: Dict[int, Optional[Tuple[float, float]]],
    preset_name: str = "test4",
    imgsz: int = 960,
    conf_thresh: float = 0.18,
    max_frames: int = 6002,
) -> Dict[str, float]:
    from ultralytics import YOLO

    calib = preset(preset_name)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")
    roi = calib.scaled_to(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))).roi_box
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total_video_frames, max_frames)

    # 预热 1 帧
    ret, frame = cap.read()
    if ret:
        crop = frame[roi.y_min:roi.y_max, roi.x_min:roi.x_max]
        model.predict(crop, imgsz=imgsz, conf=conf_thresh, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 统计项
    total_gt = sum(point is not None for fid, point in gt_dict.items() if fid < n_frames)
    decoded_frames = 0
    gt_frames_evaluated = 0
    non_gt_frames_evaluated = 0

    hits_5 = 0
    hits_10 = 0
    hits_20 = 0
    hits_25 = 0
    hits_50 = 0
    errors: List[float] = []

    false_positives = 0
    detected_on_gt = 0
    detected_confidences: List[float] = []

    t0 = time.time()
    for frame_id in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        decoded_frames += 1
        # 稀疏标注缺项是 UNKNOWN，不是无球。只对有明确标注的帧计分。
        if frame_id not in gt_dict:
            continue
        is_gt = gt_dict[frame_id] is not None
        if is_gt:
            gt_frames_evaluated += 1
        else:
            non_gt_frames_evaluated += 1

        crop = frame[roi.y_min:roi.y_max, roi.x_min:roi.x_max]
        results = model.predict(crop, imgsz=imgsz, conf=conf_thresh, verbose=False)

        best_ball = None
        best_conf = 0.0

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                c = float(boxes.conf[i].cpu().item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                cx = (xyxy[0] + xyxy[2]) / 2.0 + roi.x_min
                cy = (xyxy[1] + xyxy[3]) / 2.0 + roi.y_min
                if c > best_conf:
                    best_conf = c
                    best_ball = (cx, cy)

        if best_ball is not None:
            detected_confidences.append(best_conf)
            if is_gt:
                detected_on_gt += 1
                gt_pos = gt_dict[frame_id]
                dist = math.hypot(best_ball[0] - gt_pos[0], best_ball[1] - gt_pos[1])
                errors.append(dist)
                if dist <= 5.0:
                    hits_5 += 1
                if dist <= 10.0:
                    hits_10 += 1
                if dist <= 20.0:
                    hits_20 += 1
                if dist <= 25.0:
                    hits_25 += 1
                if dist <= 50.0:
                    hits_50 += 1
            else:
                false_positives += 1

    cap.release()
    elapsed = time.time() - t0
    if decoded_frames != n_frames:
        raise RuntimeError(f"Incomplete video decode: {decoded_frames}/{n_frames}; evaluation rejected")

    # 没有可计算定位误差的检出时显示 nan，而不是虚构完美的 0px。
    mae = float(np.mean(errors)) if errors else math.nan
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else math.nan
    total_detections = detected_on_gt + false_positives
    precision_25 = (hits_25 / total_detections * 100.0) if total_detections > 0 else 0.0

    return {
        "frames_evaluated": n_frames,
        "elapsed_s": elapsed,
        "fps": n_frames / max(elapsed, 1e-6),
        "total_gt": total_gt,
        "gt_frames": gt_frames_evaluated,
        "non_gt_frames": non_gt_frames_evaluated,
        "detected_on_gt": detected_on_gt,
        "hits_5": hits_5,
        "hits_10": hits_10,
        "hits_20": hits_20,
        "hits_25": hits_25,
        "hits_50": hits_50,
        "recall_5": (hits_5 / total_gt * 100.0) if total_gt > 0 else 0.0,
        "recall_10": (hits_10 / total_gt * 100.0) if total_gt > 0 else 0.0,
        "recall_20": (hits_20 / total_gt * 100.0) if total_gt > 0 else 0.0,
        "recall_25": (hits_25 / total_gt * 100.0) if total_gt > 0 else 0.0,
        "recall_50": (hits_50 / total_gt * 100.0) if total_gt > 0 else 0.0,
        "false_positives": false_positives,
        "fp_rate_pct": (false_positives / non_gt_frames_evaluated * 100.0) if non_gt_frames_evaluated > 0 else 0.0,
        "precision_25": precision_25,
        "mae": mae,
        "rmse": rmse,
        "mean_conf": float(np.mean(detected_confidences)) if detected_confidences else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/test_4_50s.mp4")
    parser.add_argument("--zip", default="data/test_4.zip")
    parser.add_argument("--weight_a", default="weights/yolo11n_clean_best.pt")
    parser.add_argument(
        "--weight_b",
        default="E:/2026Autumn/YOLO/table-tennis-ball-detection/Models/detect/ball_detector/weights/best.pt",
    )
    parser.add_argument("--max_frames", type=int, default=6002)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--preset", default=None, help="Preset name (default: auto guess)")
    args = parser.parse_args()

    from core.geometry import guess_preset_name
    preset_name = args.preset or guess_preset_name(args.video)

    print(f"Loading Ground Truth from {args.zip} (up to {args.max_frames} frames)...")
    gt = load_ground_truth(args.zip, max_frame=args.max_frames)
    print(f"Loaded {sum(point is not None for point in gt.values())} visible labels, "
          f"{sum(point is None for point in gt.values())} invisible labels; unknown frames excluded.")
    print("注意：不可见标注不等于物理无球；FPS包含全片解码，不是纯推理速度。")

    print("\n" + "=" * 70)
    print("Evaluating Model A: 当前在用 clean_best.pt")
    print(f"Path: {args.weight_a}")
    print("=" * 70)
    res_a = evaluate_model(
        args.weight_a, args.video, gt, preset_name=preset_name,
        imgsz=args.imgsz, conf_thresh=args.conf, max_frames=args.max_frames
    )

    print("\n" + "=" * 70)
    print("Evaluating Model B: 新训练 yolo11n_game1_v1")
    print(f"Path: {args.weight_b}")
    print("=" * 70)
    res_b = evaluate_model(
        args.weight_b, args.video, gt, preset_name=preset_name,
        imgsz=args.imgsz, conf_thresh=args.conf, max_frames=args.max_frames
    )

    # 打印对比结果
    print("\n\n" + "#" * 78)
    print(f"{'对比维度':<26} | {'Model A (clean_best)':<22} | {'Model B (game1_v1)':<22}")
    print("-" * 78)
    print(f"{'评估总帧数':<26} | {res_a['frames_evaluated']:<22} | {res_b['frames_evaluated']:<22}")
    print(f"{'真值球帧数 (GT)':<26} | {res_a['total_gt']:<22} | {res_b['total_gt']:<22}")
    print(f"{'Recall @ 5px':<26} | {res_a['recall_5']:.2f}% ({res_a['hits_5']}) | {res_b['recall_5']:.2f}% ({res_b['hits_5']})")
    print(f"{'Recall @ 10px':<26} | {res_a['recall_10']:.2f}% ({res_a['hits_10']}) | {res_b['recall_10']:.2f}% ({res_b['hits_10']})")
    print(f"{'Recall @ 20px':<26} | {res_a['recall_20']:.2f}% ({res_a['hits_20']}) | {res_b['recall_20']:.2f}% ({res_b['hits_20']})")
    print(f"{'Recall @ 25px (标准)':<26} | {res_a['recall_25']:.2f}% ({res_a['hits_25']}) | {res_b['recall_25']:.2f}% ({res_b['hits_25']})")
    print(f"{'Recall @ 50px':<26} | {res_a['recall_50']:.2f}% ({res_a['hits_50']}) | {res_b['recall_50']:.2f}% ({res_b['hits_50']})")
    print("-" * 78)
    print(f"{'平均定位误差 (MAE)':<26} | {res_a['mae']:.2f} px {'':<14} | {res_b['mae']:.2f} px")
    print(f"{'均方根误差 (RMSE)':<26} | {res_a['rmse']:.2f} px {'':<14} | {res_b['rmse']:.2f} px")
    print(f"{'不可见标注帧检出 (FP)':<26} | {res_a['false_positives']} 帧 ({res_a['fp_rate_pct']:.2f}%) {'':<6} | {res_b['false_positives']} 帧 ({res_b['fp_rate_pct']:.2f}%)")
    print(f"{'总体查准率 Precision@25px':<26} | {res_a['precision_25']:.2f}% {'':<15} | {res_b['precision_25']:.2f}%")
    print(f"{'检出平均置信度':<26} | {res_a['mean_conf']:.3f} {'':<17} | {res_b['mean_conf']:.3f}")
    print(f"{'全片解码+标注帧推理 FPS':<26} | {res_a['fps']:.1f} FPS {'':<14} | {res_b['fps']:.1f} FPS")
    print("#" * 78)


if __name__ == "__main__":
    main()
