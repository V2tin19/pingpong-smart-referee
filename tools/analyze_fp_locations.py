# -*- coding: utf-8 -*-
"""test_4 误检位置分析：地板掩码与置信度门限的实测依据。

回答两个问题（感知降噪 P1 前置测量）：
1. **误检（FP）都在哪？** 有多少落在"球台近边以下 30px"的地板区
   （可被 `--reject-floor 30` 物理丢弃），多少在台面区（只能靠模型重训）？
2. **conf=0.35 能杀多少 FP、伤多少 TP？**（Test4 报告建议的门限）

用法（需 ultralytics 环境）：
    python tools/analyze_fp_locations.py --step 2   # 隔帧采样，~3000 帧推理
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from collections import Counter

import cv2

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.geometry import preset


def load_gt(zip_path: str, max_frame: int):
    with zipfile.ZipFile(zip_path) as z:
        raw = json.loads(z.read("ball_markup.json").decode("utf-8"))
    gt = {}
    for k, v in raw.items():
        f = int(k)
        if f < max_frame and v and v.get("x", -1) >= 0 and v.get("y", -1) >= 0:
            gt[f] = (float(v["x"]), float(v["y"]))
    return gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/test_4_50s.mp4")
    ap.add_argument("--zip", default="data/test_4.zip")
    ap.add_argument("--weights", default="weights/yolo11n_clean_best.pt")
    ap.add_argument("--conf", type=float, default=0.18)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--step", type=int, default=2, help="隔 N 帧采样")
    ap.add_argument("--max-frames", type=int, default=6002)
    args = ap.parse_args()

    calib = preset("test4")
    roi = calib.roi_box
    NEAR_EDGE_Y = max(p[1] for p in calib.table_polygon)     # 750
    MASK_LINE = NEAR_EDGE_Y + 30.0                            # 掩码建议线 780
    CONF_LINE = 0.35                                          # 报告建议门限

    gt = load_gt(args.zip, args.max_frames)
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)

    tp = 0
    fp_total = 0
    fp_by_zone = Counter()          # floor / above_mask
    fp_by_conf = Counter()          # weak(<0.35) / strong(>=0.35)
    fp_survive_both = []            # 既在掩码之上、conf 又高的 FP（只能靠重训解决）
    tp_conf_below_35 = 0            # conf=0.35 会误杀的 TP
    tp_below_mask = 0               # 掩码会误杀的 TP（预期 0）
    tp_sizes = []                   # (w, h, ar) 真球框尺寸样例
    fp_sizes = []                   # (w, h, ar) 误检框尺寸样例

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_id >= args.max_frames:
            break
        if frame_id % args.step == 0:
            crop = frame[roi.y_min:roi.y_max, roi.x_min:roi.x_max]
            results = model.predict(crop, imgsz=args.imgsz, conf=args.conf, verbose=False)
            dets = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    c = float(boxes.conf[i].cpu().item())
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    w = float(xyxy[2] - xyxy[0])
                    h = float(xyxy[3] - xyxy[1])
                    dets.append((
                        (xyxy[0] + xyxy[2]) / 2.0 + roi.x_min,
                        (xyxy[1] + xyxy[3]) / 2.0 + roi.y_min,
                        c, w, h,
                    ))

            gt_pos = gt.get(frame_id)
            for (x, y, c, w, h) in dets:
                if gt_pos is not None and math.hypot(x - gt_pos[0], y - gt_pos[1]) <= 25.0:
                    tp += 1
                    tp_sizes.append((w, h, w / max(h, 1e-6)))
                    if y > MASK_LINE:
                        tp_below_mask += 1
                    if c < CONF_LINE:
                        tp_conf_below_35 += 1
                else:
                    fp_total += 1
                    fp_sizes.append((w, h, w / max(h, 1e-6)))
                    zone = "floor" if y > MASK_LINE else "above_mask"
                    fp_by_zone[zone] += 1
                    if c < CONF_LINE:
                        fp_by_conf["weak"] += 1
                    else:
                        fp_by_conf["strong"] += 1
                    if zone == "above_mask" and c >= CONF_LINE:
                        fp_survive_both.append(
                            (frame_id, round(x), round(y), round(c, 2), round(w), round(h))
                        )
        frame_id += 1

    cap.release()

    print("=" * 70)
    print(f"采样帧数: {math.ceil(args.max_frames / args.step)}  (conf>={args.conf})")
    print(f"TP（与真值 25px 内匹配的检出）: {tp}")
    print(f"FP（全部误检）: {fp_total}")
    print("-" * 70)
    print(f"FP 位置分布（掩码线 y>{MASK_LINE:.0f} = 近边+30px）:")
    print(f"  地板区（可被 --reject-floor 30 物理丢弃）: "
          f"{fp_by_zone['floor']}  ({100.0*fp_by_zone['floor']/max(fp_total,1):.1f}%)")
    print(f"  台面区（掩码管不到）:                        "
          f"{fp_by_zone['above_mask']}  ({100.0*fp_by_zone['above_mask']/max(fp_total,1):.1f}%)")
    print(f"FP 置信度分布（门限线 conf={CONF_LINE}）:")
    print(f"  弱误检 conf<{CONF_LINE}（提门限可杀）:       "
          f"{fp_by_conf['weak']}  ({100.0*fp_by_conf['weak']/max(fp_total,1):.1f}%)")
    print(f"  强误检 conf>={CONF_LINE}（只能靠重训）:      "
          f"{fp_by_conf['strong']}  ({100.0*fp_by_conf['strong']/max(fp_total,1):.1f}%)")
    print("-" * 70)
    killed = fp_by_zone["floor"] + sum(
        1 for _ in ()
    )
    survive = len(fp_survive_both)
    print(f"两项叠加后仍存活的 FP（台面区且 conf>=0.35）: {survive}  "
          f"({100.0*survive/max(fp_total,1):.1f}%)")
    print(f"误杀检查: 掩码误杀 TP {tp_below_mask} 个；conf=0.35 误杀 TP "
          f"{tp_conf_below_35} 个 ({100.0*tp_conf_below_35/max(tp,1):.1f}% 的 TP)")
    if fp_survive_both:
        print(f"\n存活 FP 样例（帧, x, y, conf, w, h）前 15 个: {fp_survive_both[:15]}")

    # ---- bbox 尺寸可分性（潜在的零成本过滤手段）----
    if tp_sizes and fp_sizes:
        def side(v, idx):
            s = sorted(x[idx] for x in v)
            return s[0], s[len(s) // 2], s[-1]
        tp_w = side(tp_sizes, 0); tp_h = side(tp_sizes, 1); tp_ar = side(tp_sizes, 2)
        fp_w = side(fp_sizes, 0); fp_h = side(fp_sizes, 1); fp_ar = side(fp_sizes, 2)
        print("\n----- bbox 尺寸分布（w, h 单位 px；ar = w/h）-----")
        print(f"TP: w(min/中位/max)={tp_w[0]:.0f}/{tp_w[1]:.0f}/{tp_w[2]:.0f}  "
              f"h={tp_h[0]:.0f}/{tp_h[1]:.0f}/{tp_h[2]:.0f}  "
              f"ar={tp_ar[0]:.2f}/{tp_ar[1]:.2f}/{tp_ar[2]:.2f}")
        print(f"FP: w(min/中位/max)={fp_w[0]:.0f}/{fp_w[1]:.0f}/{fp_w[2]:.0f}  "
              f"h={fp_h[0]:.0f}/{fp_h[1]:.0f}/{fp_h[2]:.0f}  "
              f"ar={fp_ar[0]:.2f}/{fp_ar[1]:.2f}/{fp_ar[2]:.2f}")
        # 常见过滤器的效果预估：尺寸上下限 + 长宽比上限
        for name, keep in (
            ("max(w,h)<=40", lambda w, h, ar: max(w, h) <= 40),
            ("max(w,h)<=30", lambda w, h, ar: max(w, h) <= 30),
            ("min(w,h)>=4 且 max(w,h)<=40", lambda w, h, ar: min(w, h) >= 4 and max(w, h) <= 40),
        ):
            tp_k = sum(1 for (w, h, ar) in tp_sizes if keep(w, h, ar))
            fp_k = sum(1 for (w, h, ar) in fp_sizes if keep(w, h, ar))
            print(f"  过滤器[{name}]: 保留 TP {tp_k}/{len(tp_sizes)}"
                  f" ({100.0*tp_k/len(tp_sizes):.1f}%), "
                  f"保留 FP {fp_k}/{len(fp_sizes)} "
                  f"({100.0*fp_k/len(fp_sizes):.1f}%)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
