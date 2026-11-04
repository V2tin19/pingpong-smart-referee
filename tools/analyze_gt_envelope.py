# -*- coding: utf-8 -*-
"""分析 test_4 官方真值：真实球的活动包络与帧间速度分布。

目的（感知侧降噪待办的前置测量）：
1. 真实球的图像坐标包络（y 分布）→ 确定球台近边以下多少像素是"真球从不出现的
   地板区"，为 ROI 掩码（Test4 报告 §四.1 第2项）给出**零召回损失**的安全边界；
2. 真实球的帧间位移/速度分布（120fps）→ 为"单帧位移硬截断"（§四.1 第3项）
   给出物理上界依据。

用法：
    python tools/analyze_gt_envelope.py [--zip data/test_4.zip] [--max-frame 6002]
"""
from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from collections import defaultdict


def load_gt(zip_path: str, max_frame: int):
    with zipfile.ZipFile(zip_path) as z:
        raw = json.loads(z.read("ball_markup.json").decode("utf-8"))
    gt = {}
    for k, v in raw.items():
        f = int(k)
        if f < max_frame and v and v.get("x", -1) >= 0 and v.get("y", -1) >= 0:
            gt[f] = (float(v["x"]), float(v["y"]))
    return gt


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(int(len(s) * p / 100.0), len(s) - 1)
    return s[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/test_4.zip")
    ap.add_argument("--max-frame", type=int, default=6002)
    ap.add_argument("--fps", type=float, default=120.0)
    args = ap.parse_args()

    gt = load_gt(args.zip, args.max_frame)
    print(f"真值帧数（有球）: {len(gt)} / {args.max_frame}")

    # ---- test4 标定（core/geometry.py）----
    TABLE = ((522, 625), (1259, 625), (1439, 735), (348, 750))
    NEAR_EDGE_Y = max(p[1] for p in TABLE)      # 750：球台近边（图像最下方台面）
    FAR_EDGE_Y = min(p[1] for p in TABLE)       # 625：远边
    ROI_Y = (320, 920)
    ROI_X = (140, 1650)

    xs = [p[0] for p in gt.values()]
    ys = [p[1] for p in gt.values()]
    print("\n===== 真实球坐标包络 =====")
    print(f"x: min={min(xs):.0f} p1={pct(xs,1):.0f} 中位={statistics.median(xs):.0f} "
          f"p99={pct(xs,99):.0f} max={max(xs):.0f}   (ROI x=[{ROI_X[0]},{ROI_X[1]}])")
    print(f"y: min={min(ys):.0f} p1={pct(ys,1):.0f} 中位={statistics.median(ys):.0f} "
          f"p99={pct(ys,99):.0f} max={max(ys):.0f}   (台面远边y={FAR_EDGE_Y}, 近边y={NEAR_EDGE_Y}, ROI y=[{ROI_Y[0]},{ROI_Y[1]}])")

    below = [(f, y) for f, (_, y) in gt.items() if y > NEAR_EDGE_Y]
    print(f"\n真球出现在近边(y>{NEAR_EDGE_Y})以下的帧数: {len(below)} / {len(gt)}"
          f" ({100.0*len(below)/len(gt):.2f}%)")
    if below:
        by = sorted(y for _, y in below)
        print(f"  这些帧的 y 分布: min={by[0]:.0f} 中位={by[len(by)//2]:.0f} max={by[-1]:.0f}")
        top10 = below[:10]
        print(f"  样例(帧,y): {[(f, round(y)) for f, y in top10]}")

    # ---- 帧间位移（同一条轨迹内的连续观测）----
    frames = sorted(gt)
    jumps = []          # 连续帧位移（px/帧）
    speeds = []         # px/s
    run = []            # 当前连续段
    for f in frames:
        if run and f - run[-1] == 1:
            x0, y0 = gt[run[-1]]
            x1, y1 = gt[f]
            d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            jumps.append(d)
            speeds.append(d * args.fps)
        run.append(f)

    print(f"\n===== 真实球帧间运动（{args.fps:.0f}fps，连续帧对 {len(jumps)} 个）=====")
    print(f"位移 px/帧: p50={pct(jumps,50):.1f} p95={pct(jumps,95):.1f} "
          f"p99={pct(jumps,99):.1f} p99.9={pct(jumps,99.9):.1f} max={max(jumps):.1f}")
    print(f"速度 px/s : p50={pct(speeds,50):.0f} p95={pct(speeds,95):.0f} "
          f"p99={pct(speeds,99):.0f} max={max(speeds):.0f}")

    big = sorted(zip(jumps, frames), reverse=True)[:10]
    print(f"最大位移样例 (px, 帧): {[(round(d), f) for d, f in big]}")

    # 大位移发生时球的位置（是否在台面上方正常击球区）
    print("\n结论参数建议（零召回损失边界）：")
    print(f"  位移截断安全值: > {max(jumps)*1.3:.0f} px/帧@120fps "
          f"(实测最大 {max(jumps):.0f} px 的 1.3 倍)")
    y_safe = max(ys)
    print(f"  地板掩码安全线: y > {y_safe:.0f}（真球历史最低点，"
          f"近边以下 {y_safe - NEAR_EDGE_Y:.0f}px）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
