# -*- coding: utf-8 -*-
"""
yolo_core/scripts/build_game1_dataset.py
=======================================

从 E:/2026Autumn/602Lab/game_1.mp4 与 game_1.zip 提取并构建 YOLO 乒乓球检测数据集：
1. 全量提取 6,799 帧标注球心数据（包含 6,723 真实球帧 + 76 遮挡/出界空标签帧）；
2. 引入速度自适应 BBox（依据相邻帧 dx, dy 动态扩展 BBox，覆盖 120fps 高速击球运动模糊）；
3. 挖掘 800 帧赛段间隔负样本（运动员持拍走动、试球准备、台面反光、白色球鞋，空标签训练抑噪）；
4. 按 Rally（连续回合）划分 Train / Val（85% / 15%），杜绝相邻帧数据穿越；
5. 统一裁剪至球台运动良态 ROI，与推理端 YOLOAdapter 输入尺度完全对齐。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


# 默认 ROI 裁剪框（覆盖 game_1 中 99.9% 的球心轨迹与整个比赛台面及近网高球区域）
DEFAULT_ROI = {
    "y_min": 140,
    "y_max": 860,
    "x_min": 60,
    "x_max": 1860,
}


def load_annotations(zip_path: str) -> Dict[int, Optional[Tuple[float, float]]]:
    """从 game_1.zip 读取 ball_markup.json，返回 {frame_id: (x, y) or None}。"""
    with zipfile.ZipFile(zip_path, "r") as z:
        raw = json.loads(z.read("ball_markup.json").decode("utf-8"))
    annotations: Dict[int, Optional[Tuple[float, float]]] = {}
    for k, v in raw.items():
        fid = int(k)
        if v and v.get("x", -1) >= 0 and v.get("y", -1) >= 0:
            annotations[fid] = (float(v["x"]), float(v["y"]))
        else:
            annotations[fid] = None
    return annotations


def group_into_rallies(
    annotated_frames: List[int], max_gap: int = 5
) -> List[List[int]]:
    """将标注帧按连续性切分为独立回合（Rallies）。"""
    if not annotated_frames:
        return []
    sorted_fids = sorted(annotated_frames)
    rallies: List[List[int]] = []
    current_rally: List[int] = [sorted_fids[0]]

    for fid in sorted_fids[1:]:
        if fid - current_rally[-1] <= max_gap:
            current_rally.append(fid)
        else:
            rallies.append(current_rally)
            current_rally = [fid]
    if current_rally:
        rallies.append(current_rally)
    return rallies


def sample_negative_frames(
    total_frames: int,
    annotated_set: Set[int],
    num_negatives: int = 800,
    margin_frames: int = 30,
) -> List[int]:
    """在无标注的非比赛对打间隙中均匀采样纯背景负样本。"""
    # 构造禁采区（标注帧前后各 30 帧以内不采，避免漏标球进入负样本）
    forbidden = set()
    for f in annotated_set:
        for offset in range(-margin_frames, margin_frames + 1):
            forbidden.add(f + offset)

    # 找出所有允许采样的候选帧
    candidates = [
        f for f in range(100, total_frames - 100)
        if f not in forbidden
    ]
    if len(candidates) <= num_negatives:
        return candidates

    # 均匀跨度采样
    step = len(candidates) / float(num_negatives)
    sampled = [candidates[int(i * step)] for i in range(num_negatives)]
    return sorted(sampled)


def compute_adaptive_bbox(
    curr_pt: Tuple[float, float],
    prev_pt: Optional[Tuple[float, float]],
    next_pt: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """根据前后帧估计的速度矢量计算自适应 BBox 尺寸 (w, h)（单位 px）。"""
    dx, dy = 0.0, 0.0
    if prev_pt is not None:
        dx = abs(curr_pt[0] - prev_pt[0])
        dy = abs(curr_pt[1] - prev_pt[1])
    elif next_pt is not None:
        dx = abs(curr_pt[0] - next_pt[0])
        dy = abs(curr_pt[1] - next_pt[1])

    # 基础球尺寸 ~16px，沿运动方向按速度分量自适应扩展，上限 55px
    w = max(18.0, min(55.0, 16.0 + 0.85 * dx))
    h = max(18.0, min(55.0, 16.0 + 0.85 * dy))
    return w, h


def build_dataset(
    video_path: str,
    zip_path: str,
    output_dir: str,
    roi: Dict[str, int] = DEFAULT_ROI,
    val_ratio: float = 0.15,
    num_negatives: int = 800,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 72)
    print("【数据集构建】从 OpenTTGames game_1 提取 YOLO 乒乓球训练集")
    print(f"视频路径: {video_path}")
    print(f"标注压缩包: {zip_path}")
    print(f"目标目录: {output_dir}")
    print(f"ROI 裁剪框: [{roi['y_min']}..{roi['y_max']}, {roi['x_min']}..{roi['x_max']}] "
          f"({roi['x_max']-roi['x_min']}x{roi['y_max']-roi['y_min']})")
    print("=" * 72)

    # 1. 解析标注与划分 Rally
    annotations = load_annotations(zip_path)
    total_annotated = len(annotations)
    positive_count = sum(1 for v in annotations.values() if v is not None)
    print(f"读取到 {total_annotated} 标注帧（其中有效球心帧 {positive_count}，空帧/遮挡 {total_annotated - positive_count}）")

    annotated_fids = sorted(annotations.keys())
    rallies = group_into_rallies(annotated_fids)
    print(f"聚合为 {len(rallies)} 个连续回合 (Rallies)")

    # 按回合随机拆分 Train 与 Val
    random.shuffle(rallies)
    n_val_rallies = int(len(rallies) * val_ratio)
    val_rallies = rallies[:n_val_rallies]
    train_rallies = rallies[n_val_rallies:]

    train_fids_set = {f for r in train_rallies for f in r}
    val_fids_set = {f for r in val_rallies for f in r}
    print(f"训练集包含 {len(train_rallies)} 个回合 ({len(train_fids_set)} 帧)")
    print(f"验证集包含 {len(val_rallies)} 个回合 ({len(val_fids_set)} 帧)")

    # 2. 采样负样本
    cap = cv2.VideoCapture(video_path)
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {video_total_frames}")

    negative_fids = sample_negative_frames(
        video_total_frames, set(annotated_fids), num_negatives=num_negatives
    )
    print(f"在赛段间隙采样了 {len(negative_fids)} 帧负样本（纯背景/走动/反光）")

    random.shuffle(negative_fids)
    n_val_neg = int(len(negative_fids) * val_ratio)
    val_neg_set = set(negative_fids[:n_val_neg])
    train_neg_set = set(negative_fids[n_val_neg:])

    # 建立目标帧分流字典：frame_id -> ("train" | "val", is_negative)
    target_dispatch: Dict[int, Tuple[str, bool]] = {}
    for f in train_fids_set:
        target_dispatch[f] = ("train", False)
    for f in val_fids_set:
        target_dispatch[f] = ("val", False)
    for f in train_neg_set:
        target_dispatch[f] = ("train", True)
    for f in val_neg_set:
        target_dispatch[f] = ("val", True)

    sorted_targets = sorted(target_dispatch.keys())
    max_target_fid = sorted_targets[-1]
    targets_set = set(sorted_targets)
    print(f"全量处理帧数: {len(sorted_targets)} 帧 (从帧 {sorted_targets[0]} 到 {max_target_fid})")

    # 3. 准备输出目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # 4. 单向高速顺序遍历视频提取
    roi_ymin = roi["y_min"]
    roi_ymax = roi["y_max"]
    roi_xmin = roi["x_min"]
    roi_xmax = roi["x_max"]
    crop_w = float(roi_xmax - roi_xmin)
    crop_h = float(roi_ymax - roi_ymin)

    extracted_counts = defaultdict(int)
    t0 = time.time()
    last_log_t = t0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_fid = 0

    while current_fid <= max_target_fid:
        if current_fid not in targets_set:
            # 高速跳帧（不解码完整像素）
            if not cap.grab():
                break
            current_fid += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        split, is_neg = target_dispatch[current_fid]
        crop = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

        lines = []
        if not is_neg:
            ball_pos = annotations.get(current_fid)
            if ball_pos is not None:
                gx, gy = ball_pos
                prev_pos = annotations.get(current_fid - 1)
                next_pos = annotations.get(current_fid + 1)
                bw, bh = compute_adaptive_bbox(ball_pos, prev_pos, next_pos)

                # 转换到 ROI 局部坐标
                cx_crop = gx - roi_xmin
                cy_crop = gy - roi_ymin

                # 仅保留在 ROI 内的目标
                if 0 <= cx_crop < crop_w and 0 <= cy_crop < crop_h:
                    norm_cx = cx_crop / crop_w
                    norm_cy = cy_crop / crop_h
                    norm_bw = bw / crop_w
                    norm_bh = bh / crop_h
                    lines.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")

        # 写入图像与标签
        img_name = f"g1_{current_fid:06d}.jpg"
        lbl_name = f"g1_{current_fid:06d}.txt"
        img_out = os.path.join(output_dir, "images", split, img_name)
        lbl_out = os.path.join(output_dir, "labels", split, lbl_name)

        cv2.imwrite(img_out, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        with open(lbl_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        extracted_counts[split] += 1
        current_fid += 1

        if time.time() - last_log_t > 5.0:
            last_log_t = time.time()
            done = extracted_counts["train"] + extracted_counts["val"]
            pct = done / float(len(sorted_targets)) * 100.0
            fps_val = current_fid / max(time.time() - t0, 1e-5)
            print(f"  进度: {done}/{len(sorted_targets)} 帧 ({pct:.1f}%) | 视频游标: 帧 {current_fid} | 速度: {fps_val:.1f} fps")

    cap.release()
    elapsed = time.time() - t0
    print("=" * 72)
    print("【提取完成】")
    print(f"总耗时: {elapsed:.1f} 秒")
    print(f"训练集图像: {extracted_counts['train']} 张")
    print(f"验证集图像: {extracted_counts['val']} 张")

    # 5. 生成 data.yaml
    abs_data_path = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""# YOLO Table Tennis Dataset (Game 1 Clean & Dense)
path: {abs_data_path}
train: images/train
val: images/val

names:
  0: ball
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"数据集配置文件已生成: {yaml_path}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="E:/2026Autumn/602Lab/game_1.mp4")
    parser.add_argument("--zip", default="E:/2026Autumn/602Lab/game_1.zip")
    parser.add_argument("--output", default="data/game1_dataset")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--negatives", type=int, default=800)
    args = parser.parse_args()

    build_dataset(
        args.video,
        args.zip,
        args.output,
        val_ratio=args.val_ratio,
        num_negatives=args.negatives,
    )
