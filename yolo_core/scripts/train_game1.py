# -*- coding: utf-8 -*-
"""
yolo_core/scripts/train_game1.py
===============================

在 game_1 高质量比赛数据集上训练高稳定性 YOLO 乒乓球检测模型：
1. 继承 yolo11n.pt / yolo11s.pt 预训练底座；
2. 采用 imgsz=960 高分辨率训练，契合小球高精定位与推理端 ROI 尺度；
3. 针对高速运动模糊与小目标优化数据增强（合理 scale，禁用颠倒翻转，收尾阶段关闭 mosaic）；
4. 产出高精度权重存入 weights/ 供裁判系统 run_referee.py 直接装载。
"""

from __future__ import annotations

import argparse
import os
import sys

from ultralytics import YOLO


def train(
    data_yaml: str = "data/game1_dataset/data.yaml",
    base_model: str = "yolo11n.pt",
    epochs: int = 40,
    imgsz: int = 960,
    batch: int = 16,
    project_dir: str = "yolo_core/runs",
    name: str = "yolo11n_game1",
    device: str = "0",
) -> str:
    # 优先寻找本地预训练底座，避免外网下载
    candidate_bases = [
        base_model,
        os.path.join("E:/2026Autumn/YOLO", base_model),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "yolo11n_clean_best.pt"),
    ]
    actual_base = base_model
    for cand in candidate_bases:
        if os.path.isfile(cand):
            actual_base = cand
            break

    print("=" * 72)
    print("【YOLO 乒乓球检测模型训练】")
    print(f"基础底座: {actual_base}")
    print(f"数据配置: {os.path.abspath(data_yaml)}")
    print(f"训练周期: {epochs} epochs | 图像尺度: {imgsz}px | Batch Size: {batch}")
    print(f"运行设备: cuda:{device}")
    print("=" * 72)

    model = YOLO(actual_base)

    results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        project=os.path.abspath(project_dir),
        name=name,
        exist_ok=True,
        # 针对小目标与高速球定制的数据增广与优化调度
        cos_lr=True,       # 余弦退火学习率平滑衰减
        patience=10,       # 10轮无提升早停保护
        save_period=5,     # 每5轮保存快照
        fliplr=0.5,        # 乒乓球台左右严格对称，等同数据量翻倍
        flipud=0.0,        # 严禁上下翻转（地面不应出现在上方）
        degrees=0.0,       # 固定机位无需大角度旋转
        scale=0.3,         # 抑制过度缩放（防止小球被缩至亚像素丢失）
        mosaic=1.0,        # 前期拼接学习复杂上下文
        close_mosaic=10,   # 最后 10 轮关闭 mosaic，收敛至真实连续比赛场景
        hsv_h=0.015,       # 适度色彩抖动适应光照变化
        hsv_s=0.5,
        hsv_v=0.3,
        verbose=True,
    )

    best_pt = os.path.join(project_dir, name, "weights", "best.pt")
    print("=" * 72)
    print("【训练完成】")
    print(f"最优权重保存至: {os.path.abspath(best_pt)}")
    print("=" * 72)
    return best_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/game1_dataset/data.yaml")
    parser.add_argument("--base", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="yolo11n_game1")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        base_model=args.base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
