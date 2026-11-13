# -*- coding: utf-8 -*-
"""在离线增广后的 game1 数据集上重训（路线A 批准超参）。

数据：data/game1_dataset_aug（train 9759 = 6509 原 + 3250 增广；val 1090 干净）。
增广已在离线完成（速度方向模糊/copy-paste/光照），故在线 mixup/multi_scale/copy_paste 保持 0
（copy_paste 对纯 bbox 为空转）。mosaic 保留 + close_mosaic 收尾。
设备可配：实验室 3×3090 用 --device 0,1,2（或单卡 0）。
"""
from __future__ import annotations

import argparse
import os

from ultralytics import YOLO


def train(data_yaml, base_model, epochs, imgsz, batch, device, workers, name, project_dir):
    candidate_bases = [base_model, os.path.join("E:/2026Autumn/YOLO", base_model),
                       os.path.join("weights", "yolo11n_clean_best.pt")]
    actual_base = next((c for c in candidate_bases if os.path.isfile(c)), base_model)
    print(f"base={actual_base} data={os.path.abspath(data_yaml)} device={device}")
    model = YOLO(actual_base)
    model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=workers,
        project=os.path.abspath(project_dir), name=name, exist_ok=True,
        cos_lr=True, patience=10, save_period=5, amp=True,
        # 批准超参：离线增广已承担模糊/贴图/光照，在线仅保留几何/色彩/拼接
        mosaic=1.0, close_mosaic=10, scale=0.3,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        fliplr=0.5, flipud=0.0, degrees=0.0,
        mixup=0.0, multi_scale=0.0, copy_paste=0.0,
        verbose=True,
    )
    best = os.path.join(project_dir, name, "weights", "best.pt")
    print("best:", os.path.abspath(best))
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/game1_dataset_aug/data.yaml")
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="yolo11n_game1_aug_v1")
    ap.add_argument("--project", default="yolo_core/runs")
    a = ap.parse_args()
    train(a.data, a.base, a.epochs, a.imgsz, a.batch, a.device, a.workers, a.name, a.project)