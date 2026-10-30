"""
Train Clean YOLO11n on Noise-Free Mixed Dataset
"""
import os
from ultralytics import YOLO

def train():
    data_yaml = os.path.abspath("mixed_tt_dataset/data.yaml")
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=data_yaml,
        epochs=35,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project="runs/detect",
        name="yolo11n_clean",
        exist_ok=True,
        hsv_h=0.4,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.6,
        mosaic=1.0,
        verbose=True
    )
    print("Training finished! Best weights saved to runs/detect/yolo11n_clean/weights/best.pt")

if __name__ == '__main__':
    train()
