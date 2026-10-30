"""
Train YOLO11n on Roboflow v7 Table Tennis Dataset (2,876 images)
"""
import os
from ultralytics import YOLO

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_yaml = os.path.join(base_dir, "data", "roboflow_ball_v7", "data.yaml")
    
    # 优先使用本地已有的 yolo11n.pt 底座，避免重复联网下载
    base_weight = "E:/2026Autumn/YOLO/yolo11n.pt"
    if not os.path.exists(base_weight):
        base_weight = "yolo11n.pt"

    print("=" * 60)
    print(f"Loading Base YOLO11n: {base_weight}")
    print(f"Dataset Configuration: {data_yaml}")
    print("=" * 60)

    model = YOLO(base_weight)

    results = model.train(
        data=data_yaml,
        epochs=40,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project=os.path.join(base_dir, "yolo_core", "runs"),
        name="yolo11n_roboflow_v7",
        exist_ok=True,
        verbose=True
    )
    
    best_pt = os.path.join(base_dir, "yolo_core", "runs", "yolo11n_roboflow_v7", "weights", "best.pt")
    print("=" * 60)
    print("Training Complete!")
    print(f"Best weights saved to: {best_pt}")
    print("=" * 60)

if __name__ == '__main__':
    train()
