#!/usr/bin/env bash
# 在 DGX_station_2 隔离工作区 ~/pp_aug_ws 内启动增广重训。
# 仅用物理 GPU 1,2,3（GPU0 留给容器内其他任务）；日志/输出全部落在本工作区。
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1,2,3
nohup ./venv/bin/python scripts/train_game1_aug.py \
  --data data/game1_dataset_aug/data.yaml \
  --base "$PWD/yolo11n.pt" \
  --device 0,1,2 --batch 48 --workers 8 \
  --project "$PWD/runs" --name yolo11n_game1_aug_v1 \
  > logs/train.log 2>&1 &
echo "TRAIN_PID=$!"