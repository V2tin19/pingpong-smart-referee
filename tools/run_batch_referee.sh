#!/usr/bin/env bash
# 批量渲染 7 个测试视频的状态机 HUD 视频（本机 GPU）。
# 降质加速：imgsz 768 + 720p(0.667) + 60fps。状态机仍逐帧运行，仅渲染采样。
# 机位：test_1/test_7 无精确标定，回退 game2（近似，比分仅供参考）；其余用对应预设。
set -u
cd "$(dirname "$0")/.."
PY="${PY:-D:/Apps/miniconda/envs/pytorch/python.exe}"
IDS="${IDS:-1 2 3 4 5 6 7}"
OUT=output/batch_20261113
mkdir -p "$OUT/videos" "$OUT/logs"
declare -A PRESET=( [1]=game2 [2]=test2 [3]=test3 [4]=test4 [5]=test3 [6]=test3 [7]=game2 )
for i in $IDS; do
  echo "=== referee test_$i (preset ${PRESET[$i]}) ==="
  "$PY" -u run_referee.py --video "data/test_$i.mp4" \
    --weights weights/yolo11n_game1_aug_best.pt \
    --preset "${PRESET[$i]}" --imgsz 768 \
    --render-scale 0.667 --render-fps 60 \
    --output "$OUT/videos/test${i}_referee.mp4" \
    > "$OUT/logs/test${i}_referee.log" 2>&1 || echo "test_$i RENDER_FAILED"
  tail -n 6 "$OUT/logs/test${i}_referee.log"
done
echo REFEREE_ALL_DONE