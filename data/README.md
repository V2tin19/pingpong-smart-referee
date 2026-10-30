# data/ —— 输入数据

本目录存放**原始视频与标注**。大文件不入 git（见根目录 `.gitignore`）。

## 当前内容

| 文件 | 说明 |
|:---|:---|
| `test_2.mp4` | 连续对拉视频，1920×1080 @ **120fps**，3600 帧 / 30 秒。与 `test_game_2.avi` 同机位（已实测验证） |

> 其余测试视频仍在原始工作区 `E:\2026Autumn\YOLO\table-tennis-detection\` 下，未随本仓库迁移：
> `test_game_2.avi` 在 `videos\`，`test_rally.mp4` / `test_plos.mp4` 在 `test\`
> （两处子目录不同，找视频时注意）。

## 待补：test_2 的标注（真值）

**目前 `test_2.mp4` 没有真值**，因此只能做定性检查，**无法给出判分准确率**。
现有标注文件 `test_dataset_info.csv` 只覆盖 game_id 0 / 2 / 3，其中 game_id 2 就是
`test_game_2.avi`。

标注到位后放到本目录，命名建议 `test_2_annotations.csv`。

### 期望格式（与 OpenTTGames 现有标注一致）

```csv
game_id,frame_id,file_name,a,b,c,d,mean
2,88,88_2.png,0.29322916666666665,0.46296296296296297,0.00625,0.011111111111111112,70.077
```

| 列 | 含义 |
|:---|:---|
| `game_id` | 比赛编号（test_2 需给一个**未被占用**的编号；现有为 0 / 2 / 3） |
| `frame_id` | 帧序号，**从 0 开始**，与统一观测的 `frame_id` 对齐 |
| `file_name` | 可留空或填占位名 |
| `a` | 球心 **x 归一化坐标**（0~1，乘以图像宽即得像素 x） |
| `b` | 球心 **y 归一化坐标**（0~1，乘以图像高即得像素 y） |
| `c`, `d` | 归一化框宽/框高，可留空 |
| `mean` | 可留空 |

**注意**：
* 坐标是**归一化**的，评测脚本会按 1920×1080 还原；若标注来自其他分辨率，请说明。
* 只在**有球**的帧写行；漏检帧不写行（评测脚本按"该帧无真值"处理）。

### 标注到位后怎么跑

```powershell
$py = 'D:\Apps\miniconda\envs\pytorch\python.exe'

# 感知层指标（Recall / Precision / MAE / RMSE）
& $py tests\verify_adapter_benchmark.py `
    --video   data/test_2.mp4 `
    --gt      data/test_2_annotations.csv `
    --weights yolo_core/weights/yolo11n_clean_best.pt `
    --preset  test2 `
    --game-id <你给的编号> `
    --test-split-start 0

# 端到端（事件 + 计分 + 渲染比分 UI）
& $py tests/run_end_to_end.py `
    --video   data/test_2.mp4 `
    --weights yolo_core/weights/yolo11n_clean_best.pt `
    --preset  test2 `
    --output  output/output_referee_test2.mp4
```

> `--preset test2` 复用 game2 的球台标定（已实测确认两视频同机位）。
> 若换新素材，需要重新标定——见 `core/geometry.py` 的 `TableCalibration`。

### 若要评测**判分准确率**

还需要**事件级真值**（落台 / 击球 / 出界 / 回合结束 / 得分方），
球位置标注不足以评估计分。建议按数据集文档的建议记录这五类标签。
