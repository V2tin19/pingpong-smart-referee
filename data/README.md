# data/ —— 输入数据

本目录存放**原始视频与官方标注**。大文件不入 git（见根目录 `.gitignore`）。

## 当前内容

| 文件 | 说明 |
|:---|:---|
| `test_1.mp4` / `test_1.zip` | OpenTTGames 测试视频 1 + 官方标注 |
| `test_2.mp4` / `test_2.zip` | 连续对拉视频，1920×1080 @ **120fps**（与 game2 同机位，已实测验证）+ 官方标注 |
| `test_3.mp4` / `test_3.zip` | OpenTTGames 测试视频 3 + 官方标注 |
| `test_4.mp4` / `test_4.zip` | 深绿台面机位（`docs/test4_evaluation_report.md` 评测对象）+ 官方标注 |
| `test_4_50s.mp4` | test_4 前 50 秒剪辑（6002 帧 @120fps），门控验证与感知降噪的标准切片 |
| `roboflow_ball_v7/` | Roboflow 开源球检测数据集（训练用）。注意报告 §一 结论：泛杂场景数据反而有害，训练只考虑与测试机位一致的数据 |

> 每个 `*.zip` 内含 `ball_markup.json`（逐帧球心真值）、`events_markup.json`
> （物理事件标签）与 `segmentation_masks/`。
> `test_game_2.avi` / `test_rally.mp4` / `test_plos.mp4` 仍在原始工作区
> `E:\2026Autumn\YOLO\table-tennis-ball-detection\` 下，未随本仓库迁移。

## 官方标注怎么用

`ball_markup.json` 是感知层评测与降噪参数的**实测依据**（需视觉环境）：

```powershell
$py = 'D:\Apps\miniconda\envs\pytorch\python.exe'

# 真值包络与速度分布 —— 地板掩码 / 位移截断的零召回损失安全边界
& $py tools\analyze_gt_envelope.py --zip data/test_4.zip

# 误检位置 / 置信度 / bbox 尺寸分析 —— 降噪手段的效果预估
& $py tools\analyze_fp_locations.py --zip data/test_4.zip --step 2

# 权重精度基准（Recall / FP / MAE，对照官方真值）
& $py tools\compare_weights_benchmark.py --conf 0.18

# 端到端（事件 + 计分 + 渲染比分 UI；权重在根目录 weights/）
& $py tests\run_end_to_end.py `
    --video data/test_4_50s.mp4 --weights weights/yolo11n_clean_best.pt `
    --preset test4 --output output/test4_referee.mp4
```

> `--preset` 机位预设见 `core/geometry.py`（test2 / game2 / rally / plos / test4）；
> 换新机位需重新标定。

## 仍待补：判分真值

官方标注只有**球心位置与弹跳/擦网标签**，没有"谁得分"——判分准确率
（例如门控版在 test_4 上判 0:1 是否正确）目前**无法量化**。自录评测集
需要按数据集文档 P017 补充事件级标注：落台 / 击球 / 出界 / 回合结束 / 得分方。
