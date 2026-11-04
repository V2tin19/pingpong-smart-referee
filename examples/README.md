# examples/ 示例观测数据

本目录包含轻量级的逐帧观测样例（JSONL 格式），符合组长《TTNet、TrackNet、YOLO 共用计分状态机方案》§7.1 数据契约。

用于**免模型、免视频、快速验证裁判状态机逻辑**。

---

## 文件列表

| 文件名 | 来源 | 帧数 | 说明 |
| :--- | :--- | :--- | :--- |
| `sample_tracknet_600f.jsonl` | TrackNet 分支 | 600 帧 | 1080p@120fps 慢动作对拉视频实测观测（含 15 个击球/反弹事件）。**门控版回归样例**：旧版曾在帧 111 误判 `SERVE_FAULT` 送出 0:1（曾印在本 README），门控 1 上线后正确判 0:0 |
| `sample_yolo_300f.jsonl` | YOLO 分支 | 300 帧 | 1080p 比赛对拉实测观测（含 20 个物理事件）。同为从回合中段剪出的剪辑（无发球序列），门控版正确判 **0:0**——证据不足默认不判分 |

---

## 快速自测

在仓库根目录下直接运行：

```bash
# 测试 TrackNet 观测流
python run_referee.py --jsonl examples/sample_tracknet_600f.jsonl --preset test2

# 测试 YOLO 观测流
python run_referee.py --jsonl examples/sample_yolo_300f.jsonl --preset game2
```
