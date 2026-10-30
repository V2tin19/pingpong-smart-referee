# 乒乓球视觉智能裁判系统 · 数据契约规范

**标准依据**：实验室《TTNet、TrackNet、YOLO 共用计分状态机方案》第 7 节（硬性遵守）。  
**设计目标**：统一 YOLO、TrackNet、TTNet 以及未来任何新增硬件传感器的数据格式，使得下游轨迹处理与计分状态机完全独立于前端感知技术。

---

## 一、感知层输出：统一逐帧观测（FrameObservation, §7.1）

感知算法（或边缘计算模块）必须为视频的每一帧输出如下格式的字典或 JSONL 行：

```json
{
  "frame_id": 1234,
  "timestamp_s": 10.283,
  "source": "yolo",
  "image_size": [1920, 1080],
  "ball": {
    "visible": true,
    "center_px": [850.0, 420.0],
    "bbox_xyxy": [844.0, 414.0, 856.0, 426.0],
    "confidence": 0.92
  },
  "event_scores": {
    "bounce": null,
    "net_hit": null
  }
}
```

### 1. 字段详细定义
| 字段 | 类型 | 取值范围 | 说明 |
| :--- | :--- | :--- | :--- |
| `frame_id` | `int` | $\ge 0$ | 视频或图像流的帧序号，**从 0 开始计数** |
| `timestamp_s` | `float` | $\ge 0.0$ | **被预测帧**的视频物理时间（秒），计算公式：`frame_id / fps` |
| `source` | `str` | `"yolo" \| "tracknet" \| "ttnet"` | 数据来源模型标记 |
| `image_size` | `[int, int]` | `[w, h]` | 原始图像的分辨率（宽高），禁止填模型裁剪尺寸 |
| `ball.visible` | `bool` | `true \| false` | 该帧是否检测到小球 |
| `ball.center_px` | `[float, float]` | `[x, y]` 或 `null` | **原始分辨率下**的小球中心像素坐标（左上角为原点） |
| `ball.bbox_xyxy` | `[float, ...]` | `[x1, y1, x2, y2]` 或 `null` | 小球边界框。TrackNet 等无框模型填 `null` |
| `ball.confidence`| `float` | $0.0 \sim 1.0$ 或 `null` | 模型原始检出置信度 |
| `event_scores` | `dict` | `bounce` / `net_hit` | 具备端到端事件检测能力的模型填分数，否则填 `null` |

### 2. 三条硬性校验规则（违反即在构建时抛异常）
1. **漏检不得填位**：当 `visible=false` 时，`center_px`、`bbox_xyxy`、`confidence` **必须全为 `null`**。严禁沿用上一帧坐标，严禁填 `(0, 0)`（`(0,0)` 是图像合法坐标，下游无法区分）。
2. **坐标原图还原**：`center_px` 必须撤销任何 ROI 裁剪、Letterbox 缩放，还原到真实相机原始分辨率下。
3. **带外多候选通道**：一帧若检出多个候选框，主观测 `ball` 只保留置信度最高的目标；其余候选走适配器带外通道 `adapter.candidates_for(frame_id)`，严禁擅自修改 §7.1 的基础字段集合。

---

## 二、事件层输出：物理裁判事件（RefereeEvent, §7.4）

轨迹处理器在平滑并判定小球运动特征后，输出纯粹的**物理事件**，作为计分状态机的输入：

```json
{
  "event_id": "bounce_00042",
  "timestamp_s": 10.283,
  "type": "BOUNCE",
  "side": "A",
  "position_px": [850.0, 420.0],
  "confidence": 0.90
}
```

### 字段说明
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `event_id` | `str` | 全局唯一自增事件 ID，形如 `bounce_00001`, `racket_hit_00002` |
| `timestamp_s` | `float` | 物理事件发生的精确时间戳（秒） |
| `type` | `str` | 事件枚举：`BOUNCE`（落台）、`RACKET_HIT`（球拍击球）、`NET_TOUCH`（触网）、`OUT`（出界） |
| `side` | `str` | 物理半台归属：`"A"`（画面左侧半台）、`"B"`（画面右侧半台）、`"UNKNOWN"` |
| `position_px` | `[float, float]` | 事件触发时的 2D 像素位置 |
| `confidence` | `float` | **事件级**综合置信度（结合轨迹连续度与几何投影综合得出） |
