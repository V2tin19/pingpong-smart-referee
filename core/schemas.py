"""
core/schemas.py
===============

统一数据契约（Data Contract）—— TTNet / TrackNet / YOLO 三分支共用。

本模块严格对齐组长下发的《TTNet、TrackNet、YOLO 共用计分状态机方案》：

* §7.1  :class:`FrameObservation`  统一逐帧观测（**适配器**的输出）
* §7.3  漏检语义：``visible=False`` 且 center / bbox / confidence 必须全为 ``None``，
        不允许填上一次的位置，也不允许用 ``(0, 0)`` 表示丢失
* §7.4  :class:`RefereeEvent`      事件处理层的输出（**计分状态机**的输入）

设计约束
--------
1. **零第三方依赖**：仅用标准库。三条分支的 torch / opencv 版本各不相同，
   但都必须在自己的环境里 ``import core.schemas`` 成功，才能交换数据。
2. **契约自校验**：非法状态在构造时立即抛错，而不是等状态机判错分才发现。
   例如声明 ``visible=False`` 却塞了一个坐标，构造即 ``ValueError``。
3. **可序列化**：``to_dict()`` 的 JSON 形状与 §7.1 逐字段一致，可直接跨进程、
   跨模型、跨语言（C++/嵌入式端）交换。

坐标约定（§7.1）
----------------
统一使用**原始图像像素坐标**：左上角为原点，x 向右为正，y 向下为正。
撤销缩放、裁剪、letterbox 补边是**适配器的责任**，下游模块不需要知道模型
输入尺寸。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Source",
    "Side",
    "EventType",
    "BallObservation",
    "EventScores",
    "FrameObservation",
    "RefereeEvent",
    "make_event_id",
]


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------
class Source(str, Enum):
    """观测来源，对应组长方案 §7.1 的 ``source`` 字段。"""

    YOLO = "yolo"
    TRACKNET = "tracknet"
    TTNET = "ttnet"


class Side(str, Enum):
    """球台两侧，对应 §7.4 的 ``side`` 字段。

    以**摄像机画面**为基准：``A`` = 画面左侧半台，``B`` = 画面右侧半台。
    注意这与"发球方/接发方"不是一回事——主客关系由状态机结合发球轮次决定，
    适配器与事件层只负责报告物理位置在哪一侧。
    """

    A = "A"
    B = "B"
    UNKNOWN = "UNKNOWN"


class EventType(str, Enum):
    """物理事件类型，对应组长方案 §7.4 的 ``type`` 字段。

    组长方案 §7.4 给出的示例是 ``"BOUNCE"``，注释写"落台、击球、擦网等"，
    属于开放式列举。这里把三分支实际需要的事件收敛成 4 类：

    * ``BOUNCE``      球落台（垂直速度由下转上的拐点，且落点在台面内）
    * ``RACKET_HIT``  球拍击球（水平速度反向）
    * ``NET_TOUCH``   擦网 / 触网绳
    * ``OUT``         出界（落地、飞出画面、越过底线）

    .. note::
       ``OUT`` 是**本分支补充**的（组长文档只列举到"擦网等"），
       完整事件枚举需要与组长及另两条分支对齐后确认。
    """

    BOUNCE = "BOUNCE"
    RACKET_HIT = "RACKET_HIT"
    NET_TOUCH = "NET_TOUCH"
    OUT = "OUT"


_VALID_SOURCES = {s.value for s in Source}
_VALID_SIDES = {s.value for s in Side}
_VALID_EVENT_TYPES = {s.value for s in EventType}


# ---------------------------------------------------------------------------
# 内部校验工具
# ---------------------------------------------------------------------------
def _as_float(value: Any, name: str) -> float:
    """把 value 转成有限浮点数，否则抛错。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}: {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _as_point(value: Any, name: str) -> Tuple[float, float]:
    """把长度 2 的序列规范化为 ``(x, y)`` 浮点元组。"""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a 2-element sequence (x, y), got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"{name} must have exactly 2 elements (x, y), got {len(value)}")
    return (_as_float(value[0], f"{name}[0]"), _as_float(value[1], f"{name}[1]"))


def _as_bbox(value: Any, name: str) -> Tuple[float, float, float, float]:
    """把长度 4 的序列规范化为 ``(x1, y1, x2, y2)``，并检查顺序合法。"""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a 4-element sequence (x1, y1, x2, y2), got {type(value).__name__}")
    if len(value) != 4:
        raise ValueError(f"{name} must have exactly 4 elements (x1, y1, x2, y2), got {len(value)}")
    x1, y1, x2, y2 = (_as_float(v, f"{name}[{i}]") for i, v in enumerate(value))
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{name} must satisfy x1 <= x2 and y1 <= y2, got {(x1, y1, x2, y2)}")
    return (x1, y1, x2, y2)


def _as_confidence(value: Any, name: str) -> float:
    """置信度必须落在 [0, 1]。"""
    v = _as_float(value, name)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {v}")
    return v


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _as_source(value: Any) -> str:
    raw = value.value if isinstance(value, Source) else str(value).strip().lower()
    if raw not in _VALID_SOURCES:
        raise ValueError(
            f"source must be one of {sorted(_VALID_SOURCES)} (see §7.1), got {value!r}"
        )
    return raw


def _as_side(value: Any) -> str:
    raw = value.value if isinstance(value, Side) else str(value).strip().upper()
    if raw not in _VALID_SIDES:
        raise ValueError(f"side must be one of {sorted(_VALID_SIDES)} (see §7.4), got {value!r}")
    return raw


def _as_event_type(value: Any) -> str:
    raw = value.value if isinstance(value, EventType) else str(value).strip().upper()
    if raw not in _VALID_EVENT_TYPES:
        raise ValueError(
            f"event type must be one of {sorted(_VALID_EVENT_TYPES)} (see §7.4), got {value!r}"
        )
    return raw


# ---------------------------------------------------------------------------
# §7.1 球的观测
# ---------------------------------------------------------------------------
@dataclass
class BallObservation:
    """单帧的球观测。

    §7.3 的漏检语义由本类**强制实施**：一旦 ``visible=False``，那么
    ``center_px`` / ``bbox_xyxy`` / ``confidence`` 必须全部为 ``None``。
    这从机制上杜绝了"漏检时填上一次位置"和"用 (0, 0) 表示丢失"两种典型错误。

    Parameters
    ----------
    visible:
        本帧是否真的检测到球（用模型输出，而不是用跟踪器预测）。
    center_px:
        球心在**原始图像**中的像素坐标 ``(x, y)``；漏检时为 ``None``。
    bbox_xyxy:
        检测框 ``(x1, y1, x2, y2)``；模型不提供框（如 TrackNet 热图）时填 ``None``。
    confidence:
        模型原始分数；没有可靠分数时填 ``None``。
        注意：不同模型的分数含义不同，未经校准**不能**直接互相比较（§7.2）。
    """

    visible: bool
    center_px: Optional[Tuple[float, float]] = None
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.visible, bool):
            raise TypeError(f"visible must be a bool, got {type(self.visible).__name__}")

        if not self.visible:
            # §7.3：漏检时三个字段必须为空，不允许任何"占位值"
            leftovers = {
                "center_px": self.center_px,
                "bbox_xyxy": self.bbox_xyxy,
                "confidence": self.confidence,
            }
            filled = [k for k, v in leftovers.items() if v is not None]
            if filled:
                raise ValueError(
                    "§7.3 violation: when visible=False, "
                    f"{', '.join(filled)} must be None (do NOT carry over the previous "
                    "position and do NOT use (0, 0) to mean 'lost')."
                )
            return

        if self.center_px is None:
            raise ValueError("§7.1 violation: visible=True requires center_px")
        self.center_px = _as_point(self.center_px, "center_px")
        if self.bbox_xyxy is not None:
            self.bbox_xyxy = _as_bbox(self.bbox_xyxy, "bbox_xyxy")
        if self.confidence is not None:
            self.confidence = _as_confidence(self.confidence, "confidence")

    @classmethod
    def missing(cls) -> "BallObservation":
        """构造一个合规的"本帧没看到球"观测（§7.3）。"""
        return cls(visible=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visible": self.visible,
            "center_px": list(self.center_px) if self.center_px is not None else None,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BallObservation":
        # 跨分支兼容：TrackNet 分支交付的 schema 里带一个可选的 ``ball.predicted``
        # 字段，表示 center_px 是**轨迹推断位置**而非模型检出。
        #
        # 这里**显式拒绝**而不是静默忽略。原因：本方法只挑自己认识的键，未知键会被
        # 无声丢掉——那样一个"预测点"就会被当成真实检测混进轨迹，正是组长方案 §7.3
        # 要防的"被误认为真实检测"。宁可报错，也不要静默出错。
        #
        # 组长 §7.3 明确"轨迹预测由后续追踪模块完成"，因此预测值本就不应出现在观测里。
        # 本分支的做法是把预测点标记为 ``MotionState.PREDICTED`` 留在事件层的轨迹缓冲中，
        # 绝不写进 FrameObservation。
        if data.get("predicted"):
            raise ValueError(
                "ball.predicted=True is not accepted: a trajectory-inferred position must not be "
                "carried as a detection (§7.3). Keep predictions inside the trajectory/event layer "
                "and mark them there; FrameObservation.ball is for real detections only."
            )
        return cls(
            visible=bool(data["visible"]),
            center_px=data.get("center_px"),
            bbox_xyxy=data.get("bbox_xyxy"),
            confidence=data.get("confidence"),
        )


# ---------------------------------------------------------------------------
# §7.1 模型自带的事件预测分数
# ---------------------------------------------------------------------------
@dataclass
class EventScores:
    """模型自带的事件**候选分数**（不是已确认的事件，见 §7.5）。

    YOLO / TrackNet 没有这个能力时保持 ``None``；TTNet 填热图或分类分数。
    ``None`` 与 ``0.0`` 语义不同：``None`` = 该模型不提供此输出，
    ``0.0`` = 该模型明确认为不是事件。下游做多模型融合时必须区分这两者。
    """

    bounce: Optional[float] = None
    net_hit: Optional[float] = None

    def __post_init__(self) -> None:
        if self.bounce is not None:
            self.bounce = _as_confidence(self.bounce, "event_scores.bounce")
        if self.net_hit is not None:
            self.net_hit = _as_confidence(self.net_hit, "event_scores.net_hit")

    def to_dict(self) -> Dict[str, Any]:
        return {"bounce": self.bounce, "net_hit": self.net_hit}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EventScores":
        if not data:
            return cls()
        return cls(bounce=data.get("bounce"), net_hit=data.get("net_hit"))


# ---------------------------------------------------------------------------
# §7.1 统一逐帧观测
# ---------------------------------------------------------------------------
@dataclass
class FrameObservation:
    """适配器对**某一帧**的观测结果（组长方案 §7.1）。

    这是三条分支之间唯一需要对齐的东西：谁的模型都往这个结构里填，
    后面的轨迹/事件层与状态机就完全不用改。

    Parameters
    ----------
    frame_id:
        被预测帧在视频中的序号（从 0 开始）。
    timestamp_s:
        **被预测帧**的视频时间，不是推理完成时间（§7.1、§7.5）。
        连续帧模型（输入 3 帧预测中间帧）必须填中间帧的时间。
    source:
        ``"yolo"`` / ``"tracknet"`` / ``"ttnet"``。
    image_size:
        原始图像尺寸 ``[width, height]``，不是模型输入尺寸。
    ball:
        :class:`BallObservation`。
    event_scores:
        :class:`EventScores`，模型无此能力时留默认空值。

    Notes
    -----
    本类的字段集合**严格等于组长方案 §7.1**，是冻结接口：不增、不减、不改名。
    适配器的额外能力（例如 YOLO 一帧内的多个候选框）必须走**带外通道**，
    见 :meth:`core.base_adapter.BaseAdapter.candidates_for`，不得塞进本结构——
    否则三分支的"统一观测"会退化成各自扩展，违背组长方案 §1/§6 的初衷。
    """

    frame_id: int
    timestamp_s: float
    source: str
    image_size: Tuple[int, int]
    ball: BallObservation
    event_scores: EventScores = field(default_factory=EventScores)

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise TypeError(f"frame_id must be an int, got {type(self.frame_id).__name__}")
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {self.frame_id}")

        self.timestamp_s = _as_float(self.timestamp_s, "timestamp_s")
        if self.timestamp_s < 0:
            raise ValueError(f"timestamp_s must be >= 0, got {self.timestamp_s}")

        self.source = _as_source(self.source)

        if isinstance(self.image_size, (str, bytes)) or not isinstance(self.image_size, Sequence):
            raise TypeError("image_size must be a 2-element sequence [width, height]")
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must have exactly 2 elements, got {len(self.image_size)}")
        self.image_size = (
            _as_positive_int(self.image_size[0], "image_size[0] (width)"),
            _as_positive_int(self.image_size[1], "image_size[1] (height)"),
        )

        if not isinstance(self.ball, BallObservation):
            raise TypeError(f"ball must be BallObservation, got {type(self.ball).__name__}")
        if not isinstance(self.event_scores, EventScores):
            raise TypeError(f"event_scores must be EventScores, got {type(self.event_scores).__name__}")

    # -- 便捷属性 ---------------------------------------------------------
    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]

    @property
    def ball_visible(self) -> bool:
        return self.ball.visible

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """输出与组长方案 §7.1 逐字段一致的字典（键集合是冻结的）。"""
        return {
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "source": self.source,
            "image_size": [self.image_size[0], self.image_size[1]],
            "ball": self.ball.to_dict(),
            "event_scores": self.event_scores.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameObservation":
        return cls(
            frame_id=data["frame_id"],
            timestamp_s=data["timestamp_s"],
            source=data["source"],
            image_size=tuple(data["image_size"]),
            ball=BallObservation.from_dict(data["ball"]),
            event_scores=EventScores.from_dict(data.get("event_scores")),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "FrameObservation":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# §7.4 事件
# ---------------------------------------------------------------------------
def make_event_id(event_type: Any, sequence: int) -> str:
    """生成稳定的事件 ID，形如 ``bounce_00042``（§7.4 示例风格）。

    事件 ID 用于跨模型去重与幂等写入，必须在一次比赛内唯一。
    """
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise TypeError("sequence must be an int")
    if sequence < 0:
        raise ValueError("sequence must be >= 0")
    return f"{_as_event_type(event_type).lower()}_{sequence:05d}"


@dataclass
class RefereeEvent:
    """事件处理层输出的物理事件（组长方案 §7.4），也是状态机的唯一输入。

    Parameters
    ----------
    event_id:
        稳定唯一 ID，见 :func:`make_event_id`。
    timestamp_s:
        事件发生的视频时间（秒）。
    type:
        :class:`EventType` 之一。
    side:
        事件发生在哪一侧半台（:class:`Side`）；无法判定时用 ``UNKNOWN``，
        不要瞎猜——状态机会把 ``UNKNOWN`` 当作信息缺失处理。
    position_px:
        事件发生位置的原始图像像素坐标；不适用时 ``None``。
    confidence:
        事件置信度 [0, 1]。注意这是**事件级**置信度，由事件层融合得出，
        不是某个模型的原始分数（§7.2 明确不同模型分数不可直接比较）。
    sources:
        参与证据融合的模型列表，例如 ``["ttnet", "tracknet"]``。
    """

    event_id: str
    timestamp_s: float
    type: str
    side: str = Side.UNKNOWN.value
    position_px: Optional[Tuple[float, float]] = None
    confidence: Optional[float] = None
    sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        self.event_id = self.event_id.strip()

        self.timestamp_s = _as_float(self.timestamp_s, "timestamp_s")
        if self.timestamp_s < 0:
            raise ValueError(f"timestamp_s must be >= 0, got {self.timestamp_s}")

        self.type = _as_event_type(self.type)
        self.side = _as_side(self.side)

        if self.position_px is not None:
            self.position_px = _as_point(self.position_px, "position_px")
        if self.confidence is not None:
            self.confidence = _as_confidence(self.confidence, "confidence")

        if isinstance(self.sources, (str, bytes)):
            raise TypeError("sources must be a list of source names, not a bare string")
        self.sources = [_as_source(s) for s in self.sources]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp_s": self.timestamp_s,
            "type": self.type,
            "side": self.side,
            "position_px": list(self.position_px) if self.position_px is not None else None,
            "confidence": self.confidence,
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefereeEvent":
        return cls(
            event_id=data["event_id"],
            timestamp_s=data["timestamp_s"],
            type=data["type"],
            side=data.get("side", Side.UNKNOWN.value),
            position_px=data.get("position_px"),
            confidence=data.get("confidence"),
            sources=list(data.get("sources") or []),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "RefereeEvent":
        return cls.from_dict(json.loads(text))
