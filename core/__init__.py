"""
core —— 乒乓球视觉智能裁判的**共用**核心层（三分支共享）。

模块划分
--------
* :mod:`core.schemas`        统一数据契约 ``FrameObservation`` / ``RefereeEvent``
* :mod:`core.base_adapter`   适配器抽象基类（模板方法 + 契约自校验）
* :mod:`core.geometry`       球台标定与 ROI（与模型无关的静态几何）

``import core`` 只加载 :mod:`core.schemas`（**纯标准库**，零第三方依赖），
以保证 TTNet、TrackNet、YOLO 三条分支在各自环境里都能直接交换契约数据；
事件层与几何模块（``trajectory_processor`` / ``homography``）按需单独导入，
它们依赖 opencv。
"""

from .schemas import (
    BallObservation,
    EventScores,
    EventType,
    FrameObservation,
    RefereeEvent,
    Side,
    Source,
    make_event_id,
)

__all__ = [
    "BallObservation",
    "EventScores",
    "EventType",
    "FrameObservation",
    "RefereeEvent",
    "Side",
    "Source",
    "make_event_id",
]

__version__ = "0.1.0"
