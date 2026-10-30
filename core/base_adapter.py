"""
core/base_adapter.py
====================

模型适配器抽象基类 —— 三分支的统一入口。

组长方案的核心分工是：**适配器只回答"模型看到了什么"**，绝不输出"谁得分"。
本基类用模板方法（Template Method）把这条规矩做成**机制**而不是**约定**：

* 子类只实现 :meth:`BaseAdapter._predict_frame`（纯感知逻辑）；
* 对外暴露的 :meth:`predict_frame` 由基类统一完成：
  时间戳单调性检查、返回类型校验、``source`` 一致性校验、
  ``frame_id`` / ``timestamp_s`` 回显校验、帧计数。

这样一来，任何适配器只要写错"把预测值当检测值"或者"时间戳对应错帧"
这类典型 bug（§7.3 / §7.5），都会在开发阶段立刻抛错，而不是等状态机判错分。
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schemas import BallObservation, EventScores, FrameObservation, Source

__all__ = ["BaseAdapter"]


class BaseAdapter(abc.ABC):
    """所有感知模型适配器的基类。

    子类必须实现
    ------------
    ``source``
        模型标识，只能是 ``"yolo"`` / ``"tracknet"`` / ``"ttnet"``。
    ``_predict_frame(frame, frame_id, timestamp_s)``
        纯感知：输入一帧 BGR 图像，输出该帧的 :class:`FrameObservation`。

    不该做的事
    ----------
    * 不要输出比分、不要调用状态机、不要判断"谁失分"；
    * 不要在漏检时填充上一帧位置或用 ``(0, 0)`` 占位（§7.3，基类会拦）；
    * 不要把跟踪器的预测位置伪装成检测结果（预测属于事件处理层）。
    """

    def __init__(self, *, strict_time: bool = False) -> None:
        """
        Parameters
        ----------
        strict_time:
            为 ``True`` 时，若某帧的 ``timestamp_s`` 小于上一帧则直接抛错。
            单次顺序播放视频时应开启；视频循环、跳转回放等场景可以关闭。
        """
        self._strict_time = bool(strict_time)
        self._last_timestamp: Optional[float] = None
        self._frames_processed = 0
        self.last_inference_ms: Optional[float] = None

    # -- 子类必须提供 -----------------------------------------------------
    @property
    @abc.abstractmethod
    def source(self) -> str:
        """模型标识（``"yolo"`` / ``"tracknet"`` / ``"ttnet"``）。"""

    @abc.abstractmethod
    def _predict_frame(self, frame: np.ndarray, frame_id: int, timestamp_s: float) -> FrameObservation:
        """纯感知实现：产出该帧的观测。不要在这里做任何判分。"""

    # -- 对外模板方法 -----------------------------------------------------
    def predict_frame(self, frame: np.ndarray, frame_id: int, timestamp_s: float) -> FrameObservation:
        """对**一帧**做观测，返回经过契约校验的 :class:`FrameObservation`。

        Parameters
        ----------
        frame:
            BGR 图像（``numpy.ndarray``，``H x W x 3``）。
        frame_id:
            被预测帧的序号。
        timestamp_s:
            **被预测帧**的视频时间（秒），不是推理完成的墙钟时间。
            连续帧模型（如输入三帧预测中间帧）必须传中间帧的时间（§7.5）。

        Raises
        ------
        TypeError
            输入不是图像，或子类返回了错误类型。
        ValueError
            时间戳倒退（``strict_time=True`` 时），或子类返回的观测
            与输入帧不一致（时间对齐错误，§7.5）。
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy array (BGR image), got {type(frame).__name__}")
        if frame.ndim not in (2, 3):
            raise ValueError(f"frame must be HxW or HxWxC, got shape {frame.shape}")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise TypeError(f"frame_id must be an int, got {type(frame_id).__name__}")
        if frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {frame_id}")
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)):
            raise TypeError(f"timestamp_s must be a real number, got {type(timestamp_s).__name__}")
        timestamp_s = float(timestamp_s)

        if self._strict_time and self._last_timestamp is not None and timestamp_s < self._last_timestamp:
            raise ValueError(
                f"timestamp_s went backwards: {timestamp_s} < {self._last_timestamp} "
                f"(adapter={self.source}, frame_id={frame_id}). "
                "Pass strict_time=False for looping/seeking playback."
            )

        observation = self._predict_frame(frame, frame_id, timestamp_s)
        self._validate_observation(observation, frame, frame_id, timestamp_s)

        self._last_timestamp = timestamp_s
        self._frames_processed += 1
        return observation

    # -- 供子类使用的工具 -------------------------------------------------
    def candidates_for(self, frame_id: int) -> List[BallObservation]:
        """带外通道：返回该帧的**全部可见候选**（可选能力，默认无）。

        默认返回空列表，含义是"本适配器每帧只给一个球观测，请直接使用
        :attr:`FrameObservation.ball`"。一帧可能检出多个目标（如 YOLO 的多个
        检测框）的适配器可以覆盖本方法，按置信度降序给出全部候选。

        为什么放在这里而不是放进 :class:`FrameObservation`
        ---------------------------------------------------
        组长方案 §7.1 的字段集合是**冻结接口**，适配器之间的能力差异不应污染
        统一观测结构——否则三分支会各自加字段，"统一观测"退化成各说各话
        （违背 §1/§6 的初衷）。因此多候选这类"部分模型才有的能力"走带外通道，
        消费方按需取用，取不到就退回 ``ball``::

            cands = adapter.candidates_for(frame_id) or (
                [obs.ball] if obs.ball.visible else []
            )

        ``frame_id`` 必须与最近一次 :meth:`predict_frame` 的入参一致；
        不一致说明调用方在乱序取用，返回空列表并由调用方回退到 ``ball``。
        """
        return []

    def missing_observation(self, frame_id: int, timestamp_s: float, image_size: Tuple[int, int]) -> FrameObservation:
        """构造一个完全合规的"本帧没看到球"观测（§7.3），避免手写占位值。"""
        return FrameObservation(
            frame_id=frame_id,
            timestamp_s=float(timestamp_s),
            source=self.source,
            image_size=image_size,
            ball=BallObservation.missing(),
            event_scores=EventScores(),
        )

    def reset(self) -> None:
        """重置跨帧状态（换视频、跳转、开始新回合时调用）。

        无状态适配器（如纯检测的 YOLO）可以什么都不做。
        """
        self._last_timestamp = None
        self._frames_processed = 0
        self.last_inference_ms = None

    def close(self) -> None:
        """释放模型/显存等资源，默认无操作。"""

    # -- 元信息 -----------------------------------------------------------
    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def last_timestamp(self) -> Optional[float]:
        return self._last_timestamp

    def describe(self) -> Dict[str, Any]:
        """适配器的自描述，供基准脚本与日志记录（不含任何判分信息）。"""
        return {
            "source": self.source,
            "class": type(self).__name__,
            "frames_processed": self._frames_processed,
            "strict_time": self._strict_time,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} source={self.source!r} frames={self._frames_processed}>"

    # -- 内部校验 ---------------------------------------------------------
    def _validate_observation(
        self,
        observation: Any,
        frame: np.ndarray,
        frame_id: int,
        timestamp_s: float,
    ) -> None:
        """校验子类返回值与输入帧的一致性（§7.5 时间对齐）。"""
        if not isinstance(observation, FrameObservation):
            raise TypeError(
                f"{type(self).__name__}._predict_frame must return FrameObservation, "
                f"got {type(observation).__name__}"
            )

        # 1) 来源必须与适配器自身一致，防止把 TTNet 的输出挂到 YOLO 名下
        if observation.source != Source(self.source).value:
            raise ValueError(
                f"observation.source={observation.source!r} does not match adapter.source="
                f"{self.source!r} (adapter={type(self).__name__})"
            )

        # 2) 帧身份必须回显输入，防止"预测了别的帧"这种时间对齐错误（§7.5）
        if observation.frame_id != frame_id:
            raise ValueError(
                f"observation.frame_id={observation.frame_id} does not echo the requested "
                f"frame_id={frame_id} (adapter={type(self).__name__}). "
                "Continuous-frame models must report the frame they actually predicted."
            )
        if abs(observation.timestamp_s - timestamp_s) > 1e-6:
            raise ValueError(
                f"observation.timestamp_s={observation.timestamp_s} does not echo the requested "
                f"timestamp_s={timestamp_s} (adapter={type(self).__name__})."
            )

        # 3) 图像尺寸必须与真实帧一致，防止裁剪/缩放后的尺寸泄漏到下游
        h, w = frame.shape[:2]
        if observation.image_size != (w, h):
            raise ValueError(
                f"observation.image_size={observation.image_size} must equal the input frame size "
                f"{(w, h)}; adapters must report the ORIGINAL frame size, not the model input size "
                f"(adapter={type(self).__name__})."
            )
