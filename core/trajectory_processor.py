"""
core/trajectory_processor.py
============================

轨迹与事件处理层：把**逐帧观测**转换成**物理事件**。

职责边界
--------
本模块是组长方案 §1 架构里"轨迹/事件处理"那一层：

.. code-block:: text

    YOLO / TrackNet / TTNet 适配器
        → FrameObservation（逐帧观测）
        → 【本模块】轨迹与事件处理
        → RefereeEvent（物理事件）
        → 计分状态机

**本模块不判分。** 它只回答"发生了什么物理事件"（落台/击球/擦网/出界），
"该给谁加分"完全由状态机决定——这与组长方案 §1/§7 的分工一致。

与模型无关
----------
本模块只吃 :class:`~core.schemas.FrameObservation`，不接触像素、不知道模型是谁。
TrackNet / TTNet 的观测可以直接灌进来复用同一套事件逻辑
（组长方案 §3："需要分开的是事件提取逻辑，而不是计分规则"）。

对组长硬性要求的落实
--------------------
* **§4-1 用时间戳，不用固定帧数**：所有超时、冷却、去重窗口一律以秒为单位，
  按 ``timestamp_s`` 判定。老代码的 ``lost_frames > 6``、``lost_frames >= 3``
  已替换为秒比较，因此**换帧率行为不变**（有单测保证）。
* **§4-2 不把"球消失"当失分**：本层不判分；且 ``OUT`` 事件**只在有正向证据时
  产生**（球在下降且已越过台面边界），绝不因"球消失超过 N 秒"而触发。
* **§5 单目不投影空中球**：本层**不做**任何单应性投影，只输出像素坐标
  ``position_px``。落点解算属后续 ``core/homography.py``，且只在 BOUNCE 瞬态调用。
* **§7.5 事件去重**：同类型事件在 ``dedup_window_s`` 内聚合为单条。

两个关键不变量
--------------
1. **只有真实观测（MEASURED / SMOOTHED）才能产生事件。**
   卡尔曼外推点（PREDICTED）只用于维持轨迹连续性，绝不参与事件判定。
   这修掉了老代码 ``update_physics_engine(..., is_pred)`` 收了参数却从未使用的问题
   （``smart_referee_yolo.py:412``）——即"事件可能由未观测到的位置产生"。
2. **新目标必须经双帧位移确认才建立跟踪**，防止静止反光（球鞋、地标）与
   单帧孤立噪点起轨。

刻意不做的两件事
----------------
* **不访问像素**：老代码的"预测点邻域局部帧差补球"需要读图，与"模型无关"冲突。
  若后续要补回，应作为**适配器侧的带外能力**（配合预测提示），而不是让本层读图。
* **不做补间插值**：补间只为渲染平滑服务，而渲染是可视化层的职责；
  且补间点本就不可用于事件判定。轨迹断连交由可视化层按帧间隔/距离自行截断。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.geometry import TableCalibration
from core.homography import TableHomography
from core.schemas import (
    BallObservation,
    EventType,
    FrameObservation,
    RefereeEvent,
    make_event_id,
)

__all__ = [
    "MotionState",
    "TrackPoint",
    "ProcessorParams",
    "TrajectoryEventProcessor",
    "REFERENCE_WIDTH",
    "REFERENCE_HEIGHT",
]

# 空间阈值（px）的参考分辨率。老代码的所有像素阈值都在这个分辨率下标定。
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080


# ---------------------------------------------------------------------------
# 轨迹点
# ---------------------------------------------------------------------------
class MotionState(str, Enum):
    """轨迹点的来源。区分"观测"与"推算"是 §7.3 的核心要求。"""

    MEASURED = "MEASURED"          # 模型真实检出
    PREDICTED = "PREDICTED"        # 卡尔曼外推（无观测支撑）
    SMOOTHED = "SMOOTHED"          # 被防尖刺校正过的观测点
    INTERPOLATED = "INTERPOLATED"  # 补间点（本层不产生，预留给可视化层）


@dataclass
class TrackPoint:
    """轨迹上的一个点。"""

    frame_id: int
    timestamp_s: float
    x: float
    y: float
    state: MotionState
    confidence: Optional[float] = None

    @property
    def is_observed(self) -> bool:
        """是否由真实观测支撑。**只有这类点能产生事件。**"""
        return self.state in (MotionState.MEASURED, MotionState.SMOOTHED)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessorParams:
    """门限参数。

    时间类参数单位为**秒**（组长 §4-1）；空间类单位为**参考分辨率下的像素**
    （``REFERENCE_WIDTH x REFERENCE_HEIGHT``），运行时按实际分辨率等比缩放，
    以适配多源视频（自录 + 公开数据集，分辨率各不相同，见 A8 结论）。

    默认值迁移自 ``smart_referee_yolo.py`` 的调参结果以保证既有指标可复现；
    老代码中以帧数表达的阈值已按 25fps 换算成秒并注明出处。
    """

    # ---- 时间（秒）------------------------------------------------------
    max_extrapolation_s: float = 0.08   # 老: max(2, int(fps*0.08))
    track_reset_s: float = 0.15         # 老: max(4, int(fps*0.15))
    reacquire_s: float = 0.12           # 老: lost_frames >= 3 @25fps
    cooldown_hit_s: float = 0.18        # 老: max(5, int(fps*0.18))
    cooldown_bounce_s: float = 0.22     # 老: max(7, int(fps*0.22))
    cooldown_net_s: float = 0.25        # 老: max(8, int(fps*0.25))
    dedup_window_s: float = 0.08        # 见 决策与风险记录.md R-005
    confirm_window_s: float = 0.12      # 双帧确认允许的最大间隔

    # ---- 双门限置信度 ---------------------------------------------------
    gate_near_px: float = 45.0
    gate_mid_px: float = 90.0
    conf_near: float = 0.20
    conf_mid: float = 0.38
    conf_new: float = 0.35

    # ---- 双帧位移起步确认 -----------------------------------------------
    # 注意：这两个门限是**速度**（px/s），不是"每帧位移"。
    #
    # 为什么必须是速度（实测教训）：最初沿用老代码的"每帧位移"写法
    # （3.5px ~ 65px/帧 @25fps），结果起轨能力**强依赖帧率**：
    #   * 帧率太低 → 快球每帧位移超过上限 → 起不了轨；
    #   * 帧率太高 → 慢球每帧位移低于下限 → 起不了轨。
    # 实测扫描（球速 × 帧率）显示：没有任何单一帧率能覆盖全部球速范围。
    # 改成速度后，起轨判据与帧率无关，只与球的物理运动有关。
    #
    # 默认值依据**实测球速分布**（三份视频统计，px/s）：
    #   test_game_2   p5=339  中位=408  p95=830   最大=1602
    #   test_2        p5=425  中位=2133 p95=4734  最大=5106
    #   test_rally    p5=33   中位=137  p95=1469  最大=3318
    # 即真实范围约 33 ~ 5100 px/s。下限取 50（足以挡掉静止反光，又容得下慢球）；
    # 上限取 8000（约为实测最大值的 1.6 倍，对应约 30 m/s 的极限扣杀）。
    # 注意：老代码的 65px/帧 @25fps = 1625 px/s **只够覆盖 test_game_2 那种慢球**，
    # 用它当上限会把 test_2 里的快球全部拒掉。
    confirm_min_speed_px_s: float = 50.0
    confirm_max_speed_px_s: float = 8000.0

    # ---- 防尖刺与分段断连 -----------------------------------------------
    spike_dev_px: float = 28.0
    spike_span_px: float = 90.0
    segment_gap_px: float = 85.0

    # ---- 事件判据 -------------------------------------------------------
    net_half_width_px: float = 30.0
    bounce_vy_down: float = 1.2         # px/s（老代码原值，实为"正在向下"判据）
    bounce_vy_up: float = -0.8          # px/s
    hit_vx_min: float = 2.0             # px/s
    net_slowdown_ratio: float = 0.5
    # 出界判据：落点距台面边界的**厘米**容差（不是像素！）。
    # 用厘米而非像素，是因为像素阈值在透视下各向异性严重——球台长边 274cm
    # 映射约 1100px，短边 152.5cm 只映射约 135px，同一个像素阈值在两个方向上
    # 代表的物理距离差了近一个数量级。
    out_margin_cm: float = 6.0
    # 出界只在"良态区域"判定：要求落点在图像上位于球台**近边之下**。
    #
    # 理由（实测）：球台远端上方那片区域透视压缩极严重——距台面顶边约 40px
    # 的位置，单应性会算出"台外 85cm"。透视把像素噪声放大成几十厘米的虚假出界。
    # 老代码用 `cy > 670` 做同样的限制。牺牲远端长球的召回，换出界事件的可靠性，
    # 与组长 §4-2"不拿弱证据判分"的精神一致：远端长球交给状态机的超时判据处理。
    out_requires_near_region: bool = True

    # ---- 静止死锁 -------------------------------------------------------
    # 同样是**时间 + 距离**判据，不是"每帧位移"：
    # 若球在 static_max_s 内相对参考点移动不足 static_min_travel_px，判为静止。
    # 用"每帧位移 < 1px"的老写法在高帧率下会把慢球误判成静止
    # （150px/s @240fps 每帧只动 0.63px）。
    static_min_travel_px: float = 3.0
    static_max_s: float = 0.12

    # ---- 轨迹缓冲 -------------------------------------------------------
    max_trajectory_points: int = 90

    def scaled(self, width: int, height: int) -> "ProcessorParams":
        """按实际分辨率等比缩放所有**空间类**阈值（时间阈值、比例、厘米不变）。

        空间量纲（px 与 px/s）都按同一系数缩放：分辨率缩放改变像素距离，
        时间不变，因此速度（px/s）与位移（px）的缩放系数相同。
        ``bounce_vy_* / hit_vx_min`` 数值上接近 0（ε 门限），缩放与否行为上
        几乎无差别，但为了"分辨率无关"的完整性仍一并缩放。
        """
        scale = _resolution_scale(width, height)
        if math.isclose(scale, 1.0):
            return self
        spatial_fields = (
            # 像素距离类
            "gate_near_px",
            "gate_mid_px",
            "spike_dev_px",
            "spike_span_px",
            "segment_gap_px",
            "net_half_width_px",
            # 像素速度类（px/s）
            "confirm_min_speed_px_s",
            "confirm_max_speed_px_s",
            "bounce_vy_down",
            "bounce_vy_up",
            "hit_vx_min",
        )
        return replace(self, **{f: getattr(self, f) * scale for f in spatial_fields})


def _resolution_scale(width: int, height: int) -> float:
    """相对参考分辨率的等比缩放系数（面积等效）。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid resolution {width}x{height}")
    return math.sqrt((width / REFERENCE_WIDTH) * (height / REFERENCE_HEIGHT))


# ---------------------------------------------------------------------------
# 时钟驱动卡尔曼
# ---------------------------------------------------------------------------
class _ClockDrivenKalman:
    """按 ``dt = t_i - t_{i-1}`` 动态刷新状态转移矩阵的卡尔曼滤波。

    与老代码的区别：老代码在 ``set_fps()`` 里一次性定死 ``dt``；
    这里每次预测都按**真实时间差**重建 F，因此天然兼容变帧率视频。
    """

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[2, 2] = 200.0
        self.kf.processNoiseCov[3, 3] = 200.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def _set_dt(self, dt: float) -> None:
        dt = max(float(dt), 1e-6)
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

    def initialize(self, x: float, y: float, vx: float, vy: float) -> None:
        self.kf.statePost = np.array([[x], [y], [vx], [vy]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 5.0
        self.initialized = True

    def predict(self, dt: float) -> Tuple[float, float]:
        self._set_dt(dt)
        p = self.kf.predict()
        return (float(p[0, 0]), float(p[1, 0]))

    def correct(self, x: float, y: float) -> Tuple[float, float]:
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]], np.float32))
        return (float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]))

    @property
    def velocity(self) -> Tuple[float, float]:
        if not self.initialized:
            return (0.0, 0.0)
        return (float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0]))


@dataclass
class _Tentative:
    """等待双帧确认的新目标。"""

    frame_id: int
    timestamp_s: float
    x: float
    y: float
    confidence: Optional[float]


# ---------------------------------------------------------------------------
# 主处理器
# ---------------------------------------------------------------------------
class TrajectoryEventProcessor:
    """把逐帧观测转换成物理事件（**不判分**）。

    Parameters
    ----------
    calibration:
        球台标定。会按每条观测的实际分辨率自动缩放，因此同一实例可处理不同
        分辨率的视频流（多源数据，见 ``决策与风险记录.md`` A8）。
    params:
        门限参数，见 :class:`ProcessorParams`。
    source:
        填充 ``RefereeEvent.sources`` 的来源名（``"yolo"`` / ``"tracknet"`` / ``"ttnet"``）。

    Examples
    --------
    >>> from core.geometry import preset
    >>> from core.trajectory_processor import TrajectoryEventProcessor
    >>> proc = TrajectoryEventProcessor(preset("game2"))
    >>> events = proc.update(observation)   # doctest: +SKIP
    """

    def __init__(
        self,
        calibration: TableCalibration,
        params: Optional[ProcessorParams] = None,
        source: str = "yolo",
    ) -> None:
        if not isinstance(calibration, TableCalibration):
            raise TypeError(f"calibration must be TableCalibration, got {type(calibration).__name__}")
        self._calibration = calibration
        self._params = params or ProcessorParams()
        self._source = source
        self._homography_cache: Dict[Tuple[int, int], TableHomography] = {}
        self.reset()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空全部跨帧状态（换视频、跳转、开始新回合）。"""
        self._kalman = _ClockDrivenKalman()
        self._trajectory: Deque[TrackPoint] = deque(maxlen=self._params.max_trajectory_points)
        self._vel_history: Deque[Tuple[float, float, float]] = deque(maxlen=10)

        self._last_timestamp: Optional[float] = None
        self._last_prediction: Optional[Tuple[float, float]] = None
        self._last_measured: Optional[TrackPoint] = None
        self._last_raw: Optional[Tuple[float, float]] = None
        self._lost_since: Optional[float] = None
        self._static_since: Optional[float] = None
        self._static_ref: Optional[Tuple[float, float, float]] = None   # (t, x, y)
        self._consecutive_measured = 0
        self._tentative: Optional[_Tentative] = None

        self._last_event_time: Dict[str, float] = {}
        self._event_counters: Dict[str, int] = {}
        self._recent_events: Deque[RefereeEvent] = deque(maxlen=64)

        self.stats: Dict[str, Any] = {
            "frames": 0,
            "measured": 0,
            "predicted": 0,
            "smoothed": 0,
            "rejected_static": 0,
            "tentative_started": 0,
            "confirmed_tracks": 0,
            "events": {t.value: 0 for t in EventType},
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def update(
        self,
        observation: FrameObservation,
        candidates: Optional[Sequence[BallObservation]] = None,
    ) -> List[RefereeEvent]:
        """喂入一帧观测，返回**本帧产生的事件**（通常 0 或 1 条）。

        Parameters
        ----------
        observation:
            适配器输出的统一逐帧观测（§7.1）。
        candidates:
            该帧全部候选（来自适配器带外通道 ``candidates_for``）。不传时退化为
            只用 ``observation.ball``。多候选用于"预测点邻域内放宽置信度"的双门限。
        """
        if not isinstance(observation, FrameObservation):
            raise TypeError(f"observation must be FrameObservation, got {type(observation).__name__}")

        t = observation.timestamp_s
        if self._last_timestamp is not None and t < self._last_timestamp:
            raise ValueError(
                f"timestamp_s went backwards: {t} < {self._last_timestamp}. "
                "Call reset() when seeking or looping."
            )
        dt = t - self._last_timestamp if self._last_timestamp is not None else 0.0
        if dt < 0:
            dt = 0.0

        params = self._params.scaled(*observation.image_size)
        calib = self._calibration.scaled_to(*observation.image_size)
        homography = self._homography_for(observation.image_size)
        self.stats["frames"] += 1

        # 1. 双门限置信度选点
        chosen = self._select_candidate(observation, candidates, params)

        # 2. 状态推进（返回本帧的轨迹点，可能是观测点/外推点/None）
        point = self._advance(chosen, observation, dt, t, params)
        if point is not None:
            self._trajectory.append(point)

        # 3. 防尖刺（只校正已记录的观测点）
        self._apply_anti_spike(params)

        # 4. 事件提取（严格只认观测点）
        events = self._extract_events(point, calib, homography, params, t)

        self._last_timestamp = t
        return events

    def _homography_for(self, image_size: Tuple[int, int]) -> TableHomography:
        """按分辨率缓存单应性矩阵（4 点解算很便宜，但没必要每帧重算）。"""
        cached = self._homography_cache.get(image_size)
        if cached is None:
            cached = TableHomography(self._calibration, image_size=image_size)
            self._homography_cache[image_size] = cached
        return cached

    # ------------------------------------------------------------------
    # 1. 候选选择（双门限置信度）
    # ------------------------------------------------------------------
    def _select_candidate(
        self,
        observation: FrameObservation,
        candidates: Optional[Sequence[BallObservation]],
        params: ProcessorParams,
    ) -> Optional[BallObservation]:
        pool: List[BallObservation] = []
        if candidates:
            pool = [c for c in candidates if c.visible]
        elif observation.ball.visible:
            pool = [observation.ball]
        if not pool:
            return None

        pred = self._last_prediction
        if not self._kalman.initialized or pred is None:
            # 未跟踪：必须达到较高置信度，防止低分噪点起轨
            best = max(pool, key=lambda c: c.confidence or 0.0)
            return best if (best.confidence or 0.0) >= params.conf_new else None

        best: Optional[BallObservation] = None
        best_score = -1e9
        for cand in pool:
            conf = cand.confidence if cand.confidence is not None else 1.0
            dist = math.hypot(cand.center_px[0] - pred[0], cand.center_px[1] - pred[1])
            if dist <= params.gate_near_px:
                if conf < params.conf_near:
                    continue          # 近邻放宽到 0.20，召回高速拖影
            elif dist <= params.gate_mid_px:
                if conf < params.conf_mid:
                    continue
            else:
                continue              # 超出搜索范围，交给重捕逻辑
            score = conf - (dist / 180.0) * 0.1
            if score > best_score:
                best_score = score
                best = cand
        return best

    # ------------------------------------------------------------------
    # 2. 状态推进
    # ------------------------------------------------------------------
    def _advance(
        self,
        chosen: Optional[BallObservation],
        observation: FrameObservation,
        dt: float,
        t: float,
        params: ProcessorParams,
    ) -> Optional[TrackPoint]:
        # 每帧先做一次预测：既用于下一帧的候选门限，也用于本帧外推
        self._last_prediction = (
            self._kalman.predict(dt) if (self._kalman.initialized and dt > 0) else None
        )

        if chosen is None:
            return self._on_missing(t, observation.frame_id)

        pos = chosen.center_px

        # ---- 未跟踪：走双帧位移确认 ----
        if not self._kalman.initialized:
            return self._try_acquire(pos, chosen.confidence, observation.frame_id, t, params)

        # ---- 已跟踪：静止死锁检测 ----
        # 注意：静止计时器由 _is_static 内部维护（位移恢复时它会自行清零），
        # 这里**不能**无条件重置，否则累加器永远到不了阈值。
        if self._is_static(pos, t, params):
            self.stats["rejected_static"] += 1
            self._break_track()
            return None

        smoothed = self._kalman.correct(pos[0], pos[1])
        self._consecutive_measured += 1
        self._lost_since = None
        self.stats["measured"] += 1
        point = TrackPoint(
            observation.frame_id, t, smoothed[0], smoothed[1], MotionState.MEASURED, chosen.confidence
        )
        self._last_measured = point
        return point

    def _try_acquire(
        self,
        pos: Tuple[float, float],
        confidence: Optional[float],
        frame_id: int,
        t: float,
        params: ProcessorParams,
    ) -> Optional[TrackPoint]:
        """双帧位移确认：连续两帧都检出、且**速度**在物理合理区间，才建立跟踪。

        这条门限同时挡掉两类干扰：静止反光（速度过低）与单帧孤立噪点。

        判据用**速度（px/s）而非每帧位移**，因此与帧率无关——见
        :attr:`ProcessorParams.confirm_min_speed_px_s` 的说明。
        """
        if self._tentative is None:
            self._tentative = _Tentative(frame_id, t, pos[0], pos[1], confidence)
            self.stats["tentative_started"] += 1
            return None

        tn = self._tentative
        gap = t - tn.timestamp_s
        if gap > params.confirm_window_s:
            # 间隔太久，重新起算
            self._tentative = _Tentative(frame_id, t, pos[0], pos[1], confidence)
            self.stats["tentative_started"] += 1
            return None

        dist = math.hypot(pos[0] - tn.x, pos[1] - tn.y)
        speed = dist / max(gap, 1e-6)
        if not (params.confirm_min_speed_px_s <= speed <= params.confirm_max_speed_px_s):
            # 速度不合理：可能是静止物，也可能是另一个物体，重新起算
            self._tentative = _Tentative(frame_id, t, pos[0], pos[1], confidence)
            self.stats["tentative_started"] += 1
            return None

        # 确认：用两点初始化速度
        vx = (pos[0] - tn.x) / max(gap, 1e-6)
        vy = (pos[1] - tn.y) / max(gap, 1e-6)
        self._kalman.initialize(pos[0], pos[1], vx, vy)
        self._consecutive_measured = 2
        self._lost_since = None
        self._static_since = None
        self._last_raw = pos
        self._tentative = None
        self.stats["confirmed_tracks"] += 1
        self.stats["measured"] += 1

        # 把确认的第一帧补进轨迹（它是真实观测，只是当时还没确认）
        first = TrackPoint(tn.frame_id, tn.timestamp_s, tn.x, tn.y, MotionState.MEASURED, tn.confidence)
        self._trajectory.append(first)
        self._last_measured = first

        point = TrackPoint(frame_id, t, pos[0], pos[1], MotionState.MEASURED, confidence)
        self._last_measured = point
        return point

    def _on_missing(self, t: float, frame_id: int = -1) -> Optional[TrackPoint]:
        """本帧没有可用观测：先短时外推，超时则断轨。

        这里产生的是 PREDICTED 点，**永远不会**用于事件判定。

        ``frame_id`` 传当前帧的真实帧号（不是 -1 哨兵）：外推点对应的就是
        这一帧，可视化层的轨迹分段按帧号间隔判断连续性，写 -1 会把
        "观测→外推→观测" 误判为断连，尾迹画不连续。
        """
        if self._lost_since is None:
            self._lost_since = t
        lost_s = t - self._lost_since

        if (
            self._kalman.initialized
            and self._consecutive_measured >= 2
            and lost_s < self._params.max_extrapolation_s
            and self._last_prediction is not None
        ):
            self.stats["predicted"] += 1
            p = self._last_prediction
            return TrackPoint(frame_id, t, p[0], p[1], MotionState.PREDICTED, None)

        if lost_s > self._params.track_reset_s:
            self._break_track()
        return None

    def _is_static(self, pos: Tuple[float, float], t: float, params: ProcessorParams) -> bool:
        """静止死锁检测：球在 ``static_max_s`` 内相对参考点移动不足即判静止。

        用**时间 + 距离**而非"每帧位移"，因此与帧率无关。
        （老写法"每帧位移 < 1px"在高帧率下会把慢球误判成静止：
        150px/s @240fps 每帧只动 0.63px。）
        """
        if self._static_ref is None:
            self._static_ref = (t, pos[0], pos[1])
            return False

        t0, x0, y0 = self._static_ref
        if math.hypot(pos[0] - x0, pos[1] - y0) >= params.static_min_travel_px:
            # 已经明显移动过 → 重置参考点
            self._static_ref = (t, pos[0], pos[1])
            self._static_since = None
            return False

        if self._static_since is None:
            self._static_since = t0
        return (t - self._static_since) >= params.static_max_s

    def _break_track(self) -> None:
        self._kalman = _ClockDrivenKalman()
        self._consecutive_measured = 0
        self._static_since = None
        self._static_ref = None
        self._last_raw = None

    # ------------------------------------------------------------------
    # 3. 防尖刺
    # ------------------------------------------------------------------
    def _apply_anti_spike(self, params: ProcessorParams) -> None:
        """三点曲率防尖刺：中点偏离首尾中点过远且首尾跨度不大时，拉回中点。"""
        pts = [p for p in self._trajectory if p.is_observed]
        if len(pts) < 3:
            return
        a, b, c = pts[-3], pts[-2], pts[-1]
        mx, my = (a.x + c.x) / 2.0, (a.y + c.y) / 2.0
        d_spike = math.hypot(b.x - mx, b.y - my)
        d_span = math.hypot(c.x - a.x, c.y - a.y)
        if d_spike > params.spike_dev_px and d_span < params.spike_span_px:
            b.x, b.y = mx, my
            b.state = MotionState.SMOOTHED
            self.stats["smoothed"] += 1

    # ------------------------------------------------------------------
    # 4. 事件提取
    # ------------------------------------------------------------------
    def _extract_events(
        self,
        point: Optional[TrackPoint],
        calib: TableCalibration,
        homography: TableHomography,
        params: ProcessorParams,
        t: float,
    ) -> List[RefereeEvent]:
        """从轨迹拐点提取物理事件。

        **只有真实观测点才能产生事件。** PREDICTED 点直接返回空——这正是
        "事件不得由未观测位置产生"的实现。
        """
        if point is None or not point.is_observed:
            return []

        vx, vy = self._kalman.velocity
        self._vel_history.append((t, vx, vy))
        if len(self._vel_history) < 3:
            return []

        _, vx_prev, vy_prev = self._vel_history[-2]
        _, vx_curr, vy_curr = self._vel_history[-1]

        events: List[RefereeEvent] = []

        if (
            vx_prev * vx_curr < 0
            and (abs(vx_prev) > params.hit_vx_min or abs(vx_curr) > params.hit_vx_min)
        ):
            # 击球方归属：**用球被击出后的飞行方向推断，而不是用检出位置**。
            #
            # 物理依据：球员一定把球打向对方半台。所以球往右飞(vx>0) → 击球方在左(A)；
            # 球往左飞(vx<0) → 击球方在右(B)。
            #
            # 为什么不能用位置：速度反转是在**实际击球之后几帧**才被卡尔曼检出的，
            # 而球很快（120fps 下约 10px/帧），检出时球已越过球网，`side_of(x)` 会
            # 把击球方判成对面那一侧，进而让状态机把正常对打误判成 OWN_COURT。
            hit_side = "A" if vx_curr > 0 else "B"
            ev = self._emit(
                EventType.RACKET_HIT, t, point, calib, params, point.confidence, side=hit_side
            )
            if ev:
                events.append(ev)

        if self._is_bounce(vx_prev, vy_prev, vx_curr, vy_curr, params):
            # 落点拐点 + 落点位置 → 区分"落台"与"落在台外"。
            # 判定在**物理厘米空间**做（不是像素）：像素阈值在透视下各向异性，
            # 同一阈值在长边/短边方向代表的物理距离能差近一个数量级。
            landing = homography.locate_landing(point.position)
            if landing.distance_to_edge_cm >= -params.out_margin_cm:
                # 落台侧归属用**球台物理坐标**（x_cm 与 137cm 球网线比较），
                # 不用像素域的 ``calib.side_of()``：透视下图像上的一条竖直线
                # 并不对应球台平面上恒定的物理 x，近网落点按像素判会归错侧
                # （与 R-017"击球方按飞行方向归属"是同一类修正）。
                # 此处投影是合法的：落点拐点意味着球就在台面平面上（§5）。
                ev = self._emit(
                    EventType.BOUNCE, t, point, calib, params, point.confidence,
                    side=homography.side_of_cm(landing.x_cm),
                )
            elif self._in_out_region(point, calib, params):
                # 球在台外发生了方向反转 = 落在台外
                ev = self._emit(
                    EventType.OUT, t, point, calib, params, (point.confidence or 0.0) * 0.5
                )
            else:
                # 台外拐点但位于病态区域（远端压缩区）→ 不报，交给状态机超时判据
                ev = None
            if ev:
                events.append(ev)

        if self._is_net_touch(point, calib, params, vx_prev, vx_curr):
            ev = self._emit(EventType.NET_TOUCH, t, point, calib, params, point.confidence)
            if ev:
                events.append(ev)

        return events

    def _is_bounce(
        self,
        vx_prev: float,
        vy_prev: float,
        vx_curr: float,
        vy_curr: float,
        params: ProcessorParams,
    ) -> bool:
        """落台反弹判据。

        **关键守卫：水平速度若反向，就不是落台，而是球拍击球。**

        球拍击球同样会造成垂直速度反转（球员把球往上打），因此只靠"垂直拐点"
        无法区分二者。二者的本质差别在水平方向：**落台时水平速度基本保持，
        击球时水平速度反向**。

        （老代码里有这条守卫 ``v_prev[0] * v_curr[0] >= -1.0``，本模块重写时曾遗漏。
        实测后果：``RACKET_HIT`` 与 ``BOUNCE`` 同帧双触发，"击球方"与"落台侧"
        变成同一侧，状态机把正常对打误判为 12 次 OWN_COURT。）
        """
        if vx_prev * vx_curr < 0:
            return False

        if vy_prev > params.bounce_vy_down and vy_curr < params.bounce_vy_up:
            return True
        # 兜底：最近 5 个观测点的 y 呈"先降后升"的局部极大
        obs = [p for p in self._trajectory if p.is_observed][-5:]
        if len(obs) == 5:
            ys = [p.y for p in obs]
            if ys[2] >= ys[0] + 2 and ys[2] >= ys[1] and ys[2] >= ys[3] and ys[2] >= ys[4] + 2:
                return True
        return False

    def _in_out_region(
        self, point: TrackPoint, calib: TableCalibration, params: ProcessorParams
    ) -> bool:
        """出界判定是否处于"良态区域"。

        要求落点在图像上位于球台**近边之下**（球已落到台面以下）。远端上方那片
        区域透视压缩严重，几十像素的检测抖动会被单应性放大成几十厘米的虚假出界，
        因此在那里不做出界判定。参见 ``ProcessorParams.out_requires_near_region``。
        """
        if not params.out_requires_near_region:
            return True
        near_edge_y = max(p[1] for p in calib.table_polygon)
        return point.y > near_edge_y

    def _is_net_touch(
        self,
        point: TrackPoint,
        calib: TableCalibration,
        params: ProcessorParams,
        vx_prev: float,
        vx_curr: float,
    ) -> bool:
        """擦网判定。

        刻意保留在**像素空间**，与落台的厘米空间判定形成有意的对照：
        擦网发生在球飞行途中（空中球），把它投影到台面去算"距球网多少厘米"
        正是组长方案 §5 禁止的操作。因此这里只用图像上的邻近性
        （``net_x`` 附近的窄带 + 球网高度带）作为证据，不假装知道物理位置。
        """
        if abs(point.x - calib.net_x) > params.net_half_width_px:
            return False
        if not (calib.net_zone[0] <= point.y <= calib.net_zone[1]):
            return False
        return abs(vx_curr) < params.net_slowdown_ratio * max(abs(vx_prev), 5.0)

    # ------------------------------------------------------------------
    # 事件生成与去重
    # ------------------------------------------------------------------
    def _emit(
        self,
        event_type: EventType,
        t: float,
        point: TrackPoint,
        calib: TableCalibration,
        params: ProcessorParams,
        confidence: Optional[float],
        side: Optional[str] = None,
    ) -> Optional[RefereeEvent]:
        """生成事件。

        ``side`` 为 ``None`` 时按事件位置推断（落台、擦网、出界都适用）；
        击球事件会**显式传入**由飞行方向推断的击球方，见 ``_extract_events``。
        """
        key = event_type.value
        cooldowns = {
            EventType.RACKET_HIT: params.cooldown_hit_s,
            EventType.BOUNCE: params.cooldown_bounce_s,
            EventType.NET_TOUCH: params.cooldown_net_s,
            EventType.OUT: params.cooldown_bounce_s,
        }
        # 冷却：按秒判定（§4-1），不再是帧计数
        last = self._last_event_time.get(key)
        if last is not None and (t - last) < cooldowns[event_type]:
            return None

        # 去重：同类型事件在 dedup_window_s 内聚合为单条（§7.5）
        for prev in reversed(self._recent_events):
            if prev.type == key and (t - prev.timestamp_s) < params.dedup_window_s:
                return None

        self._event_counters[key] = self._event_counters.get(key, 0) + 1
        event = RefereeEvent(
            event_id=make_event_id(event_type, self._event_counters[key]),
            timestamp_s=t,
            type=event_type,
            side=side if side is not None else calib.side_of(point.x),
            position_px=(point.x, point.y),
            confidence=confidence,
            sources=[self._source],
        )
        self._last_event_time[key] = t
        self._recent_events.append(event)
        self.stats["events"][key] += 1
        return event

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    @property
    def trajectory(self) -> List[TrackPoint]:
        return list(self._trajectory)

    @property
    def observed_trajectory(self) -> List[TrackPoint]:
        return [p for p in self._trajectory if p.is_observed]

    @property
    def is_tracking(self) -> bool:
        return self._kalman.initialized

    @property
    def lost_seconds(self) -> Optional[float]:
        if self._lost_since is None or self._last_timestamp is None:
            return None
        return self._last_timestamp - self._lost_since

    def describe(self) -> Dict[str, Any]:
        return {
            "source": self._source,
            "calibration": self._calibration.name,
            "tracking": self.is_tracking,
            "lost_seconds": self.lost_seconds,
            "stats": {k: v for k, v in self.stats.items() if k != "events"},
            "events": dict(self.stats["events"]),
        }

    def __repr__(self) -> str:
        return (
            f"<TrajectoryEventProcessor calib={self._calibration.name!r} "
            f"tracking={self.is_tracking} events={sum(self.stats['events'].values())}>"
        )
