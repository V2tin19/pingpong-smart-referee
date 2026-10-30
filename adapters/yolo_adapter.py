"""
adapters/yolo_adapter.py
========================

YOLO11n 感知适配器（符合组长方案 §7.1 / §7.3）。

职责边界
--------
**只做"看到什么"**：

1. 球台 ROI 自适应裁剪（内部细节，对外不可见）；
2. YOLO11n 单帧推理；
3. 把检测框中心与框体从**裁剪坐标**还原到**原始图像像素坐标**；
4. 输出 :class:`FrameObservation`。

**不做**（与老版本 ``smart_referee_yolo.py`` 的关键区别）：

* 不做卡尔曼跟踪、不外推、不填历史位置 —— 那是事件处理层的事；
* 不做双阈值门限、不做双帧确认、不做防尖刺 —— 那些需要跨帧状态，
  属于 ``TrajectoryEventProcessor``；
* 不判断落台/击球/谁得分。

这样拆分之后，"提高召回"的跟踪策略改动不会碰适配器，"换模型"也不会
碰跟踪策略——两条变化轴彻底解耦。

坐标还原
--------
Ultralytics 返回的 ``xyxy`` 已经是**输入裁剪图**的坐标（其内部的 letterbox
补边已在后处理里撤销）。因此这里只需加上 ROI 的偏移量：

.. code-block:: text

    x_full = x_crop + roi.x_min
    y_full = y_crop + roi.y_min

下游拿到的永远是 1920x1080 这类**原始分辨率**坐标，不需要知道模型输入尺寸。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# 注意：`core/` 与 `adapters/` 是本仓库根目录下的两个**顶层包**。
# 该目录布局是本分支自定的，参考了《YOLO 工作全貌》草稿的 §4.1，
# 但不属于组长《共用计分状态机方案》的硬性约定，可以按组内需要调整。
# 因此这里使用绝对导入。运行脚本时需保证仓库根目录在 sys.path 上
# （直接运行根目录下的脚本，或设置 PYTHONPATH）。
from core.base_adapter import BaseAdapter
from core.geometry import ROIBox, TableCalibration
from core.schemas import BallObservation, EventScores, FrameObservation, Source

__all__ = ["YOLOAdapter"]


class YOLOAdapter(BaseAdapter):
    """把 YOLO11n 的检测框翻译成统一逐帧观测。

    Parameters
    ----------
    weights_path:
        ``.pt`` 权重路径。文件不存在时立即报错（不要拖到第一帧才炸）。
    calibration:
        :class:`~core.geometry.TableCalibration` 机位标定。提供时会自动
        按实际视频分辨率缩放，并使用其 ``roi_box`` 作为裁剪框。
        这是**推荐**的用法。
    roi_box:
        仅给出裁剪框、没有完整标定时的简化用法，``(y_min, y_max, x_min, x_max)``。
        与 ``calibration`` 二选一；都给时以 ``calibration`` 为准。
        两者都不给则对整帧推理（不裁剪）。
    imgsz:
        推理分辨率（Ultralytics 的 ``imgsz``）。960 是本项目实测的性价比拐点。
    conf_threshold:
        送进模型的置信度下限。注意这里是**模型侧**过滤，不是跟踪门限；
        设得比最终门限低一些，才能把低分候选交给事件层判断。
    iou_threshold:
        NMS 的 IoU 阈值。
    device:
        ``None`` 表示交给 Ultralytics 自动选（有 GPU 用 GPU）；
        也可显式传 ``"0"`` / ``"cpu"``。
    max_candidates:
        单帧最多保留多少个候选，防止极端画面下候选爆炸。
    strict_time:
        见 :class:`~core.base_adapter.BaseAdapter`。

    Examples
    --------
    >>> from core.geometry import preset
    >>> from adapters import YOLOAdapter
    >>> adapter = YOLOAdapter("weights/yolo11n_clean_best.pt",
    ...                       calibration=preset("game2"))
    >>> obs = adapter.predict_frame(frame, frame_id=0, timestamp_s=0.0)
    >>> obs.ball.visible, obs.ball.center_px    # doctest: +SKIP
    (True, (912.5, 431.0))
    """

    def __init__(
        self,
        weights_path: Union[str, os.PathLike],
        *,
        calibration: Optional[TableCalibration] = None,
        roi_box: Optional[Sequence[int]] = None,
        imgsz: int = 960,
        conf_threshold: float = 0.18,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        max_candidates: int = 8,
        strict_time: bool = False,
    ) -> None:
        super().__init__(strict_time=strict_time)

        weights_path = os.fspath(weights_path)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        self.weights_path = os.path.abspath(weights_path)

        self._calibration = calibration
        self._static_roi: Optional[ROIBox] = (
            ROIBox.from_sequence(roi_box) if roi_box is not None else None
        )
        if calibration is not None and roi_box is not None:
            # 明确以标定为准，避免"传了两个互相矛盾的几何"这种隐性 bug
            self._static_roi = calibration.roi_box

        if imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}")
        if not 0.0 < conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1], got {conf_threshold}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if max_candidates <= 0:
            raise ValueError(f"max_candidates must be positive, got {max_candidates}")

        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device
        self.max_candidates = int(max_candidates)

        # 惰性加载：import ultralytics 与加载权重都推迟到第一次推理，
        # 让这个模块在没有 torch 的环境里也能被 import（便于跨分支交换契约）。
        self._model: Any = None
        self._resolved_calibration: Optional[TableCalibration] = None
        self._last_roi_offset: Tuple[int, int] = (0, 0)
        self._last_crop_size: Optional[Tuple[int, int]] = None
        # 带外多候选（不进 §7.1 观测结构）
        self._candidates_frame: Optional[int] = None
        self._candidates: List[BallObservation] = []

    # ------------------------------------------------------------------
    # 基类契约
    # ------------------------------------------------------------------
    @property
    def source(self) -> str:
        return Source.YOLO.value

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "weights": os.path.basename(self.weights_path),
                "imgsz": self.imgsz,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "device": self.device,
                "calibration": (
                    self._resolved_calibration.name
                    if self._resolved_calibration is not None
                    else None
                ),
                "roi": (
                    list(self._resolved_calibration.roi_box.to_tuple())
                    if self._resolved_calibration is not None
                    else (list(self._static_roi.to_tuple()) if self._static_roi is not None else None)
                ),
            }
        )
        return info

    def reset(self) -> None:
        super().reset()
        self._last_roi_offset = (0, 0)
        self._last_crop_size = None
        self._candidates_frame = None
        self._candidates = []

    def close(self) -> None:
        self._model = None

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _ensure_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO  # 局部导入：无 torch 环境也能 import 本模块

            self._model = YOLO(self.weights_path)
        return self._model

    def _resolve_roi(self, frame_width: int, frame_height: int) -> Optional[ROIBox]:
        """按实际分辨率解析出本帧要用的裁剪框（每次重新缩放，天然支持分辨率变化）。"""
        if self._calibration is not None:
            calib = self._calibration.scaled_to(frame_width, frame_height)
            self._resolved_calibration = calib
            return calib.roi_box
        return self._static_roi

    # ------------------------------------------------------------------
    # 核心推理
    # ------------------------------------------------------------------
    def _predict_frame(self, frame: np.ndarray, frame_id: int, timestamp_s: float) -> FrameObservation:
        height, width = frame.shape[:2]
        image_size = (int(width), int(height))

        roi = self._resolve_roi(width, height)
        if roi is None:
            crop, offset = frame, (0, 0)
        else:
            crop, offset = roi.crop(frame)

        self._last_roi_offset = offset
        self._last_crop_size = (int(crop.shape[1]), int(crop.shape[0]))

        model = self._ensure_model()
        kwargs: Dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        t0 = time.perf_counter()
        results = model(crop, **kwargs)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._extract_detections(results, offset)
        if not detections:
            # §7.3：漏检就是漏检，不填历史位置、不用 (0,0) 占位
            return self.missing_observation(frame_id, timestamp_s, image_size)

        detections.sort(key=lambda d: d.confidence or 0.0, reverse=True)
        detections = detections[: self.max_candidates]

        # 多候选走带外通道，不进 §7.1 的冻结字段集合
        self._candidates_frame = frame_id
        self._candidates = detections

        return FrameObservation(
            frame_id=frame_id,
            timestamp_s=float(timestamp_s),
            source=self.source,
            image_size=image_size,
            ball=detections[0],
            event_scores=EventScores(),  # YOLO 不具备事件预测能力（§7.2）
        )

    def candidates_for(self, frame_id: int) -> List[BallObservation]:
        """返回该帧 ROI 内检出的全部候选框（按置信度降序）。

        带外能力：YOLO 一帧可能同时检出真球与干扰物（球鞋反光、吊灯边缘），
        事件层需要拿到全部候选才能做"低分候选交给时域门限判断"的关联决策。
        详见 :meth:`core.base_adapter.BaseAdapter.candidates_for`。
        """
        if self._candidates_frame != frame_id:
            return []
        return list(self._candidates)

    def _extract_detections(
        self, results: Any, offset: Tuple[int, int]
    ) -> List[BallObservation]:
        """把 Ultralytics 的结果转成**原图坐标**的 :class:`BallObservation` 列表。"""
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        confs = boxes.conf.detach().cpu().numpy().astype(float)
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)

        off_x, off_y = float(offset[0]), float(offset[1])
        detections: List[BallObservation] = []
        for conf, box in zip(confs, xyxy):
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            cx = (x1 + x2) / 2.0 + off_x
            cy = (y1 + y2) / 2.0 + off_y
            detections.append(
                BallObservation(
                    visible=True,
                    center_px=(cx, cy),
                    bbox_xyxy=(x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y),
                    confidence=min(max(float(conf), 0.0), 1.0),
                )
            )
        return detections

    # ------------------------------------------------------------------
    # 诊断属性
    # ------------------------------------------------------------------
    @property
    def calibration(self) -> Optional[TableCalibration]:
        """本帧实际生效的（已按分辨率缩放的）标定参数，仅用于调试与可视化。"""
        return self._resolved_calibration

    @property
    def last_roi_offset(self) -> Tuple[int, int]:
        """上一帧实际使用的 ROI 偏移 ``(offset_x, offset_y)``。"""
        return self._last_roi_offset

    @property
    def last_crop_size(self) -> Optional[Tuple[int, int]]:
        """上一帧送入模型的裁剪图尺寸 ``(width, height)``。"""
        return self._last_crop_size
