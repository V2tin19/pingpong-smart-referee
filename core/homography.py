"""
core/homography.py
==================

球台单应性变换：把**落点**从图像像素坐标解算到球台物理坐标（厘米）。

组长方案 §5 的限制（硬性）
--------------------------
    单目摄像头不能把空中球的位置直接通过桌面透视矩阵转换成真实落点。
    透视变换只适用于桌面平面；应先通过时间序列判断落台时刻，再定位落点，
    否则会把空中球误判为出界。

本模块用**接口设计**把这条限制做成机制，而不是靠调用方自觉：

* 唯一对外的主入口 :meth:`TableHomography.locate_landing_from_event` **只接受
  ``BOUNCE`` 类型的事件**，传入其它类型直接抛 ``ValueError``。
  这样"把空中球投影到桌面"在代码层面就写不出来。
* 低层的 :meth:`TableHomography.to_table_cm` 仍然公开，但文档明确标注它
  只对**桌面平面上的点**有意义，仅供可视化层画雷达使用。

坐标约定
--------
球台物理坐标系（厘米，ITTF 标准台 274 x 152.5）：

.. code-block:: text

    x_cm: 沿**长边** 0 → 274，对应标定多边形 polygon[0] → polygon[1]
    y_cm: 沿**短边** 0 → 152.5，对应 polygon[0] → polygon[3]
    原点 = polygon[0]（画面中的左上角）

这与老代码 ``smart_referee_yolo.py`` 的 ``project_to_minimap`` 完全一致
（它把 ``polygon[0]→[1]`` 映射到雷达的宽 240px ↔ 274cm，``[1]→[2]`` 映射到高
134px ↔ 152.5cm），因此 2D 雷达的观感与历史版本保持可比。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from core.geometry import TABLE_LENGTH_CM, TABLE_WIDTH_CM, TableCalibration
from core.schemas import EventType, RefereeEvent

__all__ = ["LandingPoint", "TableHomography"]


@dataclass(frozen=True)
class LandingPoint:
    """一个落点的解算结果（球台物理坐标）。

    Attributes
    ----------
    x_cm, y_cm:
        球台物理坐标（厘米）。原点为 ``polygon[0]``。
    in_table:
        是否落在台面内（含 ``tolerance_cm`` 容差）。
    distance_to_edge_cm:
        到台面边界的**带符号**距离（厘米）：台内为正，台外为负。
        可以直接用来判断"擦边"还是"明显出界"。
    image_px:
        对应的原始图像像素坐标，便于在画面上叠加显示。
    """

    x_cm: float
    y_cm: float
    in_table: bool
    distance_to_edge_cm: float
    image_px: Tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "x_cm": round(self.x_cm, 2),
            "y_cm": round(self.y_cm, 2),
            "in_table": self.in_table,
            "distance_to_edge_cm": round(self.distance_to_edge_cm, 2),
            "image_px": [round(self.image_px[0], 2), round(self.image_px[1], 2)],
        }


def _signed_distance_to_rect(
    x: float, y: float, width: float, height: float
) -> float:
    """点到矩形 ``[0,width] x [0,height]`` 的带符号距离（内正外负）。"""
    if 0.0 <= x <= width and 0.0 <= y <= height:
        return min(x, width - x, y, height - y)
    dx = max(-x, x - width, 0.0)
    dy = max(-y, y - height, 0.0)
    return -math.hypot(dx, dy)


class TableHomography:
    """图像像素 ↔ 球台厘米坐标的双向单应性映射。

    Parameters
    ----------
    calibration:
        球台标定。若给了 ``image_size``，会先按该分辨率缩放。
    image_size:
        实际视频分辨率 ``(width, height)``。``None`` 表示 ``calibration``
        已经是目标分辨率下的标定。
    tolerance_cm:
        ``in_table`` 判定容差（厘米）。默认 1cm，用于吸收标定误差与
        球体半径带来的边缘歧义。

    Raises
    ------
    ValueError
        标定四边形退化（面积为零/自交），或单应性矩阵解算失败。
    """

    def __init__(
        self,
        calibration: TableCalibration,
        image_size: Optional[Tuple[int, int]] = None,
        tolerance_cm: float = 1.0,
    ) -> None:
        if not isinstance(calibration, TableCalibration):
            raise TypeError(f"calibration must be TableCalibration, got {type(calibration).__name__}")
        if image_size is not None:
            calibration = calibration.scaled_to(int(image_size[0]), int(image_size[1]))

        self.calibration = calibration
        self.tolerance_cm = float(tolerance_cm)

        src = np.asarray(calibration.table_polygon, dtype=np.float32)
        if src.shape != (4, 2):
            raise ValueError(f"table_polygon must be 4 points, got {src.shape}")

        # 目标：物理厘米坐标。polygon[0]→[1] 为长边(274)，[1]→[2] 为短边(152.5)
        dst = np.asarray(
            [
                [0.0, 0.0],
                [TABLE_LENGTH_CM, 0.0],
                [TABLE_LENGTH_CM, TABLE_WIDTH_CM],
                [0.0, TABLE_WIDTH_CM],
            ],
            dtype=np.float32,
        )

        if abs(cv2.contourArea(src)) < 1e-6:
            raise ValueError(f"degenerate table_polygon (zero area): {calibration.table_polygon}")

        matrix, _ = cv2.findHomography(src, dst, 0)
        if matrix is None:
            raise ValueError(
                f"failed to solve homography for polygon {calibration.table_polygon}"
            )
        self._H = matrix.astype(np.float64)

        inv = np.linalg.inv(self._H)
        self._H_inv = inv

        # 自检：四个角必须精确落到物理矩形的四角，否则说明标定或解算有问题
        self._verify_corners(src)

        # 球网在球台物理坐标下的位置。
        #
        # 注意：这里**不用**标定里的 net_x 反推。透视投影下，图像中的一条竖直
        # 线（球网）并不对应球台平面上的恒定 x——沿着那条线，物理 x 是变化的。
        # 而 ITTF 标准台面由球网**精确平分长边**，所以物理位置就是 274/2 = 137cm，
        # 这是精确值，比任何像素反推都可靠。
        self.net_x_cm = TABLE_LENGTH_CM / 2.0

    # ------------------------------------------------------------------
    # 映射
    # ------------------------------------------------------------------
    def to_table_cm(self, point_px: Sequence[float]) -> Tuple[float, float]:
        """图像像素 → 球台厘米坐标。

        .. warning::
           透视变换**只对桌面平面上的点成立**。把空中球的位置传进来会得到
           错误的"落点"（组长方案 §5）。要解算落点请用
           :meth:`locate_landing_from_event`。
        """
        x, y = float(point_px[0]), float(point_px[1])
        v = self._H @ np.array([x, y, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-12:
            raise ValueError(f"point {point_px} maps to infinity (on the horizon line)")
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def to_image_px(self, x_cm: float, y_cm: float) -> Tuple[float, float]:
        """球台厘米坐标 → 图像像素（用于在画面上回投雷达点）。"""
        v = self._H_inv @ np.array([float(x_cm), float(y_cm), 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-12:
            raise ValueError(f"cm point ({x_cm}, {y_cm}) maps to infinity")
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    # ------------------------------------------------------------------
    # 落点解算（§5 的唯一合法入口）
    # ------------------------------------------------------------------
    def locate_landing_from_event(self, event: RefereeEvent) -> LandingPoint:
        """从 **BOUNCE 事件**解算落点。

        这是本模块对外的主入口，也是组长方案 §5 的机制化落实：
        **只有 ``BOUNCE`` 类型的事件才允许调用**。传入 ``RACKET_HIT`` /
        ``NET_TOUCH`` / ``OUT`` 都会抛错——因为那些事件的位置不在桌面平面上，
        投影过去就是错的。

        Raises
        ------
        ValueError
            事件类型不是 ``BOUNCE``，或事件没有位置信息。
        """
        if not isinstance(event, RefereeEvent):
            raise TypeError(f"event must be RefereeEvent, got {type(event).__name__}")
        if event.type != EventType.BOUNCE.value:
            raise ValueError(
                f"§5 violation: landing point can only be solved from a BOUNCE event, "
                f"got {event.type!r}. Projecting an airborne ball onto the table plane "
                f"produces a bogus landing point."
            )
        if event.position_px is None:
            raise ValueError(f"event {event.event_id} has no position_px")
        return self.locate_landing(event.position_px)

    def locate_landing(self, point_px: Sequence[float]) -> LandingPoint:
        """把一个**已知在桌面上**的像素点解算成 :class:`LandingPoint`。

        仅供已经通过 §5 校验的调用路径使用（即 :meth:`locate_landing_from_event`），
        或可视化层画雷达。
        """
        x_cm, y_cm = self.to_table_cm(point_px)
        dist = _signed_distance_to_rect(x_cm, y_cm, TABLE_LENGTH_CM, TABLE_WIDTH_CM)
        return LandingPoint(
            x_cm=x_cm,
            y_cm=y_cm,
            in_table=dist >= -self.tolerance_cm,
            distance_to_edge_cm=dist,
            image_px=(float(point_px[0]), float(point_px[1])),
        )

    # ------------------------------------------------------------------
    # 便捷判定
    # ------------------------------------------------------------------
    def is_in_table(self, point_px: Sequence[float]) -> bool:
        """像素点是否落在台面内（含容差）。同样只对桌面平面上的点有意义。"""
        return self.locate_landing(point_px).in_table

    def side_of_cm(self, x_cm: float) -> str:
        """按球台物理坐标判断左右半台（``"A"`` / ``"B"``，与 §7.4 对齐）。

        比在像素域比较 ``net_x`` 更稳：不受机位透视影响。
        """
        return "A" if x_cm < self.net_x_cm else "B"

    def calibration_net_x_cm(self, y_px: Optional[float] = None) -> float:
        """**诊断用**：标定里的 ``net_x`` 在球台平面上对应的物理 x（厘米）。

        理想情况下应接近 :attr:`net_x_cm`（137cm）。偏离过大说明标定的球网位置
        与球台四边形不自洽，值得重新标定。``y_px`` 默认取球台多边形重心的 y。
        """
        if y_px is None:
            ys = [p[1] for p in self.calibration.table_polygon]
            y_px = sum(ys) / len(ys)
        return self.to_table_cm((float(self.calibration.net_x), float(y_px)))[0]

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _verify_corners(self, src: np.ndarray) -> None:
        """自检：标定四角必须精确映射到物理矩形四角。"""
        expected = np.asarray(
            [
                [0.0, 0.0],
                [TABLE_LENGTH_CM, 0.0],
                [TABLE_LENGTH_CM, TABLE_WIDTH_CM],
                [0.0, TABLE_WIDTH_CM],
            ],
            dtype=np.float64,
        )
        mapped = np.asarray([self.to_table_cm(p) for p in src], dtype=np.float64)
        err = float(np.max(np.abs(mapped - expected)))
        if err > 1e-3:
            raise ValueError(
                f"homography self-check failed: corner error {err:.6f} cm exceeds 1e-3 cm. "
                f"calibration={self.calibration.name}"
            )

    def describe(self) -> dict:
        return {
            "calibration": self.calibration.name,
            "table_polygon": [list(p) for p in self.calibration.table_polygon],
            "table_cm": [TABLE_LENGTH_CM, TABLE_WIDTH_CM],
            "net_x_cm": round(self.net_x_cm, 2),
            "calibration_net_x_cm": round(self.calibration_net_x_cm(), 2),
            "tolerance_cm": self.tolerance_cm,
        }

    def __repr__(self) -> str:
        return (
            f"<TableHomography calib={self.calibration.name!r} "
            f"net_x_cm={self.net_x_cm:.1f} table={TABLE_LENGTH_CM}x{TABLE_WIDTH_CM}cm>"
        )
