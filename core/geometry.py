"""
core/geometry.py
================

球台几何与标定（Camera Calibration）—— 与模型无关的部分。

为什么单独抽出来
----------------
原先 ``smart_referee_yolo.py`` 用**文件名嗅探**决定标定参数::

    if "rally" in video_lower:      ...
    elif "plos" in video_lower:     ...

这导致换任何新机位都必须改源码。这里把「标定」变成**显式的数据对象**
(:class:`TableCalibration`)，作为适配器/事件处理器的构造参数传入：
换机位 = 换一个数据，不需要改一行算法。

坐标约定
--------
* 图像像素：左上角原点，x 向右、y 向下（§7.1）
* ``ROIBox`` 的字段顺序沿用既有工程的 ``(y_min, y_max, x_min, x_max)``，
  与 ``frame[ymin:ymax, xmin:xmax]`` 的切片写法一致，便于与老代码对照。
* 球台物理尺寸固定为 ITTF 标准 ``274cm x 152.5cm``（见 ``TABLE_LENGTH_CM``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ROIBox",
    "TableCalibration",
    "CALIBRATION_PRESETS",
    "guess_preset_name",
    "preset",
    "TABLE_LENGTH_CM",
    "TABLE_WIDTH_CM",
    "TABLE_HEIGHT_CM",
]

# ITTF 标准球台（cm）
TABLE_LENGTH_CM = 274.0
TABLE_WIDTH_CM = 152.5
TABLE_HEIGHT_CM = 76.0


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ROIBox:
    """感兴趣区域（表格裁剪框），半开区间 ``[y_min, y_max) x [x_min, x_max)``。

    只描述"想裁哪一块"，**不**假设它一定在画面内；越界由 :meth:`clip` 处理。
    """

    y_min: int
    y_max: int
    x_min: int
    x_max: int

    def __post_init__(self) -> None:
        for name in ("y_min", "y_max", "x_min", "x_max"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name} must be an int, got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        if self.y_max <= self.y_min:
            raise ValueError(f"y_max ({self.y_max}) must be > y_min ({self.y_min})")
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be > x_min ({self.x_min})")

    # -- 基本属性 ---------------------------------------------------------
    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def offset(self) -> Tuple[int, int]:
        """裁剪框左上角在整图中的偏移 ``(offset_x, offset_y)``。"""
        return (self.x_min, self.y_min)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """``(y_min, y_max, x_min, x_max)`` —— 与老工程 ``roi_box`` 完全一致。"""
        return (self.y_min, self.y_max, self.x_min, self.x_max)

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_sequence(cls, values: Sequence[int]) -> "ROIBox":
        """从 ``(y_min, y_max, x_min, x_max)`` 构造（兼容老代码的 list/tuple）。"""
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("roi_box must be a 4-element sequence (y_min, y_max, x_min, x_max)")
        if len(values) != 4:
            raise ValueError(f"roi_box must have exactly 4 elements, got {len(values)}")
        return cls(*(int(v) for v in values))

    # -- 运算 -------------------------------------------------------------
    def clip(self, frame_width: int, frame_height: int) -> "ROIBox":
        """把裁剪框裁到画面内。

        若裁剪框完全落在画面外（裁完为空），抛 ``ValueError``——静默返回空图
        只会把错误推到更难排查的地方。
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"invalid frame size {frame_width}x{frame_height}")
        x_min = min(max(self.x_min, 0), frame_width)
        x_max = min(max(self.x_max, 0), frame_width)
        y_min = min(max(self.y_min, 0), frame_height)
        y_max = min(max(self.y_max, 0), frame_height)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"ROI {self.to_tuple()} does not intersect the {frame_width}x{frame_height} frame"
            )
        return ROIBox(y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max)

    def crop(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """按本 ROI 裁剪一帧，返回 ``(crop, (offset_x, offset_y))``。

        ROI 会先被裁剪到画面内，返回的 offset 是**实际**生效的偏移量。
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array (BGR image)")
        h, w = frame.shape[:2]
        box = self.clip(w, h)
        crop = frame[box.y_min:box.y_max, box.x_min:box.x_max]
        return crop, box.offset

    def scaled(self, sx: float, sy: float) -> "ROIBox":
        """按比例缩放（用于把基准分辨率下的标定迁移到实际分辨率）。

        使用 ``int()`` **截断**而非四舍五入，与 ``smart_referee_yolo.py`` 的
        历史实现保持一致，保证同一标定在不同分辨率下与既有评测结果逐像素可复现。
        """
        return ROIBox(
            y_min=int(self.y_min * sy),
            y_max=int(self.y_max * sy),
            x_min=int(self.x_min * sx),
            x_max=int(self.x_max * sx),
        )


# ---------------------------------------------------------------------------
# 标定
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableCalibration:
    """一套机位标定：ROI、球台四边形、球网位置。

    这些量都是**静态**的（相机不动、球台不搬），因此完全可以在安装时标定一次、
    之后作为常量复用，不参与每帧推理。

    Attributes
    ----------
    name:
        标定名字，例如 ``"game2"``，仅用于日志与溯源。
    base_width, base_height:
        该标定参数所对应的**基准分辨率**。实际视频分辨率不同时会自动按比例缩放。
    table_polygon:
        球台四角在画面中的像素坐标，顺序 ``左上, 右上, 右下, 左下``。
        用于判断落点是否在台面内，以及单应性变换到俯视图。
    roi_box:
        送入检测模型的裁剪框。
    net_x:
        球网在画面中的 x 坐标（竖直方向的单条线），用于判定左右半台。
    net_zone:
        球网在画面中的 y 范围 ``(y0, y1)``，用于判定"擦网"事件的有效高度带。
    """

    name: str
    base_width: int
    base_height: int
    table_polygon: Tuple[Tuple[int, int], ...]
    roi_box: ROIBox
    net_x: int
    net_zone: Tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.base_width <= 0 or self.base_height <= 0:
            raise ValueError("base_width / base_height must be positive")
        if len(self.table_polygon) != 4:
            raise ValueError(f"table_polygon must have exactly 4 corners, got {len(self.table_polygon)}")
        if self.net_zone[0] >= self.net_zone[1]:
            raise ValueError(f"net_zone must satisfy y0 < y1, got {self.net_zone}")

    # -- 缩放 -------------------------------------------------------------
    def scaled_to(self, frame_width: int, frame_height: int) -> "TableCalibration":
        """把标定参数按比例映射到实际视频分辨率。

        缩放假定为 ``int()`` 截断，与 ``smart_referee_yolo.py`` 的 ``main()``
        保持一致——这是复现历史指标（comparison_results.csv）的前提。
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"invalid frame size {frame_width}x{frame_height}")
        sx = frame_width / float(self.base_width)
        sy = frame_height / float(self.base_height)
        if math.isclose(sx, 1.0) and math.isclose(sy, 1.0):
            return self
        poly = tuple((int(x * sx), int(y * sy)) for x, y in self.table_polygon)
        return replace(
            self,
            base_width=frame_width,
            base_height=frame_height,
            table_polygon=poly,
            roi_box=self.roi_box.scaled(sx, sy),
            net_x=int(self.net_x * sx),
            net_zone=(int(self.net_zone[0] * sy), int(self.net_zone[1] * sy)),
        )

    # -- 几何判定 ---------------------------------------------------------
    @property
    def table_polygon_np(self) -> np.ndarray:
        """``table_polygon`` 的 ``int32`` numpy 形式，供 ``cv2.pointPolygonTest`` 使用。"""
        return np.asarray(self.table_polygon, dtype=np.int32)

    def side_of(self, x: float) -> str:
        """根据 x 坐标判定球在画面哪一侧（``"A"`` 左 / ``"B"`` 右），与 §7.4 对齐。"""
        return "A" if x < self.net_x else "B"

    def distance_to_table(self, point: Sequence[float]) -> float:
        """点到球台多边形边界的带符号距离（正=台内，负=台外，单位像素）。

        需要 OpenCV；调用方应保证已安装。
        """
        import cv2  # 局部导入，避免本模块在无 cv2 环境下不可用

        return float(cv2.pointPolygonTest(self.table_polygon_np, (float(point[0]), float(point[1])), True))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_width": self.base_width,
            "base_height": self.base_height,
            "table_polygon": [list(p) for p in self.table_polygon],
            "roi_box": list(self.roi_box.to_tuple()),
            "net_x": self.net_x,
            "net_zone": list(self.net_zone),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableCalibration":
        return cls(
            name=data["name"],
            base_width=int(data["base_width"]),
            base_height=int(data["base_height"]),
            table_polygon=tuple((int(p[0]), int(p[1])) for p in data["table_polygon"]),
            roi_box=ROIBox.from_sequence(data["roi_box"]),
            net_x=int(data["net_x"]),
            net_zone=(int(data["net_zone"][0]), int(data["net_zone"][1])),
        )


# ---------------------------------------------------------------------------
# 已标定的机位预设
# ---------------------------------------------------------------------------
# 数值原样迁移自 yolo_core/scripts/smart_referee_yolo.py 的 main()，
# 保持与既有评测结果（comparison_results.csv）完全可复现。
CALIBRATION_PRESETS: Dict[str, TableCalibration] = {
    # OpenTTGames Game 2 / Test 2：1080p 标准正视角，蓝台白球
    "game2": TableCalibration(
        name="game2",
        base_width=1920,
        base_height=1080,
        table_polygon=((557, 519), (1329, 535), (1473, 654), (367, 631)),
        roi_box=ROIBox(y_min=200, y_max=850, x_min=150, x_max=1780),
        net_x=929,
        net_zone=(450, 642),
    ),
    # test_rally.mp4：1080p 侧视转播机位，橙球，含吊灯干扰
    "rally": TableCalibration(
        name="rally",
        base_width=1920,
        base_height=1080,
        table_polygon=((587, 575), (1305, 450), (1630, 565), (765, 786)),
        roi_box=ROIBox(y_min=350, y_max=860, x_min=80, x_max=1850),
        net_x=1148,
        net_zone=(450, 680),
    ),
    # test_plos.mp4：576p 低清抛球实验
    "plos": TableCalibration(
        name="plos",
        base_width=1024,
        base_height=576,
        table_polygon=((750, 440), (1010, 440), (1010, 560), (750, 560)),
        roi_box=ROIBox(y_min=120, y_max=560, x_min=200, x_max=1024),
        net_x=880,
        net_zone=(350, 560),
    ),
    # OpenTTGames Test 4：1080p @ 120fps，深绿台面机位
    "test4": TableCalibration(
        name="test4",
        base_width=1920,
        base_height=1080,
        table_polygon=((522, 625), (1259, 625), (1439, 735), (348, 750)),
        roi_box=ROIBox(y_min=320, y_max=920, x_min=140, x_max=1650),
        net_x=892,
        net_zone=(500, 750),
    ),
}

# test_2.mp4：与 game2 **同一机位**，但帧率不同（120fps vs 25fps）。
#
# 复用 game2 标定的依据（实测，非假定）：
#   * 把 game2 的球台多边形套在 test_2 上，台面色像素覆盖率 99.6~100%（两视频均如此）；
#   * 同一视频时刻整帧平均绝对差仅 7/255，下半幅（球台/地面）6.7/255。
# 帧率差异不影响标定（标定只与机位有关）；时间戳由视频实际帧率换算。
CALIBRATION_PRESETS["test2"] = replace(CALIBRATION_PRESETS["game2"], name="test2")


def guess_preset_name(video_path: str) -> str:
    """按文件名猜标定预设名，默认回退到 ``"game2"``。

    注意
    ----
    这是给**命令行/基准脚本**用的便利函数，用来复现历史结果。
    适配器与事件处理器**不应该**调用它——它们必须接收显式的
    :class:`TableCalibration`，否则换机位就要改代码（老代码的老毛病）。
    """
    lowered = str(video_path).lower()
    if "rally" in lowered:
        return "rally"
    if "plos" in lowered:
        return "plos"
    # 顺序要紧：test_2 / test_4 必须先于默认分支；且 "test_game_2" 不含子串 "test_2"
    if "test_4" in lowered or "test4" in lowered:
        return "test4"
    if "test_2" in lowered or "test2" in lowered:
        return "test2"
    return "game2"


def preset(name: str) -> TableCalibration:
    """按名字取标定预设，名字非法时给出可读的报错。"""
    try:
        return CALIBRATION_PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown calibration preset {name!r}; available: {sorted(CALIBRATION_PRESETS)}"
        ) from None
