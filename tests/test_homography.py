"""
tests/test_homography.py
========================

球台单应性变换的单元测试。

重点验证组长方案 §5 的限制被**机制化**落实：

* ``locate_landing_from_event`` **只接受 BOUNCE 事件**，其余类型直接抛错
  —— 让"把空中球投影到桌面"在代码层面写不出来。
* 角点必须精确映射到物理矩形四角（标定自检）。
* 像素 ↔ 厘米双向映射可往返。
* 同一物理点在 1080p / 540p 下解算出相同的厘米坐标。

::

    python tests/test_homography.py -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import TABLE_LENGTH_CM, TABLE_WIDTH_CM, preset
from core.homography import TableHomography, _signed_distance_to_rect
from core.schemas import EventType, RefereeEvent

CALIB = preset("game2")
# 球台多边形重心，位于台面内部
CENTROID_PX = (
    sum(p[0] for p in CALIB.table_polygon) / 4.0,
    sum(p[1] for p in CALIB.table_polygon) / 4.0,
)


def bounce_event(x: float, y: float) -> RefereeEvent:
    return RefereeEvent(
        event_id="bounce_00001",
        timestamp_s=1.0,
        type=EventType.BOUNCE,
        side="A",
        position_px=(x, y),
        confidence=0.9,
        sources=["yolo"],
    )


# ---------------------------------------------------------------------------
# 标定自检与角点
# ---------------------------------------------------------------------------
class TestCornerMapping(unittest.TestCase):
    def setUp(self):
        self.h = TableHomography(CALIB)

    def test_corners_map_exactly_to_physical_rectangle(self):
        expected = [
            (0.0, 0.0),
            (TABLE_LENGTH_CM, 0.0),
            (TABLE_LENGTH_CM, TABLE_WIDTH_CM),
            (0.0, TABLE_WIDTH_CM),
        ]
        for corner_px, expect in zip(CALIB.table_polygon, expected):
            got = self.h.to_table_cm(corner_px)
            self.assertAlmostEqual(got[0], expect[0], places=6, msg=f"corner {corner_px}")
            self.assertAlmostEqual(got[1], expect[1], places=6, msg=f"corner {corner_px}")

    def test_physical_rectangle_is_ittf_standard(self):
        self.assertAlmostEqual(TABLE_LENGTH_CM, 274.0)
        self.assertAlmostEqual(TABLE_WIDTH_CM, 152.5)

    def test_net_is_at_half_length(self):
        """球网精确平分长边，物理位置是 137cm。"""
        self.assertAlmostEqual(TableHomography(CALIB).net_x_cm, 137.0)

    def test_degenerate_polygon_raises(self):
        from dataclasses import replace

        degenerate = replace(
            CALIB, table_polygon=((100, 100), (100, 100), (100, 100), (100, 100))
        )
        with self.assertRaises(ValueError) as ctx:
            TableHomography(degenerate)
        self.assertIn("degenerate", str(ctx.exception))


# ---------------------------------------------------------------------------
# 双向映射
# ---------------------------------------------------------------------------
class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.h = TableHomography(CALIB)

    def test_px_to_cm_to_px(self):
        for pt in [CENTROID_PX, (600.0, 560.0), (1200.0, 600.0), (929.0, 584.0)]:
            x_cm, y_cm = self.h.to_table_cm(pt)
            back = self.h.to_image_px(x_cm, y_cm)
            self.assertAlmostEqual(back[0], pt[0], places=4, msg=str(pt))
            self.assertAlmostEqual(back[1], pt[1], places=4, msg=str(pt))

    def test_cm_to_px_to_cm(self):
        for cm in [(0.0, 0.0), (137.0, 76.25), (274.0, 152.5), (50.0, 30.0)]:
            px = self.h.to_image_px(*cm)
            back = self.h.to_table_cm(px)
            self.assertAlmostEqual(back[0], cm[0], places=4, msg=str(cm))
            self.assertAlmostEqual(back[1], cm[1], places=4, msg=str(cm))

    def test_table_centroid_is_inside(self):
        x_cm, y_cm = self.h.to_table_cm(CENTROID_PX)
        self.assertGreater(x_cm, 0.0)
        self.assertLess(x_cm, TABLE_LENGTH_CM)
        self.assertGreater(y_cm, 0.0)
        self.assertLess(y_cm, TABLE_WIDTH_CM)


# ---------------------------------------------------------------------------
# 台内 / 台外判定
# ---------------------------------------------------------------------------
class TestInTable(unittest.TestCase):
    def setUp(self):
        self.h = TableHomography(CALIB)

    def test_interior_point_is_in_table(self):
        landing = self.h.locate_landing(CENTROID_PX)
        self.assertTrue(landing.in_table)
        self.assertGreater(landing.distance_to_edge_cm, 0.0)

    def test_center_of_table_is_deep_inside(self):
        px = self.h.to_image_px(TABLE_LENGTH_CM / 2, TABLE_WIDTH_CM / 2)
        landing = self.h.locate_landing(px)
        self.assertTrue(landing.in_table)
        self.assertAlmostEqual(landing.distance_to_edge_cm, TABLE_WIDTH_CM / 2, places=3)

    def test_far_outside_point_is_out(self):
        # (200, 584) 在台面左侧之外（该高度台面横向范围约 [447, 1388]）
        landing = self.h.locate_landing((200.0, 584.0))
        self.assertFalse(landing.in_table)
        self.assertLess(landing.distance_to_edge_cm, 0.0)

    def test_signed_distance_inside_matches_nearest_edge(self):
        # 距短边 10cm、距长边 20cm 的点
        px = self.h.to_image_px(10.0, 20.0)
        landing = self.h.locate_landing(px)
        self.assertAlmostEqual(landing.distance_to_edge_cm, 10.0, places=3)

    def test_signed_distance_outside_is_negative(self):
        px = self.h.to_image_px(-30.0, 50.0)
        landing = self.h.locate_landing(px)
        self.assertFalse(landing.in_table)
        self.assertAlmostEqual(landing.distance_to_edge_cm, -30.0, places=3)

    def test_rect_distance_helper(self):
        self.assertAlmostEqual(_signed_distance_to_rect(5.0, 5.0, 274.0, 152.5), 5.0)
        self.assertAlmostEqual(_signed_distance_to_rect(-3.0, 50.0, 274.0, 152.5), -3.0)
        self.assertAlmostEqual(_signed_distance_to_rect(-3.0, -4.0, 274.0, 152.5), -5.0)


# ---------------------------------------------------------------------------
# §5 机制化：只允许从 BOUNCE 事件解算落点
# ---------------------------------------------------------------------------
class TestSection5Guard(unittest.TestCase):
    def setUp(self):
        self.h = TableHomography(CALIB)

    def test_bounce_event_is_accepted(self):
        landing = self.h.locate_landing_from_event(bounce_event(*CENTROID_PX))
        self.assertIsNotNone(landing)
        self.assertTrue(landing.in_table)

    def test_non_bounce_events_are_rejected(self):
        """§5：空中球的位置绝不能投影到桌面。非 BOUNCE 事件一律拒绝。"""
        for etype in (EventType.RACKET_HIT, EventType.NET_TOUCH, EventType.OUT):
            with self.subTest(type=etype.value):
                ev = RefereeEvent(
                    event_id="x_00001",
                    timestamp_s=1.0,
                    type=etype,
                    side="A",
                    position_px=CENTROID_PX,
                )
                with self.assertRaises(ValueError) as ctx:
                    self.h.locate_landing_from_event(ev)
                self.assertIn("§5", str(ctx.exception))

    def test_bounce_without_position_is_rejected(self):
        ev = RefereeEvent(event_id="bounce_1", timestamp_s=1.0, type=EventType.BOUNCE)
        with self.assertRaises(ValueError):
            self.h.locate_landing_from_event(ev)

    def test_non_event_input_is_rejected(self):
        with self.assertRaises(TypeError):
            self.h.locate_landing_from_event({"type": "BOUNCE"})


# ---------------------------------------------------------------------------
# 左右半台
# ---------------------------------------------------------------------------
class TestSide(unittest.TestCase):
    def setUp(self):
        self.h = TableHomography(CALIB)

    def test_side_by_cm(self):
        self.assertEqual(self.h.side_of_cm(50.0), "A")
        self.assertEqual(self.h.side_of_cm(200.0), "B")
        self.assertEqual(self.h.side_of_cm(136.9), "A")
        self.assertEqual(self.h.side_of_cm(137.1), "B")

    def test_calibration_net_consistency_is_reported(self):
        """标定里的 net_x 反推值应接近物理的 137cm（偏离大说明标定不自洽）。"""
        derived = self.h.calibration_net_x_cm()
        self.assertGreater(derived, 0.0)
        self.assertLess(derived, TABLE_LENGTH_CM)
        # game2 的标定质量：偏离 137cm 应在一个合理范围内
        self.assertLess(abs(derived - self.h.net_x_cm), 30.0)


# ---------------------------------------------------------------------------
# 分辨率无关性（多源数据）
# ---------------------------------------------------------------------------
class TestResolutionIndependence(unittest.TestCase):
    def test_same_physical_point_across_resolutions(self):
        full = TableHomography(CALIB, image_size=(1920, 1080))
        half = TableHomography(CALIB, image_size=(960, 540))

        pt_full = (931.5, 584.75)
        pt_half = (pt_full[0] * 0.5, pt_full[1] * 0.5)

        cm_full = full.to_table_cm(pt_full)
        cm_half = half.to_table_cm(pt_half)
        # 允许 int() 截断带来的少量偏差
        self.assertAlmostEqual(cm_full[0], cm_half[0], delta=2.0)
        self.assertAlmostEqual(cm_full[1], cm_half[1], delta=2.0)

    def test_calibration_rescaled_internally(self):
        h = TableHomography(CALIB, image_size=(960, 540))
        self.assertEqual(h.calibration.base_width, 960)
        self.assertEqual(h.calibration.roi_box.to_tuple(), (100, 425, 75, 890))


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------
class TestSerialization(unittest.TestCase):
    def test_landing_point_to_dict(self):
        import json

        h = TableHomography(CALIB)
        d = h.locate_landing(CENTROID_PX).to_dict()
        json.dumps(d)
        self.assertIn("x_cm", d)
        self.assertIn("in_table", d)
        self.assertIn("distance_to_edge_cm", d)

    def test_describe_is_json_friendly(self):
        import json

        h = TableHomography(CALIB)
        json.dumps(h.describe())
        self.assertIn("net_x_cm", h.describe())


if __name__ == "__main__":
    unittest.main(verbosity=2)
