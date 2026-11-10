"""
tests/test_yolo_adapter.py
==========================

YOLO 适配器的**纯逻辑**单测（不需要 torch / ultralytics —— 模型是惰性加载的，
这里只测几何过滤与构造校验）。

重点：感知降噪 P1-2（球台 ROI 掩码 / 地板检出丢弃，Test4 报告 §四.1 第 2 项）。

真值依据（``tools/analyze_gt_envelope.py``，test_4 官方标注）：
真球全程最低点 y=766（球台近边 750 以下仅 16px、504 帧中仅 5 帧触及），
而地板反光 / 球鞋误检全部位于该区域以下——掩码在近边以下 30px 裁切
**零召回损失**。

用法::

    python tests/test_yolo_adapter.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.yolo_adapter import YOLOAdapter
from core.geometry import preset
from core.schemas import BallObservation

WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "yolo11n_clean_best.pt",
)


def _det(x: float, y: float, conf: float = 0.9) -> BallObservation:
    return BallObservation(visible=True, center_px=(x, y), confidence=conf)


def _skip_if_no_weights(test):
    if not os.path.isfile(WEIGHTS):
        return unittest.skip("weights/yolo11n_clean_best.pt 不存在（克隆未携带权重）")(test)
    return test


# ---------------------------------------------------------------------------
# 感知降噪 P1-2：球台近边以下的地板检出丢弃
# ---------------------------------------------------------------------------
class TestFloorMask(unittest.TestCase):
    """test4 标定：近边 y=750（base 1920x1080）。"""

    def _adapter(self, margin=30.0):
        return YOLOAdapter(
            WEIGHTS, calibration=preset("test4"), floor_margin_px=margin
        )

    @_skip_if_no_weights
    def test_floor_detections_are_dropped(self):
        """近边以下超过 margin 的检出被物理丢弃；台面与真球最低点(766)保留。"""
        ad = self._adapter(margin=30.0)          # 丢弃线 = 750 + 30 = 780
        ad._resolve_roi(1920, 1080)               # 设定已缩放标定（生产路径同源）
        kept = ad._reject_below_table(
            [_det(900, 749.0), _det(900, 766.0), _det(900, 780.0),
             _det(900, 781.0), _det(900, 900.0)],
            frame_height=1080,
        )
        self.assertEqual([d.center_px[1] for d in kept], [749.0, 766.0, 780.0])
        self.assertEqual(ad.floor_rejected_count, 2)

    @_skip_if_no_weights
    def test_mask_scales_with_resolution(self):
        """半分辨率下丢弃线按比例缩放（375 + 15 = 390）。"""
        ad = self._adapter(margin=30.0)
        ad._resolve_roi(960, 540)
        kept = ad._reject_below_table(
            [_det(480, 389.0), _det(480, 391.0)],
            frame_height=540,
        )
        self.assertEqual([d.center_px[1] for d in kept], [389.0])
        self.assertEqual(ad.floor_rejected_count, 1)

    @_skip_if_no_weights
    def test_mask_off_by_default(self):
        """默认关闭：不设 floor_margin_px 时 describe 不报掩码。"""
        ad = YOLOAdapter(WEIGHTS, calibration=preset("test4"))
        self.assertIsNone(ad.describe()["floor_margin_px"])

    @_skip_if_no_weights
    def test_zero_margin_keeps_only_above_near_edge(self):
        """margin=0：只保留近边及以上（严格模式）。"""
        ad = self._adapter(margin=0.0)
        ad._resolve_roi(1920, 1080)
        kept = ad._reject_below_table(
            [_det(900, 750.0), _det(900, 751.0)], frame_height=1080
        )
        self.assertEqual([d.center_px[1] for d in kept], [750.0])

    @_skip_if_no_weights
    def test_negative_margin_rejected(self):
        with self.assertRaises(ValueError):
            YOLOAdapter(WEIGHTS, calibration=preset("test4"), floor_margin_px=-1.0)

    @_skip_if_no_weights
    def test_margin_requires_calibration(self):
        """roi_box 模式没有球台几何，无法定义"地板"→ 显式拒绝而非静默忽略。"""
        with self.assertRaises(ValueError):
            YOLOAdapter(WEIGHTS, roi_box=(320, 920, 140, 1650), floor_margin_px=30.0)

    @_skip_if_no_weights
    def test_counter_exposed_in_describe(self):
        ad = self._adapter(margin=30.0)
        ad._resolve_roi(1920, 1080)
        ad._reject_below_table([_det(900, 850.0)], frame_height=1080)
        d = ad.describe()
        self.assertEqual(d["floor_rejected"], 1)
        self.assertEqual(d["floor_margin_px"], 30.0)

    @_skip_if_no_weights
    def test_reset_clears_counter(self):
        ad = self._adapter(margin=30.0)
        ad._resolve_roi(1920, 1080)
        ad._reject_below_table([_det(900, 850.0)], frame_height=1080)
        ad.reset()
        self.assertEqual(ad.floor_rejected_count, 0)


class TestInferenceConfiguration(unittest.TestCase):
    def test_half_and_device_forwarded_and_missing_clears_candidates(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as weights:
            adapter = YOLOAdapter(weights.name, imgsz=640, device="0", half=True)
            model = Mock(return_value=[])
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            with patch.object(adapter, "_ensure_model", return_value=model), patch.object(
                adapter, "_extract_detections", side_effect=[[_det(30, 40)], []]
            ):
                first = adapter.predict_frame(frame, 0, 0.0)
                self.assertTrue(first.ball.visible)
                self.assertEqual(len(adapter.candidates_for(0)), 1)
                # Seeking/retrying may reuse a frame ID; no old candidates may survive.
                second = adapter.predict_frame(frame, 0, 0.0)
                self.assertFalse(second.ball.visible)
                self.assertEqual(adapter.candidates_for(0), [])
            kwargs = model.call_args.kwargs
            self.assertEqual(kwargs["imgsz"], 640)
            self.assertEqual(kwargs["device"], "0")
            self.assertTrue(kwargs["half"])
            self.assertTrue(adapter.describe()["half"])

    def test_fp32_remains_default(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as weights:
            adapter = YOLOAdapter(weights.name)
            self.assertFalse(adapter.half)
            self.assertEqual(adapter._precision_kwargs, {})

    def test_half_uses_supported_ultralytics_precision_argument(self):
        from types import SimpleNamespace
        for config, expected in [({}, {"half": True}), ({"quantize": None}, {"quantize": 16})]:
            with self.subTest(config=config), tempfile.NamedTemporaryFile(suffix=".pt") as weights:
                adapter = YOLOAdapter(weights.name, half=True)
                with patch.dict(sys.modules, {
                    "ultralytics": SimpleNamespace(YOLO=Mock()),
                    "ultralytics.cfg": SimpleNamespace(DEFAULT_CFG_DICT=config),
                }):
                    adapter._ensure_model()
                self.assertEqual(adapter._precision_kwargs, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
