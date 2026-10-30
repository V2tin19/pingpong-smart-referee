"""
tests/test_contracts.py
=======================

数据契约与适配器基类的单元测试。

覆盖组长方案里所有"写错了会判错分"的条款：

* §7.1  ``FrameObservation`` 的 JSON 形状与字段名
* §7.3  漏检必须 ``visible=False`` 且三个字段全 ``None``；(0, 0) 被拒绝
* §7.4  ``RefereeEvent`` 的字段与事件 ID 格式
* §7.5  时间对齐：``frame_id`` / ``timestamp_s`` 必须回显被预测帧
* §7.2  ``source`` / ``sides`` 的枚举严格性

只依赖标准库 + numpy，不需要 GPU、torch 或视频文件，可在任何分支环境跑::

    python tests/test_contracts.py -v
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import unittest

import numpy as np

# 让 `python tests/test_contracts.py` 也能找到 core/ 与 adapters/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_adapter import BaseAdapter
from core.geometry import (
    CALIBRATION_PRESETS,
    ROIBox,
    TableCalibration,
    guess_preset_name,
    preset,
)
from core.schemas import (
    BallObservation,
    EventScores,
    EventType,
    FrameObservation,
    RefereeEvent,
    Side,
    Source,
    make_event_id,
)


def make_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_observation(**overrides) -> FrameObservation:
    kwargs = dict(
        frame_id=1234,
        timestamp_s=20.567,
        source="yolo",
        image_size=(1920, 1080),
        ball=BallObservation(
            visible=True,
            center_px=(850.0, 420.0),
            bbox_xyxy=(844, 414, 856, 426),
            confidence=0.92,
        ),
    )
    kwargs.update(overrides)
    return FrameObservation(**kwargs)


# ---------------------------------------------------------------------------
# §7.1 观测形状
# ---------------------------------------------------------------------------
class TestFrameObservationShape(unittest.TestCase):
    def test_dict_matches_section_7_1_exactly(self):
        """to_dict() 的键集合必须与 §7.1 完全一致（不多不少）。"""
        d = make_observation().to_dict()
        self.assertEqual(
            set(d),
            {"frame_id", "timestamp_s", "source", "image_size", "ball", "event_scores"},
        )
        self.assertEqual(d["frame_id"], 1234)
        self.assertEqual(d["timestamp_s"], 20.567)
        self.assertEqual(d["source"], "yolo")
        self.assertEqual(d["image_size"], [1920, 1080])  # [width, height]
        self.assertEqual(
            d["ball"],
            {
                "visible": True,
                "center_px": [850.0, 420.0],
                "bbox_xyxy": [844.0, 414.0, 856.0, 426.0],
                "confidence": 0.92,
            },
        )
        self.assertEqual(d["event_scores"], {"bounce": None, "net_hit": None})

    def test_json_roundtrip_is_lossless(self):
        obs = make_observation()
        restored = FrameObservation.from_json(obs.to_json())
        self.assertEqual(restored.to_dict(), obs.to_dict())
        self.assertIsInstance(restored.ball.center_px, tuple)
        self.assertIsInstance(restored.image_size, tuple)

    def test_from_dict_accepts_the_leader_example(self):
        """直接吃组长方案 §7.1 里给的示例字面量。"""
        example = {
            "frame_id": 1234,
            "timestamp_s": 20.567,
            "source": "yolo",
            "image_size": [1920, 1080],
            "ball": {
                "visible": True,
                "center_px": [850.0, 420.0],
                "bbox_xyxy": [844, 414, 856, 426],
                "confidence": 0.92,
            },
            "event_scores": {"bounce": None, "net_hit": None},
        }
        obs = FrameObservation.from_dict(example)
        self.assertEqual(obs.frame_id, 1234)
        self.assertEqual(obs.ball.center_px, (850.0, 420.0))
        self.assertEqual(obs.to_dict(), example)

    def test_field_set_is_frozen_to_section_7_1(self):
        """§7.1 是**冻结接口**：字段集合不得增、删、改名。

        这条测试是防止"三分支各自加字段"导致统一观测退化的回归防线。
        若确实需要新字段，应先与组长协商修订 §7.1，再改这里。
        """
        names = [f.name for f in dataclasses.fields(FrameObservation)]
        self.assertEqual(
            names,
            ["frame_id", "timestamp_s", "source", "image_size", "ball", "event_scores"],
        )

    def test_serialized_keys_are_exactly_section_7_1(self):
        d = make_observation().to_dict()
        self.assertEqual(
            set(d), {"frame_id", "timestamp_s", "source", "image_size", "ball", "event_scores"}
        )
        self.assertEqual(set(d["ball"]), {"visible", "center_px", "bbox_xyxy", "confidence"})
        self.assertEqual(set(d["event_scores"]), {"bounce", "net_hit"})

    def test_extra_fields_are_rejected(self):
        """传 §7.1 之外的字段必须报错，避免契约悄悄扩张。"""
        with self.assertRaises(TypeError):
            make_observation(candidates=[BallObservation(visible=True, center_px=(1.0, 1.0))])

    def test_image_size_rejects_zero_and_negatives(self):
        for bad in [(0, 1080), (1920, -1)]:
            with self.assertRaises(ValueError):
                make_observation(image_size=bad)

    def test_source_is_strict(self):
        for bad in ["yolo11n", "YOLO11", "", "yolov8"]:
            with self.assertRaises(ValueError):
                make_observation(source=bad)
        self.assertEqual(make_observation(source=Source.YOLO).source, "yolo")
        self.assertEqual(make_observation(source="TrackNet").source, "tracknet")


# ---------------------------------------------------------------------------
# §7.3 漏检语义
# ---------------------------------------------------------------------------
class TestMissingDetection(unittest.TestCase):
    def test_invisible_with_center_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            BallObservation(visible=False, center_px=(100.0, 200.0))
        self.assertIn("§7.3", str(ctx.exception))

    def test_zero_zero_placeholder_is_rejected(self):
        """最常见的错误写法：(0, 0) 表示丢失。必须被拦下。"""
        with self.assertRaises(ValueError):
            BallObservation(visible=False, center_px=(0.0, 0.0))

    def test_invisible_with_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            BallObservation(visible=False, confidence=0.0)

    def test_invisible_with_bbox_is_rejected(self):
        with self.assertRaises(ValueError):
            BallObservation(visible=False, bbox_xyxy=(1, 2, 3, 4))

    def test_visible_requires_center(self):
        with self.assertRaises(ValueError):
            BallObservation(visible=True)

    def test_missing_helper_is_compliant(self):
        ball = BallObservation.missing()
        self.assertFalse(ball.visible)
        self.assertIsNone(ball.center_px)
        self.assertIsNone(ball.bbox_xyxy)
        self.assertIsNone(ball.confidence)
        self.assertEqual(
            ball.to_dict(),
            {"visible": False, "center_px": None, "bbox_xyxy": None, "confidence": None},
        )

    def test_tracknet_style_without_bbox_is_allowed(self):
        """TrackNet / TTNet 常常没有检测框，bbox 为 None 必须合法（§7.2）。"""
        ball = BallObservation(visible=True, center_px=(10.0, 20.0), confidence=0.5)
        self.assertIsNone(ball.bbox_xyxy)


# ---------------------------------------------------------------------------
# 数值校验
# ---------------------------------------------------------------------------
class TestNumericValidation(unittest.TestCase):
    def test_confidence_out_of_range(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                BallObservation(visible=True, center_px=(1.0, 2.0), confidence=bad)

    def test_non_finite_coordinates(self):
        for bad in ((float("nan"), 0.0), (float("inf"), 0.0)):
            with self.assertRaises(ValueError):
                BallObservation(visible=True, center_px=bad)

    def test_bbox_ordering(self):
        with self.assertRaises(ValueError):
            BallObservation(visible=True, center_px=(5.0, 5.0), bbox_xyxy=(10, 10, 2, 2))

    def test_event_scores_keep_none_vs_zero_distinct(self):
        """§7.2：None = 该模型不提供此输出，0.0 = 明确认为不是事件。不能混。"""
        self.assertIsNone(EventScores().bounce)
        self.assertEqual(EventScores(bounce=0.0).bounce, 0.0)
        with self.assertRaises(ValueError):
            EventScores(net_hit=2.0)


# ---------------------------------------------------------------------------
# §7.4 事件
# ---------------------------------------------------------------------------
class TestRefereeEvent(unittest.TestCase):
    def test_event_dict_matches_section_7_4(self):
        ev = RefereeEvent(
            event_id="bounce_00042",
            timestamp_s=20.567,
            type=EventType.BOUNCE,
            side=Side.B,
            position_px=(850, 420),
            confidence=0.90,
            sources=["ttnet", "tracknet"],
        )
        self.assertEqual(
            ev.to_dict(),
            {
                "event_id": "bounce_00042",
                "timestamp_s": 20.567,
                "type": "BOUNCE",
                "side": "B",
                "position_px": [850.0, 420.0],
                "confidence": 0.90,
                "sources": ["ttnet", "tracknet"],
            },
        )

    def test_event_roundtrip(self):
        ev = RefereeEvent(
            event_id="net_touch_00007",
            timestamp_s=1.5,
            type="NET_TOUCH",
            side="A",
            sources=["yolo"],
        )
        self.assertEqual(RefereeEvent.from_json(ev.to_json()).to_dict(), ev.to_dict())
        self.assertIsNone(RefereeEvent.from_json(ev.to_json()).position_px)

    def test_event_id_format(self):
        self.assertEqual(make_event_id(EventType.BOUNCE, 42), "bounce_00042")
        self.assertEqual(make_event_id("OUT", 0), "out_00000")
        self.assertEqual(make_event_id("racket_hit", 123456), "racket_hit_123456")
        with self.assertRaises(ValueError):
            make_event_id("SCORE", 1)

    def test_event_type_and_side_are_strict(self):
        with self.assertRaises(ValueError):
            RefereeEvent(event_id="x_1", timestamp_s=0.0, type="POINT")
        with self.assertRaises(ValueError):
            RefereeEvent(event_id="x_1", timestamp_s=0.0, type="BOUNCE", side="LEFT")

    def test_side_default_is_unknown(self):
        ev = RefereeEvent(event_id="x_1", timestamp_s=0.0, type="BOUNCE")
        self.assertEqual(ev.side, "UNKNOWN")

    def test_sources_rejects_bare_string(self):
        with self.assertRaises(TypeError):
            RefereeEvent(event_id="x_1", timestamp_s=0.0, type="BOUNCE", sources="yolo")


# ---------------------------------------------------------------------------
# 几何与标定
# ---------------------------------------------------------------------------
class TestGeometry(unittest.TestCase):
    def test_roi_crop_and_offset(self):
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        roi = ROIBox(y_min=10, y_max=60, x_min=20, x_max=120)
        crop, offset = roi.crop(frame)
        self.assertEqual(crop.shape, (50, 100, 3))
        self.assertEqual(offset, (20, 10))
        np.testing.assert_array_equal(crop, frame[10:60, 20:120])

    def test_roi_clip_into_frame(self):
        roi = ROIBox(y_min=50, y_max=500, x_min=100, x_max=900)
        clipped = roi.clip(200, 100)
        self.assertEqual(clipped.to_tuple(), (50, 100, 100, 200))
        self.assertEqual(clipped.offset, (100, 50))

    def test_roi_entirely_outside_raises(self):
        with self.assertRaises(ValueError):
            ROIBox(y_min=200, y_max=300, x_min=0, x_max=10).clip(100, 100)

    def test_roi_ordering_validation(self):
        with self.assertRaises(ValueError):
            ROIBox(y_min=50, y_max=10, x_min=0, x_max=10)

    def test_presets_match_legacy_hardcoded_values(self):
        """回归保护：预设必须与 smart_referee_yolo.py 里的硬编码标定逐值一致。"""
        expected = {
            "game2": ((557, 519), (1329, 535), (1473, 654), (367, 631), (200, 850, 150, 1780), 929, (450, 642)),
            "rally": ((587, 575), (1305, 450), (1630, 565), (765, 786), (350, 860, 80, 1850), 1148, (450, 680)),
            "plos": ((750, 440), (1010, 440), (1010, 560), (750, 560), (120, 560, 200, 1024), 880, (350, 560)),
        }
        for name, (p0, p1, p2, p3, roi, net_x, net_zone) in expected.items():
            calib = CALIBRATION_PRESETS[name]
            self.assertEqual(calib.table_polygon, (p0, p1, p2, p3), name)
            self.assertEqual(calib.roi_box.to_tuple(), roi, name)
            self.assertEqual(calib.net_x, net_x, name)
            self.assertEqual(calib.net_zone, net_zone, name)

    def test_scaled_to_halves_everything(self):
        half = preset("game2").scaled_to(960, 540)
        self.assertEqual(half.roi_box.to_tuple(), (100, 425, 75, 890))
        self.assertEqual(half.net_x, 464)
        self.assertEqual(half.net_zone, (225, 321))
        self.assertEqual(half.table_polygon[0], (278, 259))

    def test_scaled_to_identity_returns_self(self):
        calib = preset("game2")
        self.assertIs(calib.scaled_to(1920, 1080), calib)

    def test_side_of(self):
        self.assertEqual(preset("game2").side_of(100), "A")
        self.assertEqual(preset("game2").side_of(1500), "B")

    def test_calibration_roundtrip(self):
        calib = preset("rally")
        self.assertEqual(TableCalibration.from_dict(calib.to_dict()), calib)

    def test_guess_preset_name_is_only_a_convenience(self):
        self.assertEqual(guess_preset_name("test_rally.mp4"), "rally")
        self.assertEqual(guess_preset_name("test_plos.mp4"), "plos")
        self.assertEqual(guess_preset_name("test_game_2.avi"), "game2")
        self.assertEqual(guess_preset_name("brand-new-camera.mp4"), "game2")

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            preset("nope")


# ---------------------------------------------------------------------------
# BaseAdapter 模板方法
# ---------------------------------------------------------------------------
class _DummyAdapter(BaseAdapter):
    """用于验证基类校验逻辑的最小适配器。"""

    def __init__(self, behavior="ok", **kwargs):
        super().__init__(**kwargs)
        self.behavior = behavior

    @property
    def source(self):
        return "yolo"

    def _predict_frame(self, frame, frame_id, timestamp_s):
        h, w = frame.shape[:2]
        obs = self.missing_observation(frame_id, timestamp_s, (w, h))
        if self.behavior == "ok":
            obs.ball = BallObservation(visible=True, center_px=(1.0, 2.0), confidence=0.5)
        elif self.behavior == "miss":
            pass  # 原样返回 §7.3 合规的漏检观测
        elif self.behavior == "wrong_frame":
            obs.frame_id = frame_id + 1
        elif self.behavior == "wrong_time":
            obs.timestamp_s = timestamp_s + 0.5
        elif self.behavior == "wrong_source":
            obs.source = "ttnet"
        elif self.behavior == "wrong_size":
            obs.image_size = (w // 2, h // 2)
        elif self.behavior == "not_observation":
            return {"frame_id": frame_id}
        return obs


class TestBaseAdapter(unittest.TestCase):
    def test_happy_path(self):
        adapter = _DummyAdapter()
        obs = adapter.predict_frame(make_frame(), 0, 0.0)
        self.assertTrue(obs.ball.visible)
        self.assertEqual(adapter.frames_processed, 1)
        self.assertEqual(adapter.describe()["source"], "yolo")

    def test_missing_helper_is_compliant(self):
        obs = _DummyAdapter("miss").predict_frame(make_frame(), 7, 0.28)
        self.assertFalse(obs.ball.visible)
        self.assertEqual(obs.to_dict()["ball"]["center_px"], None)

    def test_scaling_uses_truncation_like_legacy_pipeline(self):
        """缩放必须沿用老管线的 int() 截断，否则历史指标无法逐像素复现。"""
        calib = preset("game2")
        half = calib.scaled_to(960, 540)
        # 519 * 0.5 = 259.5 -> 截断为 259（round() 会得到 260）
        self.assertEqual(half.table_polygon[0], (278, 259))
        # 850 * 0.5 = 425.0, 929 * 0.5 = 464.5 -> 464
        self.assertEqual(half.roi_box.to_tuple(), (100, 425, 75, 890))
        self.assertEqual(half.net_x, 464)

    def test_candidates_for_is_out_of_band_and_defaults_to_empty(self):
        """多候选是带外能力：默认没有，且绝不进入 §7.1 观测结构。"""
        adapter = _DummyAdapter()
        obs = adapter.predict_frame(make_frame(), 3, 0.12)
        self.assertEqual(adapter.candidates_for(3), [])
        self.assertNotIn("candidates", obs.to_dict())
        self.assertEqual(
            [f.name for f in dataclasses.fields(FrameObservation)],
            ["frame_id", "timestamp_s", "source", "image_size", "ball", "event_scores"],
        )

    def test_wrong_frame_id_is_caught(self):
        """§7.5：连续帧模型如果报错了帧号，必须立刻炸掉。"""
        with self.assertRaises(ValueError) as ctx:
            _DummyAdapter("wrong_frame").predict_frame(make_frame(), 5, 0.2)
        self.assertIn("frame_id", str(ctx.exception))

    def test_wrong_timestamp_is_caught(self):
        with self.assertRaises(ValueError) as ctx:
            _DummyAdapter("wrong_time").predict_frame(make_frame(), 5, 0.2)
        self.assertIn("timestamp_s", str(ctx.exception))

    def test_wrong_source_is_caught(self):
        with self.assertRaises(ValueError):
            _DummyAdapter("wrong_source").predict_frame(make_frame(), 5, 0.2)

    def test_model_input_size_leaking_is_caught(self):
        """适配器若误报裁剪后的尺寸，等于把模型输入尺寸泄漏给下游。"""
        with self.assertRaises(ValueError) as ctx:
            _DummyAdapter("wrong_size").predict_frame(make_frame(), 5, 0.2)
        self.assertIn("image_size", str(ctx.exception))

    def test_non_observation_return_is_caught(self):
        with self.assertRaises(TypeError):
            _DummyAdapter("not_observation").predict_frame(make_frame(), 5, 0.2)

    def test_strict_time_rejects_backwards_timestamps(self):
        adapter = _DummyAdapter(strict_time=True)
        adapter.predict_frame(make_frame(), 0, 1.0)
        with self.assertRaises(ValueError) as ctx:
            adapter.predict_frame(make_frame(), 1, 0.5)
        self.assertIn("backwards", str(ctx.exception))

    def test_non_strict_time_allows_looping_playback(self):
        adapter = _DummyAdapter(strict_time=False)
        adapter.predict_frame(make_frame(), 0, 1.0)
        adapter.predict_frame(make_frame(), 1, 0.5)  # 视频循环回放
        self.assertEqual(adapter.frames_processed, 2)

    def test_bad_inputs(self):
        adapter = _DummyAdapter()
        with self.assertRaises(TypeError):
            adapter.predict_frame(None, 0, 0.0)
        with self.assertRaises(TypeError):
            adapter.predict_frame(make_frame(), "0", 0.0)
        with self.assertRaises(ValueError):
            adapter.predict_frame(make_frame(), -1, 0.0)

    def test_reset_clears_state(self):
        adapter = _DummyAdapter(strict_time=True)
        adapter.predict_frame(make_frame(), 0, 5.0)
        adapter.reset()
        adapter.predict_frame(make_frame(), 0, 0.0)  # reset 后时间可以倒流
        self.assertEqual(adapter.frames_processed, 1)

    def test_abstract_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            BaseAdapter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 跨分支互通（与 TrackNet 分支交付的 schema 兼容性）
# ---------------------------------------------------------------------------
class TestCrossBranchInterop(unittest.TestCase):
    """与 TrackNet 分支交付的统一观测 schema 的互通性。

    TrackNet 的 schema 与 §7.1 一致，但**多了一个可选的 `ball.predicted` 字段**。
    本组测试锁定我们的处理方式：兼容读取真实数据，但拒绝"预测值伪装成检测值"。
    """

    TRACKNET_STYLE = {
        "frame_id": 42,
        "timestamp_s": 0.35,
        "source": "tracknet",
        "image_size": [1920, 1080],
        "ball": {
            "visible": True,
            "center_px": [850.0, 420.0],
            "bbox_xyxy": None,          # 热图模型没有检测框（§7.2）
            "confidence": 0.62,
        },
        "event_scores": {"bounce": None, "net_hit": None},   # 无事件能力（§7.2）
    }

    def test_tracknet_style_observation_parses(self):
        obs = FrameObservation.from_dict(self.TRACKNET_STYLE)
        self.assertEqual(obs.source, "tracknet")
        self.assertTrue(obs.ball.visible)
        self.assertIsNone(obs.ball.bbox_xyxy)
        self.assertIsNone(obs.event_scores.bounce)
        self.assertIsNone(obs.event_scores.net_hit)

    def test_roundtrip_preserves_tracknet_style(self):
        obs = FrameObservation.from_dict(self.TRACKNET_STYLE)
        self.assertEqual(obs.to_dict(), self.TRACKNET_STYLE)

    def test_predicted_flag_absent_is_fine(self):
        data = json.loads(json.dumps(self.TRACKNET_STYLE))
        self.assertNotIn("predicted", data["ball"])
        FrameObservation.from_dict(data)          # 不应抛错

    def test_predicted_false_is_fine(self):
        data = json.loads(json.dumps(self.TRACKNET_STYLE))
        data["ball"]["predicted"] = False
        obs = FrameObservation.from_dict(data)
        self.assertTrue(obs.ball.visible)

    def test_predicted_true_is_rejected_loudly(self):
        """预测值绝不能伪装成检测值（§7.3）。

        若静默接受，轨迹推断位置会被当成真实检出混进事件层——这比直接报错危险得多。
        """
        data = json.loads(json.dumps(self.TRACKNET_STYLE))
        data["ball"]["predicted"] = True
        with self.assertRaises(ValueError) as ctx:
            FrameObservation.from_dict(data)
        self.assertIn("§7.3", str(ctx.exception))
        self.assertIn("predicted", str(ctx.exception))

    def test_missing_detection_tracknet_style(self):
        """TrackNet 的漏检行：三字段全 null。"""
        miss = {
            "frame_id": 0,
            "timestamp_s": 0.0,
            "source": "tracknet",
            "image_size": [1920, 1080],
            "ball": {"visible": False, "center_px": None, "bbox_xyxy": None, "confidence": None},
            "event_scores": {"bounce": None, "net_hit": None},
        }
        obs = FrameObservation.from_dict(miss)
        self.assertFalse(obs.ball.visible)
        self.assertEqual(obs.to_dict(), miss)

    def test_event_keys_match_tracknet_event_schema(self):
        """TrackNet 的 event.schema.json 要求这 7 个键且 additionalProperties=false。"""
        ev = RefereeEvent(
            event_id="bounce_00001",
            timestamp_s=1.0,
            type="BOUNCE",
            side="A",
            position_px=(100.0, 200.0),
            confidence=0.9,
            sources=["yolo"],
        )
        self.assertEqual(
            set(ev.to_dict()),
            {"event_id", "timestamp_s", "type", "side", "position_px", "confidence", "sources"},
        )

    def test_event_types_match_tracknet_enum(self):
        """两边的物理事件枚举必须完全一致。"""
        self.assertEqual(
            {t.value for t in EventType},
            {"BOUNCE", "RACKET_HIT", "NET_TOUCH", "OUT"},
        )

    def test_sides_match_tracknet_enum(self):
        self.assertEqual({s.value for s in Side}, {"A", "B", "UNKNOWN"})

    def test_frame_id_and_side_conventions_agree(self):
        """两边独立收敛到同一约定：frame_id 从 0 起；side 为物理左右。"""
        self.assertEqual(FrameObservation.from_dict(self.TRACKNET_STYLE).frame_id, 42)
        self.assertGreaterEqual(FrameObservation(**make_observation().__dict__).frame_id, 0)
        self.assertEqual(Side.A.value, "A")
        self.assertEqual(Side.B.value, "B")


if __name__ == "__main__":
    unittest.main(verbosity=2)
