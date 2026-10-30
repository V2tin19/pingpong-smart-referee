"""
tests/test_trajectory_processor.py
==================================

轨迹与事件处理层的单元测试。

重点验证组长方案的硬性要求**真的落地了**，而不只是写在注释里：

* **§4-1** 时间戳而非固定帧数 —— 同一物理过程在 25fps / 50fps 下行为必须一致
* **§4-2** 球消失不得产生失分事件 —— 只认正向证据
* **§5**   本层不做任何单应性投影（只输出像素坐标）
* **§7.5** 事件去重
* 关键不变量：**只有真实观测点能产生事件**（PREDICTED 点不行）
* 关键不变量：**新目标必须经双帧位移确认**才能建立跟踪

用合成轨迹测试，不需要 GPU、视频或模型::

    python tests/test_trajectory_processor.py -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import TableCalibration, preset
from core.homography import TableHomography
from core.schemas import BallObservation, EventType, FrameObservation
from core.trajectory_processor import (
    MotionState,
    ProcessorParams,
    TrajectoryEventProcessor,
)

CALIB = preset("game2")          # net_x=929, net_zone=(450,642), 1920x1080
FPS = 25.0


# ---------------------------------------------------------------------------
# 合成观测工具
# ---------------------------------------------------------------------------
def obs(x: float, y: float, frame_id: int, fps: float = FPS,
        width: int = 1920, height: int = 1080, conf: float = 0.9) -> FrameObservation:
    return FrameObservation(
        frame_id=frame_id,
        timestamp_s=frame_id / fps,
        source="yolo",
        image_size=(width, height),
        ball=BallObservation(visible=True, center_px=(x, y), confidence=conf),
    )


def miss(frame_id: int, fps: float = FPS,
         width: int = 1920, height: int = 1080) -> FrameObservation:
    return FrameObservation(
        frame_id=frame_id,
        timestamp_s=frame_id / fps,
        source="yolo",
        image_size=(width, height),
        ball=BallObservation.missing(),
    )


def run(proc: TrajectoryEventProcessor, frames):
    """喂入一串观测，返回 (每帧事件列表, 全部事件)。"""
    per_frame = []
    for ob in frames:
        per_frame.append(proc.update(ob))
    return per_frame, [e for evs in per_frame for e in evs]


def count(events, etype: EventType) -> int:
    return sum(1 for e in events if e.type == etype.value)


# ---------------------------------------------------------------------------
# 门限：双帧位移确认
# ---------------------------------------------------------------------------
class TestTwoFrameConfirmation(unittest.TestCase):
    def test_single_isolated_detection_does_not_start_track(self):
        """单帧孤立噪点不得起轨。"""
        proc = TrajectoryEventProcessor(CALIB)
        _, events = run(proc, [obs(900, 580, 0), miss(1), miss(2), miss(3)])
        self.assertFalse(proc.is_tracking)
        self.assertEqual(proc.stats["confirmed_tracks"], 0)
        self.assertEqual(events, [])

    def test_static_object_never_confirms(self):
        """静止反光（位移 < 3.5px）永远无法通过确认门限。"""
        proc = TrajectoryEventProcessor(CALIB)
        run(proc, [obs(900, 580, i) for i in range(12)])
        self.assertEqual(proc.stats["confirmed_tracks"], 0)
        self.assertFalse(proc.is_tracking)

    def test_moving_pair_confirms(self):
        """位移在物理合理区间（3.5~65px）的连续两帧可以起轨。"""
        proc = TrajectoryEventProcessor(CALIB)
        run(proc, [obs(700, 580, 0), obs(720, 580, 1)])
        self.assertTrue(proc.is_tracking)
        self.assertEqual(proc.stats["confirmed_tracks"], 1)

    def test_too_fast_pair_rejected(self):
        """速度超出物理上限的两帧不确认（可能是两个不同物体）。"""
        proc = TrajectoryEventProcessor(CALIB)
        # 25fps 下 1 帧移动 500px → 12500 px/s，超过 confirm_max_speed_px_s(8000)
        run(proc, [obs(300, 580, 0), obs(800, 580, 1)])
        self.assertFalse(proc.is_tracking)

    def test_fast_smash_pair_is_accepted(self):
        """**回归测试**：真实快球（如 7500 px/s）必须能起轨。

        老代码的上限是"65px/帧 @25fps"=1625 px/s，那**只够覆盖 test_game_2 那种慢球**；
        实测 test_2 里球速最大 5106 px/s，用老上限会把快球全部拒掉。
        """
        proc = TrajectoryEventProcessor(CALIB)
        # 25fps 下 1 帧移动 300px → 7500 px/s，在物理合理区间内
        run(proc, [obs(500, 580, 0), obs(800, 580, 1)])
        self.assertTrue(proc.is_tracking)

    def test_confirmation_is_framerate_independent(self):
        """同一物理速度在不同帧率下都应能起轨（这是改成速度制门限的目的）。"""
        for fps in (25.0, 60.0, 120.0, 240.0):
            with self.subTest(fps=fps):
                proc = TrajectoryEventProcessor(CALIB)
                speed = 2000.0                      # px/s
                dt = 1.0 / fps
                run(proc, [
                    obs(500.0, 580.0, 0, fps=fps),
                    obs(500.0 + speed * dt, 580.0, 1, fps=fps),
                ])
                self.assertTrue(proc.is_tracking, f"{fps}fps 下 {speed}px/s 的球应能起轨")


# ---------------------------------------------------------------------------
# 事件：击球 / 落台
# ---------------------------------------------------------------------------
class TestEventExtraction(unittest.TestCase):
    def test_horizontal_reversal_emits_one_racket_hit(self):
        proc = TrajectoryEventProcessor(CALIB)
        xs = [500, 560, 620, 680, 740, 800, 740, 680, 620, 560, 500]
        _, events = run(proc, [obs(x, 584, i) for i, x in enumerate(xs)])
        self.assertEqual(count(events, EventType.RACKET_HIT), 1)
        self.assertEqual(count(events, EventType.BOUNCE), 0)

    def test_vertical_bounce_emits_one_bounce(self):
        proc = TrajectoryEventProcessor(CALIB)
        ys = [500, 515, 530, 545, 560, 545, 530, 515, 500]
        _, events = run(proc, [obs(931, y, i) for i, y in enumerate(ys)])
        self.assertEqual(count(events, EventType.BOUNCE), 1)

    def test_bounce_side_matches_net_x(self):
        """落点在球网左侧应报 side=A，右侧报 side=B。"""
        left = TrajectoryEventProcessor(CALIB)
        ys = [500, 515, 530, 545, 560, 545, 530, 515, 500]
        _, ev_left = run(left, [obs(700, y, i) for i, y in enumerate(ys)])
        bounces_left = [e for e in ev_left if e.type == "BOUNCE"]
        self.assertTrue(bounces_left)
        self.assertEqual(bounces_left[0].side, "A")

        right = TrajectoryEventProcessor(CALIB)
        _, ev_right = run(right, [obs(1200, y, i) for i, y in enumerate(ys)])
        bounces_right = [e for e in ev_right if e.type == "BOUNCE"]
        self.assertTrue(bounces_right)
        self.assertEqual(bounces_right[0].side, "B")

    def test_bounce_side_uses_table_cm_not_pixel_net_x(self):
        """**回归测试**：落台侧归属按**球台物理坐标**（x_cm 与 137cm 球网线比较），
        不按像素域的 ``net_x`` 比较。

        透视下图像上的一条竖直线并不对应球台平面上恒定的物理 x：game2 标定里
        ``net_x=929px``，但 y≈549 处的 929px 实际映射到 134cm（球台左半台）。
        近网落点按像素判会把 A 侧落台判成 B，进而让状态机把正常对打误判成
        OWN_COURT——与 R-017"击球方按飞行方向归属"是同一类修正。
        """
        proc = TrajectoryEventProcessor(CALIB)
        # x 匀速向右（不触发击球判据），y 先降后升（落台反弹），拐点在球网附近
        xs = [900, 906, 912, 918, 924, 930, 936, 942, 948]
        ys = [500, 515, 530, 545, 560, 545, 530, 515, 500]
        _, events = run(proc, [obs(x, y, i) for i, (x, y) in enumerate(zip(xs, ys))])
        bounces = [e for e in events if e.type == EventType.BOUNCE.value]
        self.assertTrue(bounces, "应检出一次落台")

        homography = TableHomography(CALIB)
        for bounce in bounces:
            landing = homography.locate_landing(bounce.position_px)
            # 事件自洽：side 必须等于按物理坐标的判侧
            self.assertEqual(
                bounce.side, homography.side_of_cm(landing.x_cm),
                f"落台 side 应按球台物理坐标判定，got side={bounce.side} at {bounce.position_px}",
            )
            # 本测试点必须真的落在"像素判侧 ≠ 物理判侧"的楔形区，
            # 否则两条判法结果相同，本测试退化为摆设
            self.assertNotEqual(
                bounce.side, CALIB.side_of(bounce.position_px[0]),
                "测试点应选在像素/物理判侧分歧区（近网），否则本测试无效",
            )

    def test_straight_line_emits_nothing(self):
        """匀速直线飞行不应产生任何事件。"""
        proc = TrajectoryEventProcessor(CALIB)
        _, events = run(proc, [obs(600 + 20 * i, 560, i) for i in range(12)])
        self.assertEqual(events, [])

    def test_racket_hit_and_bounce_are_mutually_exclusive(self):
        """**回归测试**：球拍击球与落台不得同帧双触发。

        球拍击球同样会造成垂直速度反转（球员把球往上打），所以只靠"垂直拐点"
        分不出落台与击球。二者的本质差别在水平方向：落台时水平速度保持，
        击球时水平方向反向。

        这个 bug 的实测后果：``RACKET_HIT`` 与 ``BOUNCE`` 同帧双触发，导致状态机里
        "击球方"与"落台侧"变成同一侧，正常对打被误判成 12 次 OWN_COURT
        （即"击球方把球打进自己半台"）。
        """
        proc = TrajectoryEventProcessor(CALIB)
        # 水平强烈反向 + 垂直同时拐头 → 只应判击球，不得判落台
        xs = [500, 560, 620, 680, 740, 800, 740, 680, 620, 560, 500]
        ys = [560, 560, 560, 560, 560, 560, 540, 520, 500, 480, 460]
        _, events = run(proc, [obs(x, y, i) for i, (x, y) in enumerate(zip(xs, ys))])
        self.assertEqual(count(events, EventType.RACKET_HIT), 1)
        self.assertEqual(count(events, EventType.BOUNCE), 0, "击球不得被误判为落台")

    def test_racket_hit_side_follows_outgoing_direction(self):
        """**回归测试**：击球方按"球被击出后的飞行方向"归属，而不是按检出位置。

        物理依据：球员一定把球打向对方半台。球往右飞 → 击球方在左(A)；
        球往左飞 → 击球方在右(B)。

        为什么不能用位置：速度反转是在实际击球**之后几帧**才被检出的，而球很快
        （120fps 下约 10px/帧），检出时球已越过球网，`side_of(x)` 会把击球方判成
        对面那一侧，进而让状态机把正常对打误判成 OWN_COURT。
        """
        proc = TrajectoryEventProcessor(CALIB)
        # 在球网右侧(x>929)发生"向左→向右"的反转 → 击球方应为左侧 A
        xs = [1300, 1250, 1200, 1150, 1100, 1050, 1100, 1150, 1200, 1250, 1300]
        _, events = run(proc, [obs(x, 470, i) for i, x in enumerate(xs)])
        hits = [e for e in events if e.type == EventType.RACKET_HIT.value]
        self.assertTrue(hits, "应检出一次击球")
        self.assertEqual(hits[0].side, "A", "球被击出后向右飞 → 击球方在左(A)")

    def test_racket_hit_side_is_b_when_ball_flies_left(self):
        proc = TrajectoryEventProcessor(CALIB)
        # 在球网左侧发生"向右→向左"的反转 → 击球方应为右侧 B
        xs = [600, 650, 700, 750, 800, 850, 800, 750, 700, 650, 600]
        _, events = run(proc, [obs(x, 470, i) for i, x in enumerate(xs)])
        hits = [e for e in events if e.type == EventType.RACKET_HIT.value]
        self.assertTrue(hits, "应检出一次击球")
        self.assertEqual(hits[0].side, "B", "球被击出后向左飞 → 击球方在右(B)")

    def test_event_fields_are_wellformed(self):
        proc = TrajectoryEventProcessor(CALIB, source="yolo")
        xs = [500, 560, 620, 680, 740, 800, 740, 680, 620, 560, 500]
        _, events = run(proc, [obs(x, 584, i) for i, x in enumerate(xs)])
        ev = events[0]
        self.assertTrue(ev.event_id.startswith("racket_hit_"))
        self.assertEqual(ev.sources, ["yolo"])
        self.assertIn(ev.side, ("A", "B", "UNKNOWN"))
        self.assertIsNotNone(ev.position_px)
        self.assertGreaterEqual(ev.timestamp_s, 0.0)

    def test_no_homography_projection_in_this_layer(self):
        """§5：本层只输出像素坐标，绝不输出厘米坐标。"""
        proc = TrajectoryEventProcessor(CALIB)
        ys = [500, 515, 530, 545, 560, 545, 530, 515, 500]
        _, events = run(proc, [obs(931, y, i) for i, y in enumerate(ys)])
        self.assertTrue(events)
        for ev in events:
            x, y = ev.position_px
            # 若被错误地做了单应性投影，坐标会落在 274x152.5 的小范围内
            self.assertGreater(x, 100.0)
            self.assertGreater(y, 100.0)
            self.assertNotIn("cm", ev.to_dict())


# ---------------------------------------------------------------------------
# §4-2 球消失不得判失分
# ---------------------------------------------------------------------------
class TestBallDisappearanceNeverScores(unittest.TestCase):
    def test_disappearance_produces_no_out_event(self):
        """球在台内消失 3 秒，不得产生 OUT 事件（组长 §4-2 硬性要求）。"""
        proc = TrajectoryEventProcessor(CALIB)
        frames = [obs(700 + 20 * i, 584, i) for i in range(8)]
        frames += [miss(i) for i in range(8, 8 + 75)]      # 3 秒无球
        _, events = run(proc, frames)
        self.assertEqual(count(events, EventType.OUT), 0)
        self.assertEqual(events, [])

    def test_disappearance_does_not_break_into_false_bounce(self):
        """消失期间的外推点不得产生落台事件。"""
        proc = TrajectoryEventProcessor(CALIB)
        frames = [obs(700, 520 + 12 * i, i) for i in range(8)]   # 持续下落
        frames += [miss(i) for i in range(8, 60)]
        _, events = run(proc, frames)
        self.assertEqual(count(events, EventType.BOUNCE), 0)

    def test_out_requires_positive_evidence(self):
        """OUT 只在"台外发生落点拐点"时报出（强证据），而非"在台外下降"。"""
        proc = TrajectoryEventProcessor(CALIB)
        # y=584 时台面横向范围约 [447, 1388]；x=200 明显在台外，且发生方向反转。
        # 同时 y=500~560 位于球台近边(654)之上 → 属病态区，需放宽区域限制才报 OUT。
        ys = [500, 520, 540, 560, 540, 520, 500]
        _, events = run(proc, [obs(200, y, i) for i, y in enumerate(ys)])
        self.assertEqual(count(events, EventType.OUT), 0, "远端压缩区不应报 OUT")
        self.assertEqual(count(events, EventType.BOUNCE), 0)

    def test_out_reported_in_well_conditioned_near_region(self):
        """球落到球台近边之下（良态区）时，台外拐点应报 OUT。"""
        from core.homography import TableHomography

        h = TableHomography(CALIB)
        near_edge_y = max(p[1] for p in CALIB.table_polygon)     # 654
        # 取台面左侧外 20cm、且图像 y 在近边之下的位置
        px = h.to_image_px(-20.0, 40.0)
        base_y = max(float(px[1]), float(near_edge_y) + 30.0)
        offsets = [-30, -15, 0, 15, 0, -15, -30]
        proc = TrajectoryEventProcessor(CALIB)
        _, events = run(proc, [obs(px[0], base_y + dy, i) for i, dy in enumerate(offsets)])
        self.assertEqual(count(events, EventType.OUT), 1)

    def test_out_region_can_be_disabled_by_param(self):
        """关掉区域限制后，远端压缩区的台外拐点会被报出来（对照）。"""
        from core.trajectory_processor import ProcessorParams

        loose = ProcessorParams(out_requires_near_region=False)
        proc = TrajectoryEventProcessor(CALIB, params=loose)
        ys = [500, 520, 540, 560, 540, 520, 500]
        _, events = run(proc, [obs(200, y, i) for i, y in enumerate(ys)])
        self.assertEqual(count(events, EventType.OUT), 1)

    def test_descending_outside_table_without_bounce_is_not_out(self):
        """仅在台外下降（还没落地）不得报 OUT——这正是误报的来源。"""
        proc = TrajectoryEventProcessor(CALIB)
        frames = [obs(200, 500 + 15 * i, i) for i in range(10)]   # 一路向下，无反转
        _, events = run(proc, frames)
        self.assertEqual(count(events, EventType.OUT), 0)

    def test_bounce_out_split_uses_centimetres(self):
        """BOUNCE / OUT 的分流在**物理厘米空间**判定，而非像素。

        像素阈值在透视下各向异性（长边 274cm≈1100px，短边 152.5cm≈135px），
        同一个像素容差在两个方向上代表的物理距离差近一个数量级。

        本用例关掉"良态区域"限制，单独验证厘米空间的分流；区域限制本身由
        :meth:`test_out_reported_in_well_conditioned_near_region` 覆盖。
        """
        from core.homography import TableHomography
        from core.trajectory_processor import ProcessorParams

        h = TableHomography(CALIB)
        offsets = [-30, -15, 0, 15, 0, -15, -30]     # 一个 V 形（先下后上）
        no_region_gate = ProcessorParams(out_requires_near_region=False)

        # 台面中心附近 → 应判 BOUNCE
        inside = h.to_image_px(137.0, 76.25)
        proc_in = TrajectoryEventProcessor(CALIB, params=no_region_gate)
        _, ev_in = run(
            proc_in, [obs(inside[0], inside[1] + dy, i) for i, dy in enumerate(offsets)]
        )
        self.assertEqual(count(ev_in, EventType.BOUNCE), 1, "台面内落点应为 BOUNCE")
        self.assertEqual(count(ev_in, EventType.OUT), 0)

        # 长边外侧 20cm（超过 out_margin_cm=6）→ 应判 OUT
        outside = h.to_image_px(-20.0, 76.25)
        proc_out = TrajectoryEventProcessor(CALIB, params=no_region_gate)
        _, ev_out = run(
            proc_out, [obs(outside[0], outside[1] + dy, i) for i, dy in enumerate(offsets)]
        )
        self.assertEqual(count(ev_out, EventType.OUT), 1, "台外 20cm 落点应为 OUT")
        self.assertEqual(count(ev_out, EventType.BOUNCE), 0)

        # 长边外侧 3cm（在 out_margin_cm=6 之内）→ 仍应判 BOUNCE（擦边容差）
        edge = h.to_image_px(-3.0, 76.25)
        proc_edge = TrajectoryEventProcessor(CALIB, params=no_region_gate)
        _, ev_edge = run(
            proc_edge, [obs(edge[0], edge[1] + dy, i) for i, dy in enumerate(offsets)]
        )
        self.assertEqual(count(ev_edge, EventType.BOUNCE), 1, "擦边球应判 BOUNCE")

    def test_out_margin_is_not_scaled_by_resolution(self):
        """厘米容差不参与分辨率缩放（这正是不用像素阈值的好处）。"""
        base = ProcessorParams()
        half = base.scaled(960, 540)
        self.assertEqual(half.out_margin_cm, base.out_margin_cm)


# ---------------------------------------------------------------------------
# 关键不变量：只有观测点能产生事件
# ---------------------------------------------------------------------------
class TestPredictedPointsNeverEmitEvents(unittest.TestCase):
    def test_extrapolated_points_emit_no_events(self):
        """卡尔曼外推期间必须零事件（修老代码 is_pred 失效的 bug）。"""
        proc = TrajectoryEventProcessor(CALIB)
        frames = [obs(500 + 40 * i, 584, i) for i in range(6)]
        frames += [miss(i) for i in range(6, 10)]     # 短时外推窗口内
        per_frame, events = run(proc, frames)
        self.assertGreater(proc.stats["predicted"], 0, "应当发生了外推")
        for idx in range(6, 10):
            self.assertEqual(per_frame[idx], [], f"第 {idx} 帧是外推帧，不得产生事件")

    def test_trajectory_marks_predicted_state(self):
        proc = TrajectoryEventProcessor(CALIB)
        frames = [obs(500 + 40 * i, 584, i) for i in range(6)]
        frames += [miss(6), miss(7)]
        run(proc, frames)
        states = {p.state for p in proc.trajectory}
        self.assertIn(MotionState.MEASURED, states)
        self.assertIn(MotionState.PREDICTED, states)
        # 观测点与推算点必须是可区分的
        predicted = [p for p in proc.trajectory if p.state == MotionState.PREDICTED]
        self.assertTrue(all(not p.is_observed for p in predicted))

    def test_predicted_points_carry_real_frame_id(self):
        """**回归测试**：外推点对应的就是"当前这一帧"，frame_id 必须是真实帧号。

        修复前写死 -1 哨兵：可视化层按帧号间隔做轨迹分段时，会把
        "观测→外推→观测"误判为断连（|-1 - 100| 远大于阈值），尾迹画不连续。
        """
        proc = TrajectoryEventProcessor(CALIB)
        frames = [obs(500 + 40 * i, 584, i) for i in range(6)]
        frames += [miss(6), miss(7)]
        frames += [obs(500 + 40 * 8, 584, 8)]
        run(proc, frames)
        predicted = [p for p in proc.trajectory if p.state == MotionState.PREDICTED]
        self.assertTrue(predicted, "应当产生了外推点")
        self.assertTrue(
            all(p.frame_id >= 0 for p in predicted),
            "外推点必须携带真实帧号，不得用 -1 哨兵",
        )
        self.assertEqual([p.frame_id for p in predicted], [6, 7])


# ---------------------------------------------------------------------------
# §4-1 帧率无关性
# ---------------------------------------------------------------------------
class TestFrameRateIndependence(unittest.TestCase):
    def test_short_gap_keeps_track_at_both_frame_rates(self):
        """0.04s 的短暂丢失，在 25fps 与 50fps 下都应保持跟踪。"""
        for fps in (25.0, 50.0):
            with self.subTest(fps=fps):
                proc = TrajectoryEventProcessor(CALIB)
                frames = [obs(500 + 20 * i, 584, i, fps=fps) for i in range(6)]
                gap_frames = int(round(0.04 * fps))
                frames += [miss(6 + k, fps=fps) for k in range(gap_frames)]
                frames += [obs(500 + 20 * (6 + gap_frames), 584, 6 + gap_frames, fps=fps)]
                run(proc, frames)
                self.assertTrue(proc.is_tracking, f"{fps}fps 下不应断轨")

    def test_long_gap_breaks_track_at_both_frame_rates(self):
        """0.20s 的丢失，在 25fps 与 50fps 下都应断轨（老代码按帧数会不一致）。"""
        for fps in (25.0, 50.0):
            with self.subTest(fps=fps):
                proc = TrajectoryEventProcessor(CALIB)
                frames = [obs(500 + 20 * i, 584, i, fps=fps) for i in range(6)]
                gap_frames = int(round(0.20 * fps))
                frames += [miss(6 + k, fps=fps) for k in range(gap_frames)]
                run(proc, frames)
                self.assertFalse(proc.is_tracking, f"{fps}fps 下应已断轨")

    def test_event_count_same_at_both_frame_rates(self):
        """同一物理轨迹（同一 px/s）在两种帧率下应产生相同数量的事件。"""
        counts = {}
        for fps in (25.0, 50.0):
            proc = TrajectoryEventProcessor(CALIB)
            # 用"时间"参数化轨迹，保证两种帧率下走的是同一条物理曲线
            n = int(round(0.5 * fps))
            frames = []
            for k in range(n):
                t = k / fps
                x = 500 + 800 * t                  # 1600 px/s 匀速
                if t > 0.25:                       # 0.25s 处水平反向
                    x = 500 + 800 * 0.25 - 800 * (t - 0.25)
                frames.append(obs(x, 584, k, fps=fps))
            _, events = run(proc, frames)
            counts[fps] = count(events, EventType.RACKET_HIT)
        self.assertEqual(counts[25.0], counts[50.0])
        self.assertEqual(counts[25.0], 1)


# ---------------------------------------------------------------------------
# 分辨率无关性（多源数据）
# ---------------------------------------------------------------------------
class TestResolutionIndependence(unittest.TestCase):
    def test_scaled_trajectory_behaves_identically(self):
        """同一条归一化轨迹在 1080p 与 540p 下事件数应一致。"""
        ys_1080 = [500, 515, 530, 545, 560, 545, 530, 515, 500]
        counts = {}
        for width, height, s in ((1920, 1080, 1.0), (960, 540, 0.5)):
            proc = TrajectoryEventProcessor(CALIB)
            frames = [
                obs(931 * s, y * s, i, width=width, height=height)
                for i, y in enumerate(ys_1080)
            ]
            _, events = run(proc, frames)
            counts[(width, height)] = count(events, EventType.BOUNCE)
        self.assertEqual(counts[(1920, 1080)], counts[(960, 540)])
        self.assertEqual(counts[(1920, 1080)], 1)

    def test_pixel_thresholds_scale(self):
        """空间阈值必须随分辨率缩放，否则小分辨率下会失效。"""
        base = ProcessorParams()
        half = base.scaled(960, 540)
        self.assertAlmostEqual(half.gate_near_px, base.gate_near_px * 0.5, places=6)
        self.assertAlmostEqual(half.spike_dev_px, base.spike_dev_px * 0.5, places=6)
        # 时间阈值不受分辨率影响
        self.assertEqual(half.cooldown_bounce_s, base.cooldown_bounce_s)
        self.assertEqual(half.dedup_window_s, base.dedup_window_s)


# ---------------------------------------------------------------------------
# 去重与冷却
# ---------------------------------------------------------------------------
class TestDedupAndCooldown(unittest.TestCase):
    def test_duplicate_bounce_within_window_is_suppressed(self):
        """同一次落台不得产生两条事件（§7.5）。"""
        proc = TrajectoryEventProcessor(CALIB)
        # 连续两个 V 形，间隔 2 帧（0.08s），应被去重/冷却压成 1 条
        ys = [500, 515, 530, 545, 560, 545, 530, 515, 500, 515, 530, 545, 560]
        _, events = run(proc, [obs(931, y, i) for i, y in enumerate(ys)])
        self.assertEqual(count(events, EventType.BOUNCE), 1)

    def test_separated_bounces_both_reported(self):
        """间隔足够远的两次落台应各报一条。"""
        proc = TrajectoryEventProcessor(CALIB)
        ys = [500, 520, 540, 560, 540, 520, 500]
        ys += [500] * 8                       # 拉开时间间隔（> 冷却 0.22s）
        ys += [500, 520, 540, 560, 540, 520, 500]
        _, events = run(proc, [obs(931, y, i) for i, y in enumerate(ys)])
        self.assertEqual(count(events, EventType.BOUNCE), 2)

    def test_event_ids_are_unique(self):
        proc = TrajectoryEventProcessor(CALIB)
        xs = [500, 560, 620, 680, 740, 800, 740, 680, 620, 560, 500]
        xs += [500, 560, 620, 680, 740, 800, 740, 680, 620, 560, 500]
        _, events = run(proc, [obs(x, 584, i) for i, x in enumerate(xs)])
        ids = [e.event_id for e in events]
        self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# 防尖刺与静止死锁
# ---------------------------------------------------------------------------
class TestGates(unittest.TestCase):
    def test_anti_spike_smooths_wild_outlier(self):
        """离群点（仍在双门限内）应被拉回首尾中点。"""
        proc = TrajectoryEventProcessor(CALIB)
        # 第 3 点被拽飞 40px：仍在预测邻域 45px 内会被接受，但偏离中点 > 28px
        xs = [600, 620, 640, 700, 680, 700, 720]
        run(proc, [obs(x, 584, i) for i, x in enumerate(xs)])
        self.assertGreaterEqual(proc.stats["smoothed"], 1)
        smoothed = [p for p in proc.trajectory if p.state == MotionState.SMOOTHED]
        self.assertTrue(smoothed)

    def test_far_outlier_rejected_before_reaching_anti_spike(self):
        """远超预测邻域的候选在双门限就被拒，根本进不了轨迹（第一道防线）。"""
        proc = TrajectoryEventProcessor(CALIB)
        xs = [600, 620, 640, 900, 680, 700, 720]
        run(proc, [obs(x, 584, i) for i, x in enumerate(xs)])
        self.assertEqual(proc.stats["smoothed"], 0)
        self.assertTrue(all(p.x < 800 for p in proc.trajectory))

    def test_static_deadlock_rejected_after_track(self):
        """已跟踪的球突然静止（吸附到反光物）应在超时后断轨。"""
        proc = TrajectoryEventProcessor(CALIB)
        frames = [obs(600 + 20 * i, 584, i) for i in range(6)]
        frames += [obs(700, 584, 6 + k) for k in range(10)]     # 突然静止
        run(proc, frames)
        self.assertGreaterEqual(proc.stats["rejected_static"], 1)


# ---------------------------------------------------------------------------
# 契约与生命周期
# ---------------------------------------------------------------------------
class TestProcessorContract(unittest.TestCase):
    def test_rejects_non_observation(self):
        proc = TrajectoryEventProcessor(CALIB)
        with self.assertRaises(TypeError):
            proc.update({"frame_id": 0})

    def test_rejects_backwards_timestamp(self):
        proc = TrajectoryEventProcessor(CALIB)
        run(proc, [obs(600, 584, 5)])
        with self.assertRaises(ValueError):
            proc.update(obs(620, 584, 2))

    def test_reset_clears_state(self):
        proc = TrajectoryEventProcessor(CALIB)
        run(proc, [obs(600 + 20 * i, 584, i) for i in range(6)])
        self.assertTrue(proc.is_tracking)
        proc.reset()
        self.assertFalse(proc.is_tracking)
        self.assertEqual(proc.trajectory, [])
        self.assertEqual(sum(proc.stats["events"].values()), 0)

    def test_accepts_candidates_from_out_of_band_channel(self):
        """带外候选通道应能驱动双门限：低分候选在预测邻域内被接受。"""
        proc = TrajectoryEventProcessor(CALIB)
        run(proc, [obs(500, 584, 0), obs(540, 584, 1), obs(580, 584, 2)])
        self.assertTrue(proc.is_tracking)
        before = proc.stats["measured"]
        # 主观测漏检，但带外候选给出一个低分近邻点（0.25 < 常规门限 0.35）
        observation = miss(3)
        low = BallObservation(visible=True, center_px=(620.0, 584.0), confidence=0.25)
        proc.update(observation, candidates=[low])
        self.assertGreater(proc.stats["measured"], before)

    def test_low_confidence_candidate_rejected_when_far_from_prediction(self):
        """远离预测点的低分候选不得被接受（双门限的另一半）。"""
        proc = TrajectoryEventProcessor(CALIB)
        run(proc, [obs(500, 584, 0), obs(540, 584, 1), obs(580, 584, 2)])
        before = proc.stats["measured"]
        far = BallObservation(visible=True, center_px=(1400.0, 584.0), confidence=0.25)
        proc.update(miss(3), candidates=[far])
        self.assertEqual(proc.stats["measured"], before)

    def test_describe_is_json_friendly(self):
        import json
        proc = TrajectoryEventProcessor(CALIB)
        run(proc, [obs(600 + 20 * i, 584, i) for i in range(6)])
        json.dumps(proc.describe())      # 不应抛错

    def test_calibration_is_rescaled_per_observation(self):
        proc = TrajectoryEventProcessor(CALIB)
        proc.update(obs(465, 292, 0, width=960, height=540))
        proc.update(obs(480, 292, 1, width=960, height=540))
        self.assertTrue(proc.is_tracking)


if __name__ == "__main__":
    unittest.main(verbosity=2)
