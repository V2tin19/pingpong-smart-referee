"""
tests/test_state_machine.py
===========================

计分状态机的单元测试。

用**合成事件序列**逐条验证判罚规则，不依赖视频与模型::

    python tests/test_state_machine.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import RefereeEvent
from core.state_machine import (
    DecisionKind,
    MatchRules,
    MatchState,
    TableTennisStateMachine,
)

_seq = {"n": 0}


def ev(etype: str, side: str = "UNKNOWN", t: float = 0.0) -> RefereeEvent:
    _seq["n"] += 1
    return RefereeEvent(
        event_id=f"{etype.lower()}_{_seq['n']:05d}",
        timestamp_s=t,
        type=etype,
        side=side,
        position_px=(900.0, 580.0),
        confidence=0.9,
        sources=["yolo"],
    )


def feed(sm: TableTennisStateMachine, events):
    out = []
    for e in events:
        out.extend(sm.update(e))
    return out


def serve_and_win_point_by(sm, t0: float, hitter: str = "A"):
    """构造一个"击球方得分"的完整回合：发球 → 对方半台 → 同侧二跳。

    注意 ``t0`` 之间要留出死球停顿（> ``serve_min_gap_s``），否则第一次击球
    不会被判为发球 —— 这是真实比赛的节奏。
    """
    other = "B" if hitter == "A" else "A"
    return feed(
        sm,
        [
            ev("RACKET_HIT", hitter, t0),
            ev("BOUNCE", hitter, t0 + 0.1),        # 发球第一跳（己方半台）
            ev("BOUNCE", other, t0 + 0.2),         # 越网落到对方半台
            ev("BOUNCE", other, t0 + 0.3),         # 对方没接到，二跳
        ],
    )


POINT_INTERVAL_S = 3.0     # 每分之间的死球停顿


# ---------------------------------------------------------------------------
# 发球阶段
# ---------------------------------------------------------------------------
class TestServePhase(unittest.TestCase):
    def test_serve_transitions_to_rally(self):
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "A", 0.1)])
        self.assertEqual(sm.state, MatchState.SERVE_IN_PROGRESS)
        feed(sm, [ev("BOUNCE", "B", 0.2)])
        self.assertEqual(sm.state, MatchState.RALLY)

    def test_serve_fault_when_not_bouncing_on_own_half(self):
        """发球没先落在自己半台 → 接发方得分。"""
        sm = TableTennisStateMachine()
        out = feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "B", 0.1)])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, DecisionKind.POINT)
        self.assertEqual(out[0].winner, "B")
        self.assertEqual(out[0].rule, "SERVE_FAULT")

    def test_serve_fault_rule_can_be_disabled(self):
        sm = TableTennisStateMachine(MatchRules(enable_serve_fault=False))
        out = feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "B", 0.1)])
        self.assertEqual(out, [])
        self.assertEqual(sm.points, {"A": 0, "B": 0})

    def test_net_touch_during_serve_is_let(self):
        """发球擦网 → LET 重发，比分不变。"""
        sm = TableTennisStateMachine()
        out = feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("NET_TOUCH", "A", 0.05)])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, DecisionKind.LET)
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertEqual(sm.state, MatchState.WAITING_SERVE)

    def test_serve_double_bounce_on_own_half(self):
        sm = TableTennisStateMachine()
        out = feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "A", 0.2),      # 己方半台二跳
            ],
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].winner, "B")
        self.assertEqual(out[0].rule, "DOUBLE_BOUNCE")


# ---------------------------------------------------------------------------
# 对打阶段
# ---------------------------------------------------------------------------
class TestRallyPhase(unittest.TestCase):
    def test_double_bounce_awards_last_hitter(self):
        sm = TableTennisStateMachine()
        out = serve_and_win_point_by(sm, 0.0, hitter="A")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, DecisionKind.POINT)
        self.assertEqual(out[0].winner, "A")
        self.assertEqual(out[0].rule, "DOUBLE_BOUNCE")
        self.assertEqual(sm.points, {"A": 1, "B": 0})

    def test_ball_into_own_court_is_a_fault(self):
        """击球后球落在自己半台 → 对手得分。"""
        sm = TableTennisStateMachine()
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),      # 发球完成，进入对打
            ],
        )
        self.assertEqual(sm.state, MatchState.RALLY)
        out = feed(sm, [ev("RACKET_HIT", "B", 0.3), ev("BOUNCE", "B", 0.4)])
        self.assertEqual(out[-1].rule, "OWN_COURT")
        self.assertEqual(out[-1].winner, "A")

    def test_out_awards_opponent_of_hitter(self):
        sm = TableTennisStateMachine()
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),
            ],
        )
        out = feed(sm, [ev("RACKET_HIT", "B", 0.3), ev("OUT", "A", 0.4)])
        self.assertEqual(out[-1].kind, DecisionKind.POINT)
        self.assertEqual(out[-1].winner, "A")
        self.assertEqual(out[-1].rule, "OUT")

    def test_out_rule_can_be_disabled(self):
        sm = TableTennisStateMachine(MatchRules(enable_out=False))
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),
                ev("RACKET_HIT", "B", 0.3),
            ],
        )
        out = feed(sm, [ev("OUT", "A", 0.4)])
        self.assertEqual(out, [])


# ---------------------------------------------------------------------------
# 回合终了候选（组长 §4-2）
# ---------------------------------------------------------------------------
class TestRallyEndCandidate(unittest.TestCase):
    def _to_rally_with_bounce_on_B(self, sm):
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),
                ev("RACKET_HIT", "B", 0.3),
                ev("BOUNCE", "A", 0.4),
                ev("RACKET_HIT", "A", 0.5),
                ev("BOUNCE", "B", 0.6),
            ],
        )

    def test_ball_disappearance_alone_never_scores(self):
        """§4-2：**没有任何落台证据**时，球消失绝不能产生得分。

        这里刻意只喂一个击球事件（没有任何落台），此时超时只能判"死球"。
        """
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0)])
        out = []
        for t in [1.0, 1.2, 1.5, 2.0, 2.5]:
            out.extend(sm.tick(t))
        self.assertEqual(sm.points, {"A": 0, "B": 0}, "球消失不得直接判分")
        self.assertTrue(all(d.kind != DecisionKind.POINT for d in out))
        self.assertGreaterEqual(sm.dead_ball_count, 1)

    def test_no_return_rule_needs_positive_bounce_evidence(self):
        """`NO_RETURN` 必须先有"球合法落在对方半台"的正向证据才会触发。

        这条同时**记录了一个已知风险**：若检测器在回合中途丢球（遮挡/漏检），
        而最后一次落台恰好落在对方半台，就会误判成"对方没接到"。
        这正是后续要加限制的地方（见 core/state_machine.py 的 NO_RETURN 说明）。
        """
        # 有正向证据 → 触发
        sm_yes = TableTennisStateMachine()
        feed(
            sm_yes,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),      # 合法落在对方半台
            ],
        )
        for t in [1.5, 2.0]:
            sm_yes.tick(t)
        self.assertEqual(sm_yes.points, {"A": 1, "B": 0})

        # 无正向证据 → 不触发
        sm_no = TableTennisStateMachine()
        feed(sm_no, [ev("RACKET_HIT", "A", 0.0)])
        for t in [1.5, 2.0]:
            sm_no.tick(t)
        self.assertEqual(sm_no.points, {"A": 0, "B": 0})

    def test_no_return_rule_scores_last_hitter(self):
        """有"球合法落在对方半台"这一正向证据时，消失才可判对方未接到。"""
        sm = TableTennisStateMachine()
        self._to_rally_with_bounce_on_B(sm)
        out = []
        for t in [1.5, 2.0, 2.5]:
            out.extend(sm.tick(t))
        points = [d for d in out if d.kind == DecisionKind.POINT]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].winner, "A")
        self.assertEqual(points[0].rule, "NO_RETURN")

    def test_no_return_rule_can_be_disabled(self):
        sm = TableTennisStateMachine(MatchRules(enable_no_return=False))
        self._to_rally_with_bounce_on_B(sm)
        out = []
        for t in [1.5, 2.0, 2.5]:
            out.extend(sm.tick(t))
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertGreaterEqual(sm.dead_ball_count, 1)

    def test_new_event_cancels_candidate(self):
        """候选期内球又出现 → 取消候选，回到对打。"""
        sm = TableTennisStateMachine()
        self._to_rally_with_bounce_on_B(sm)          # 最后一个事件在 t=0.6
        sm.tick(1.5)                                  # 0.9s 无事件 → 进入候选
        self.assertEqual(sm.state, MatchState.RALLY_END_CANDIDATE)
        feed(sm, [ev("RACKET_HIT", "B", 1.6)])
        self.assertEqual(sm.state, MatchState.RALLY)

    def test_timeouts_are_in_seconds_not_frames(self):
        """§4-1：超时以秒计。同样的时间间隔应产生同样的行为。"""
        results = []
        for _ in range(2):
            sm = TableTennisStateMachine(MatchRules(rally_end_s=0.8, confirm_s=0.4))
            self._to_rally_with_bounce_on_B(sm)
            out = []
            for t in [1.3, 1.5, 1.9]:
                out.extend(sm.tick(t))
            results.append([d.kind for d in out])
        self.assertEqual(results[0], results[1])


# ---------------------------------------------------------------------------
# 发球轮换与局分
# ---------------------------------------------------------------------------
class TestMatchFlow(unittest.TestCase):
    def test_serve_rotates_every_two_points(self):
        sm = TableTennisStateMachine()
        self.assertEqual(sm.server, "A")
        serve_and_win_point_by(sm, 0.0, "A")
        self.assertEqual(sm.server, "A")
        serve_and_win_point_by(sm, POINT_INTERVAL_S, "A")
        self.assertEqual(sm.server, "B", "两分后应换发")

    def test_hit_without_dead_ball_pause_is_not_a_serve(self):
        """**实测教训的回归测试**：没有死球停顿的击球不得被判为发球。

        否则正常对打中"球落在对方半台"会被 SERVE_FAULT 误判成发球失误
        （test_game_2 上曾误报 8 次，占全部 12 分）。
        """
        sm = TableTennisStateMachine()
        # 先打完一分
        serve_and_win_point_by(sm, 0.0, "A")
        self.assertEqual(sm.points, {"A": 1, "B": 0})

        # 紧接着（0.4s 后）又有一次击球 → 不应当作发球
        feed(sm, [ev("RACKET_HIT", "B", 0.4)])
        self.assertEqual(sm.state, MatchState.RALLY, "间隔不足时应视为对打延续")

        # 随后球落在 A 半台是正常的（B 打到 A 那边），不得判发球失误
        out = feed(sm, [ev("BOUNCE", "A", 0.5)])
        self.assertEqual(out, [], "不得误判为发球失误")
        self.assertEqual(sm.points, {"A": 1, "B": 0})

    def test_hit_after_dead_ball_pause_is_a_serve(self):
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        feed(sm, [ev("RACKET_HIT", "B", 5.0)])       # 停顿 4.7s 后
        self.assertEqual(sm.state, MatchState.SERVE_IN_PROGRESS)
        self.assertEqual(sm.server, "B")

    def test_game_over_at_11_with_two_point_lead(self):
        sm = TableTennisStateMachine()
        out = []
        for i in range(11):
            out.extend(serve_and_win_point_by(sm, i * POINT_INTERVAL_S, "A"))
        self.assertTrue(sm.game_over)
        self.assertTrue(any(d.kind == DecisionKind.GAME_OVER for d in out))
        self.assertEqual(sm.points, {"A": 11, "B": 0})

    def test_no_game_over_at_11_to_10(self):
        """11:10 不算赢，需净胜 2 分。"""
        sm = TableTennisStateMachine()
        for i in range(10):
            serve_and_win_point_by(sm, i * 2 * POINT_INTERVAL_S, "A")
            serve_and_win_point_by(sm, i * 2 * POINT_INTERVAL_S + POINT_INTERVAL_S, "B")
        serve_and_win_point_by(sm, 100.0, "A")
        self.assertEqual(sm.points, {"A": 11, "B": 10})
        self.assertFalse(sm.game_over)

    def test_no_events_after_game_over(self):
        sm = TableTennisStateMachine()
        for i in range(11):
            out = serve_and_win_point_by(sm, i * POINT_INTERVAL_S, "A")
        out = serve_and_win_point_by(sm, 200.0, "A")
        self.assertEqual(out, [])


# ---------------------------------------------------------------------------
# side=UNKNOWN 的处理（§7.4：信息缺失）
# ---------------------------------------------------------------------------
class TestUnknownSide(unittest.TestCase):
    """``RefereeEvent`` 的文档承诺"状态机会把 UNKNOWN 当作信息缺失处理"，
    这里把承诺锁进测试。

    修复前的实际行为（实测复现）：UNKNOWN 击球会静默建立 ``server="UNKNOWN"``、
    成为 ``last_hitter``，随后的落台会凭空给某一侧加分，甚至在
    DOUBLE_BOUNCE 归属时触发 ``points["UNKNOWN"]`` KeyError 崩溃。
    对会输出 UNKNOWN 的事件层（TTNet / TrackNet 分支允许填 UNKNOWN），
    这是静默判错分的隐患。

    处理原则：**已知信息不被未知信息覆盖，未知信息绝不参与判罚**。
    """

    def test_unknown_hit_never_establishes_serve(self):
        sm = TableTennisStateMachine()
        out = feed(sm, [ev("RACKET_HIT", "UNKNOWN", 0.0)])
        self.assertEqual(out, [])
        self.assertNotEqual(sm.state, MatchState.SERVE_IN_PROGRESS)
        self.assertEqual(sm.server, "A", "首发方不得被 UNKNOWN 篡改")

    def test_unknown_bounces_do_not_fault_or_score(self):
        """发球己方一跳后混入一次未知侧落台 → 不得误判 SERVE_FAULT。"""
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "A", 0.1)])
        out = feed(sm, [ev("BOUNCE", "UNKNOWN", 0.2)])
        self.assertEqual(out, [], "信息缺失的落台不得参与判罚")
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        # 之后己方二跳仍应正常判 DOUBLE_BOUNCE（UNKNOWN 不得污染状态）
        out = feed(sm, [ev("BOUNCE", "A", 0.3)])
        self.assertEqual(out[-1].rule, "DOUBLE_BOUNCE")
        self.assertEqual(out[-1].winner, "B")

    def test_unknown_hit_does_not_become_last_hitter(self):
        """UNKNOWN 击球不得成为击球方：随后同侧二跳绝不能把分加给 "UNKNOWN"
        （修复前会 KeyError 崩溃）。击球归属缺失期间，落台只记录不判罚。"""
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "A", 0.1), ev("BOUNCE", "B", 0.2)])
        feed(sm, [ev("RACKET_HIT", "UNKNOWN", 0.3)])      # 对打中混入未知方击球
        out = feed(sm, [ev("BOUNCE", "A", 0.4), ev("BOUNCE", "A", 0.5)])
        self.assertEqual(out, [], "击球归属缺失时，落台只记录不判罚")
        self.assertEqual(sm.points, {"A": 0, "B": 0})

    def test_known_hit_after_unknown_reestablishes_attribution(self):
        """已知方击球会重新建立归属，回合恢复正常判罚。"""
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "A", 0.1), ev("BOUNCE", "B", 0.2)])
        feed(sm, [ev("RACKET_HIT", "UNKNOWN", 0.3)])
        feed(sm, [ev("BOUNCE", "A", 0.4)])                # 只记录
        out = feed(sm, [ev("RACKET_HIT", "B", 0.5), ev("BOUNCE", "B", 0.6)])
        self.assertEqual(out[-1].rule, "OWN_COURT")
        self.assertEqual(out[-1].winner, "A")


# ---------------------------------------------------------------------------
# 规则统计与健壮性
# ---------------------------------------------------------------------------
class TestDiagnostics(unittest.TestCase):
    def test_rule_counts_track_which_rule_scored(self):
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        # 死球停顿后换 B 发球，发球直接落到 A 半台 → 发球失误
        feed(sm, [ev("RACKET_HIT", "B", POINT_INTERVAL_S), ev("BOUNCE", "A", POINT_INTERVAL_S + 0.1)])
        self.assertEqual(sm.rule_counts.get("DOUBLE_BOUNCE"), 1)
        self.assertEqual(sm.rule_counts.get("SERVE_FAULT"), 1)

    def test_describe_is_json_friendly(self):
        import json

        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        json.dumps(sm.describe())

    def test_backwards_timestamp_rejected(self):
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 5.0)])
        with self.assertRaises(ValueError):
            sm.update(ev("BOUNCE", "A", 1.0))

    def test_reset_clears_everything(self):
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        sm.reset()
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertEqual(sm.state, MatchState.WAITING_SERVE)
        self.assertEqual(sm.decisions, [])

    def test_rejects_non_event(self):
        sm = TableTennisStateMachine()
        with self.assertRaises(TypeError):
            sm.update({"type": "BOUNCE"})

    def test_decision_describe_is_readable(self):
        sm = TableTennisStateMachine()
        out = serve_and_win_point_by(sm, 0.0, "A")
        text = out[0].describe()
        self.assertIn("得分", text)
        self.assertIn("1:0", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
