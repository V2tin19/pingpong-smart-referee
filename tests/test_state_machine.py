"""
tests/test_state_machine.py
===========================

计分状态机的单元测试。

用**合成事件序列**逐条验证判罚规则与四条门控，不依赖视频与模型::

    python tests/test_state_machine.py -v

门控（学长 2026-11-04 布置）：
    1. 发球准入：本方落台→对方落台才升为可判分；同侧再弹=准备动作重置；候选超时作废
    2. 发球前击球校验：发球前 1.5s 内必须有击球；无球拍接触直接挡下
    3. 回合内击球要求：回合内至少一次击球才判分；同侧两弹夹击球=序列误拼不计分
    4. 判分后冷却与默认不判分：判分后 2s 冷却忽略全部事件；静默超时默认只清理证据
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


def feed_with_ticks(sm: TableTennisStateMachine, events, tick_dt: float = 0.04, t_start: float = 0.0):
    """按生产管线的节奏喂事件：先把时钟推进到事件时刻（触发超时判据），再喂事件。

    ``run_referee`` / ``run_end_to_end`` 的主循环都是"先 ``tick(t)`` 再处理当帧事件"，
    本帮助函数复刻这一顺序，使单测能覆盖**带时钟推进**的长会话行为
    （``feed`` 不推进时钟，准入超时/回合终了/冷却期满等判据无从参与）。
    """
    out = []
    now = t_start
    for e in events:
        while now + tick_dt <= e.timestamp_s:
            now += tick_dt
            out.extend(sm.tick(now))
        out.extend(sm.update(e))
        now = e.timestamp_s
    return out


def serve_and_win_point_by(sm, t0: float, hitter: str = "A"):
    """构造一个"击球方得分"的完整回合：发球准入 → 接发交换 → 对方半台二跳。

    时序满足全部四条门控（真实比赛节奏）::

        hit(发球) → 己方落台 → 对方落台(升级,门控1) → 接发击球(门控3a)
        → 回球落发球方半台 → 发球方回击 → 落对方半台 → 同侧二跳(中间无击球)

    注意各 ``t0`` 之间要留出 **判分冷却**（``score_cooldown_s=2s``）+ 死球停顿，
    本文件 ``POINT_INTERVAL_S=3.0`` 已满足。
    """
    other = "B" if hitter == "A" else "A"
    return feed(
        sm,
        [
            ev("RACKET_HIT", hitter, t0),
            ev("BOUNCE", hitter, t0 + 0.10),       # 发球第一跳（己方半台）
            ev("BOUNCE", other, t0 + 0.20),        # 越网落对方半台 → 升级为可判分
            ev("RACKET_HIT", other, t0 + 0.30),    # 接发（回合内击球 ≥1）
            ev("BOUNCE", hitter, t0 + 0.40),       # 回球落发球方半台（正常交替）
            ev("RACKET_HIT", hitter, t0 + 0.50),   # 发球方回击
            ev("BOUNCE", other, t0 + 0.60),        # 落对方半台
            ev("BOUNCE", other, t0 + 0.75),        # 对方没接到，二跳（中间无击球）
        ],
    )


POINT_INTERVAL_S = 3.0     # 每分之间的死球停顿（> 判分冷却 2s）


# ---------------------------------------------------------------------------
# 门控 1：发球准入
# ---------------------------------------------------------------------------
class TestServeAdmission(unittest.TestCase):
    def test_serve_transitions_to_rally(self):
        """本方落台→对方落台 序列完成后才升为可判分对打。"""
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "A", 0.1)])
        self.assertEqual(sm.state, MatchState.SERVE_IN_PROGRESS)
        feed(sm, [ev("BOUNCE", "B", 0.2)])
        self.assertEqual(sm.state, MatchState.RALLY)

    def test_presize_bounces_are_preparation_not_fault(self):
        """**Test4 帧 828 抢跑误判的回归测试**：发球前颠球（同侧再弹）是准备动作。

        修复前：hit → 颠球第 1 跳（seen）→ 颠球第 2 跳 → DOUBLE_BOUNCE 直接送分 0:1。
        修复后：同侧再弹重置准入候选，绝不判罚；真实发球随后仍可正常完成准入。
        """
        sm = TableTennisStateMachine()
        out = feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 6.87),      # 颠球第 1 次落台
                ev("BOUNCE", "A", 6.90),      # 颠球第 2 次落台（Test4 在这里误判）
                ev("BOUNCE", "A", 7.36),      # 颠球第 3 次落台
            ],
        )
        self.assertEqual(out, [], "颠球是准备动作，不得产生任何判罚")
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertEqual(sm.state, MatchState.SERVE_IN_PROGRESS)
        # 第 1 次颠球见证"己方落台"，第 2 次同侧再弹触发重置
        self.assertGreaterEqual(sm.describe()["gates"]["serve_prep_resets"], 1)

    def test_preparation_then_real_serve_completes_admission(self):
        """颠球重置之后，真实发球仍能完成"本方→对方"准入。"""
        sm = TableTennisStateMachine()
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.10),      # 颠球
                ev("BOUNCE", "A", 0.30),      # 颠球（重置）
                ev("RACKET_HIT", "A", 6.90),  # 正式发球击球
                ev("BOUNCE", "A", 7.00),      # 发球第一跳（己方）
                ev("BOUNCE", "B", 7.20),      # 越网落对方半台
            ],
        )
        self.assertEqual(sm.state, MatchState.RALLY, "真实发球应完成准入")
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertEqual(sm._ctx.last_hitter, "A", "升级时击球归属重置为发球方")

    def test_direct_to_opponent_bounce_does_not_score(self):
        """序列不成立（未先落本方半台）→ 不判 SERVE_FAULT，候选等待/超时作废。"""
        sm = TableTennisStateMachine()
        out = feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "B", 0.1)])
        self.assertEqual(out, [], "保守策略：己方第一跳可能只是被漏检，不判发球失误")
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertEqual(sm.state, MatchState.SERVE_IN_PROGRESS)
        self.assertGreaterEqual(sm.describe()["gates"]["serve_admission_violations"], 1)

    def test_admission_candidate_timeout_voids(self):
        """门控 1：候选超时作废（死球，不判分）。"""
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0)])
        out = []
        for t in [0.5, 1.0, 1.3]:
            out.extend(sm.tick(t))
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertTrue(any(d.rule == "SERVE_INCOMPLETE" for d in out), "候选应超时作废")
        self.assertEqual(sm.state, MatchState.WAITING_SERVE)

    def test_serve_fault_and_serve_double_bounce_are_legacy_only(self):
        """SERVE_FAULT / 发球阶段二跳判分只在旧版模式（准入关闭）下存在。"""
        legacy = MatchRules(enable_serve_admission=False)
        sm = TableTennisStateMachine(legacy)
        out = feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "B", 0.1)])
        self.assertEqual(out[0].rule, "SERVE_FAULT")
        self.assertEqual(out[0].winner, "B")

        sm2 = TableTennisStateMachine(legacy)
        out2 = feed(
            sm2,
            [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "A", 0.1), ev("BOUNCE", "A", 0.2)],
        )
        self.assertEqual(out2[-1].rule, "DOUBLE_BOUNCE")
        self.assertEqual(out2[-1].winner, "B")

    def test_net_touch_during_serve_is_let(self):
        """发球擦网 → LET 重发，比分不变。"""
        sm = TableTennisStateMachine()
        out = feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("NET_TOUCH", "A", 0.05)])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, DecisionKind.LET)
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertEqual(sm.state, MatchState.WAITING_SERVE)


# ---------------------------------------------------------------------------
# 门控 2：发球前击球校验
# ---------------------------------------------------------------------------
class TestPreServeHitCheck(unittest.TestCase):
    def test_bounce_sequence_without_any_hit_never_promotes(self):
        """手抛传球/滚球（全程无击球）→ 永远不得升级、不得判分。"""
        sm = TableTennisStateMachine()
        out = feed(
            sm,
            [
                ev("BOUNCE", "A", 0.0),
                ev("BOUNCE", "B", 0.3),
                ev("BOUNCE", "B", 0.6),
                ev("BOUNCE", "A", 0.9),
            ],
        )
        self.assertEqual(out, [])
        self.assertNotEqual(sm.state, MatchState.RALLY, "无击球证据不得升级")
        self.assertEqual(sm.points, {"A": 0, "B": 0})

    def test_stale_hit_blocked_at_promotion(self):
        """击球证据过旧（>1.5s）的落台序列 → 挡下，不得升级。"""
        sm = TableTennisStateMachine()
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),      # 己方落台（seen）
                # 之后球消失，2 秒后才在对方半台出现落台 —— 击球证据已陈旧
                ev("BOUNCE", "B", 2.0),
            ],
        )
        self.assertNotEqual(sm.state, MatchState.RALLY, "击球证据超过窗口应挡下")
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertGreaterEqual(sm.describe()["gates"]["serve_no_hit_blocked"], 1)

    def test_real_serve_within_window_promotes(self):
        """真实发球（击球后 ~0.2s 完成序列）不受窗口影响。"""
        sm = TableTennisStateMachine()
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.10),
                ev("BOUNCE", "B", 0.30),
            ],
        )
        self.assertEqual(sm.state, MatchState.RALLY)


# ---------------------------------------------------------------------------
# 门控 3：回合内击球要求 / 序列误拼守卫
# ---------------------------------------------------------------------------
class TestRallyPhase(unittest.TestCase):
    def test_double_bounce_awards_last_hitter(self):
        """真实交换后的同侧二跳 → 另一侧得分。"""
        sm = TableTennisStateMachine()
        out = serve_and_win_point_by(sm, 0.0, hitter="A")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, DecisionKind.POINT)
        self.assertEqual(out[0].winner, "A")
        self.assertEqual(out[0].rule, "DOUBLE_BOUNCE")
        self.assertEqual(sm.points, {"A": 1, "B": 0})

    def test_serve_ace_double_bounce_blocked_without_rally_hit(self):
        """门控 3a：接发方从未触球（发球 ace 型二跳）→ 不判分，宁漏勿误。"""
        sm = TableTennisStateMachine()
        out = feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),      # 升级（回合内尚无击球）
                ev("BOUNCE", "B", 0.35),     # 对方没接到，二跳
            ],
        )
        self.assertEqual(out, [], "回合内无击球不得判分")
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertGreaterEqual(sm.describe()["gates"]["no_hit_skips"], 1)
        self.assertEqual(sm.state, MatchState.RALLY)

        # 关闭门控 3a → 恢复判分（消融对照）
        sm2 = TableTennisStateMachine(MatchRules(enable_rally_hit_required=False))
        out2 = feed(
            sm2,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),
                ev("BOUNCE", "B", 0.35),
            ],
        )
        self.assertEqual(out2[-1].rule, "DOUBLE_BOUNCE")
        self.assertEqual(out2[-1].winner, "A")

    def test_splice_guard_blocks_same_side_sandwich(self):
        """门控 3b：同侧两弹之间夹击球 = 序列误拼，不计分。

        典型形态（Test4 问题一）：真实落台 → 虚假击球（地板反光）→ 同侧再弹。
        修复前会被判成 OWN_COURT / DOUBLE_BOUNCE 送分。
        """
        sm = TableTennisStateMachine()
        out = feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),      # 升级（last_bounce_side=B）
                ev("RACKET_HIT", "B", 0.3),  # 虚假击球（B 侧）
                ev("BOUNCE", "B", 0.45),     # 同侧再弹 → 夹击球
            ],
        )
        self.assertEqual(out, [], "同侧两弹夹击球应视为序列误拼")
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertGreaterEqual(sm.describe()["gates"]["splice_skips"], 1)

    def test_own_court_scores_when_splice_guard_off(self):
        """关闭门控 3b（消融）→ 旧版 OWN_COURT 恢复判分。"""
        sm = TableTennisStateMachine(MatchRules(enable_splice_guard=False))
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),
            ],
        )
        out = feed(sm, [ev("RACKET_HIT", "B", 0.3), ev("BOUNCE", "B", 0.4)])
        self.assertEqual(out[-1].rule, "OWN_COURT")
        self.assertEqual(out[-1].winner, "A")

    def test_own_court_fires_on_non_sandwich_pattern(self):
        """门控 3b 只拦"同侧夹击球"；非同侧形态的 OWN_COURT 正常判分。"""
        sm = TableTennisStateMachine()
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),      # 升级，last_bounce_side=B
                ev("RACKET_HIT", "B", 0.3),  # B 接发
                ev("RACKET_HIT", "A", 0.4),  # （检测抖动造成的连续击球）
                ev("BOUNCE", "A", 0.5),      # 落在 A 自己半台（前弹在 B，非同侧夹）
            ],
        )
        out = sm.decisions
        self.assertTrue(any(d.rule == "OWN_COURT" and d.winner == "B" for d in out))

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

    def test_out_blocked_without_rally_hit(self):
        """门控 3a 同样约束 OUT：回合内无击球不判出界。"""
        sm = TableTennisStateMachine()
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),      # 升级，回合内无击球
            ],
        )
        out = feed(sm, [ev("OUT", "A", 0.3)])
        self.assertEqual(out, [], "回合内无击球不得判 OUT")
        self.assertGreaterEqual(sm.describe()["gates"]["no_hit_skips"], 1)

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

    def test_straight_exchange_keeps_scoreing(self):
        """多拍正常交换不触发任何门控，最终二跳正常判分。"""
        sm = TableTennisStateMachine()
        out = serve_and_win_point_by(sm, 0.0, hitter="B")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].winner, "B")
        self.assertEqual(sm.points, {"A": 0, "B": 1})


# ---------------------------------------------------------------------------
# 门控 4：判分后冷却与默认不判分
# ---------------------------------------------------------------------------
class TestScoringCooldown(unittest.TestCase):
    def test_events_within_cooldown_are_ignored(self):
        """判分后 2s 冷却：期间的全部事件（含"看起来能得分"的序列）一律忽略。"""
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")           # 判分于 ~0.75s，冷却至 ~2.75s
        self.assertEqual(sm.state, MatchState.CONFIRM_POINT)
        out = feed(
            sm,
            [
                ev("RACKET_HIT", "B", 1.0),
                ev("BOUNCE", "B", 1.1),
                ev("BOUNCE", "A", 1.2),                # 若不冷却会完成的准入序列
                ev("RACKET_HIT", "A", 1.3),
                ev("BOUNCE", "B", 1.4),
                ev("BOUNCE", "B", 1.5),                # 若不冷却会判的二跳
            ],
        )
        self.assertEqual(out, [], "冷却期内事件应全部忽略")
        self.assertEqual(sm.points, {"A": 1, "B": 0})
        self.assertEqual(sm.state, MatchState.CONFIRM_POINT)
        self.assertGreaterEqual(sm.describe()["gates"]["cooldown_ignored_events"], 6)

    def test_after_cooldown_scoring_resumes(self):
        """冷却期满后的第一个事件先落地回合重置，随后正常判分。"""
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")           # 冷却至 ~2.75s
        out = serve_and_win_point_by(sm, 3.0, "B")     # 冷却已过
        self.assertEqual([d.kind for d in out][-1], DecisionKind.POINT)
        self.assertEqual(sm.points, {"A": 1, "B": 1})

    def test_serve_rotation_happens_immediately_on_award(self):
        """换发在判分瞬间计算（比分已更新），冷却只推迟证据清理。"""
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        serve_and_win_point_by(sm, 3.0, "A")
        self.assertEqual(sm.server, "B", "两分后应换发（冷却期内即生效）")

    def test_zero_cooldown_restores_immediate_reset(self):
        sm = TableTennisStateMachine(MatchRules(score_cooldown_s=0.0))
        serve_and_win_point_by(sm, 0.0, "A")
        self.assertEqual(sm.state, MatchState.WAITING_SERVE, "冷却为 0 时立即重置")

    def test_cooldown_survives_tick(self):
        """冷却由 tick 落地：期满后 tick 把状态带回等待发球。"""
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        self.assertEqual(sm.state, MatchState.CONFIRM_POINT)
        sm.tick(2.0)
        self.assertEqual(sm.state, MatchState.CONFIRM_POINT, "冷却未满不得重置")
        sm.tick(2.8)
        self.assertEqual(sm.state, MatchState.WAITING_SERVE, "冷却期满应重置")


# ---------------------------------------------------------------------------
# 回合终了候选（组长 §4-2 + 门控 4 默认不判分）
# ---------------------------------------------------------------------------
class TestRallyEndCandidate(unittest.TestCase):
    def _to_rally_with_bounce_on_B(self, sm):
        """进入可判分对打并形成"球落 B 半台、A 为最后击球方"的局面（含回合内击球）。"""
        feed(
            sm,
            [
                ev("RACKET_HIT", "A", 0.0),
                ev("BOUNCE", "A", 0.1),
                ev("BOUNCE", "B", 0.2),      # 发球序列完成 → 升级
                ev("RACKET_HIT", "B", 0.3),  # 接发（回合内击球 ≥1）
                ev("BOUNCE", "A", 0.4),      # 正常交替
                ev("RACKET_HIT", "A", 0.5),  # A 回击
                ev("BOUNCE", "B", 0.6),      # 落 B 半台，B 未再触球
            ],
        )

    def test_ball_disappearance_alone_never_scores(self):
        """§4-2：**没有任何落台证据**时，球消失绝不能产生得分。"""
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0)])
        out = []
        for t in [1.0, 1.2, 1.5, 2.0, 2.5]:
            out.extend(sm.tick(t))
        self.assertEqual(sm.points, {"A": 0, "B": 0}, "球消失不得直接判分")
        self.assertTrue(all(d.kind != DecisionKind.POINT for d in out))
        self.assertGreaterEqual(sm.dead_ball_count, 1)

    def test_no_return_needs_rally_hit_and_positive_evidence(self):
        """`NO_RETURN` 需要正向落台证据，且受门控 3a（回合内击球）约束。"""
        # 有正向证据 + 回合内有击球 → 触发
        sm_yes = TableTennisStateMachine(MatchRules(enable_no_return=True))
        self._to_rally_with_bounce_on_B(sm_yes)
        for t in [1.5, 2.0]:
            sm_yes.tick(t)
        self.assertEqual(sm_yes.points, {"A": 1, "B": 0})

        # 无正向证据（只有一次击球）→ 不触发
        sm_no = TableTennisStateMachine(MatchRules(enable_no_return=True))
        feed(sm_no, [ev("RACKET_HIT", "A", 0.0)])
        for t in [1.5, 2.0]:
            sm_no.tick(t)
        self.assertEqual(sm_no.points, {"A": 0, "B": 0})

        # 有正向证据但回合内无击球（接发方从未触球）→ 门控 3a 拦下
        sm_nohit = TableTennisStateMachine(MatchRules(enable_no_return=True))
        feed(
            sm_nohit,
            [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "A", 0.1), ev("BOUNCE", "B", 0.2)],
        )
        for t in [1.5, 2.0]:
            sm_nohit.tick(t)
        self.assertEqual(sm_nohit.points, {"A": 0, "B": 0})

    def test_no_return_rule_scores_last_hitter(self):
        sm = TableTennisStateMachine(MatchRules(enable_no_return=True))
        self._to_rally_with_bounce_on_B(sm)
        out = []
        for t in [1.5, 2.0, 2.5]:
            out.extend(sm.tick(t))
        points = [d for d in out if d.kind == DecisionKind.POINT]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].winner, "A")
        self.assertEqual(points[0].rule, "NO_RETURN")

    def test_no_return_default_is_off(self):
        """门控 4：静默超时默认只清理证据、不判分（NO_RETURN 默认关闭）。"""
        sm = TableTennisStateMachine()             # 默认 rules
        self._to_rally_with_bounce_on_B(sm)
        out = []
        for t in [1.5, 2.0, 2.5]:
            out.extend(sm.tick(t))
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        self.assertTrue(all(d.kind == DecisionKind.DEAD_BALL for d in out))

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

    def test_quick_restart_enters_admission_and_never_faults(self):
        """**R-015 场景在门控语义下的回归**：每分结束后的快速再击球进入
        发球准入候选（不再直接进入可判分对打），因此"球落在对方半台"
        绝不会被误判成 SERVE_FAULT。"""
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        self.assertEqual(sm.points, {"A": 1, "B": 0})

        # 冷却期满后的快速击球 → 准入候选（非可判分对打）
        out = feed(sm, [ev("RACKET_HIT", "B", 3.0)])
        self.assertEqual(out, [])
        self.assertEqual(sm.state, MatchState.SERVE_IN_PROGRESS)

        # B 的"发球"直接落 A 半台（序列不成立）→ 不得判发球失误
        out = feed(sm, [ev("BOUNCE", "A", 3.1)])
        self.assertEqual(out, [], "不得误判为发球失误")
        self.assertEqual(sm.points, {"A": 1, "B": 0})

    def test_hit_after_dead_ball_pause_is_a_serve(self):
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        feed(sm, [ev("RACKET_HIT", "B", 5.0)])       # 停顿 + 冷却均已过
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

    处理原则：**已知信息不被未知信息覆盖，未知信息绝不参与判罚**。
    """

    def test_unknown_hit_never_establishes_serve(self):
        sm = TableTennisStateMachine()
        out = feed(sm, [ev("RACKET_HIT", "UNKNOWN", 0.0)])
        self.assertEqual(out, [])
        self.assertNotEqual(sm.state, MatchState.SERVE_IN_PROGRESS)
        self.assertEqual(sm.server, "A", "首发方不得被 UNKNOWN 篡改")

    def test_unknown_bounces_do_not_fault_or_score(self):
        """UNKNOWN 落台不判罚、不污染准入进度；随后的同侧再弹按准备动作重置。"""
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0), ev("BOUNCE", "A", 0.1)])
        out = feed(sm, [ev("BOUNCE", "UNKNOWN", 0.2)])
        self.assertEqual(out, [], "信息缺失的落台不得参与判罚")
        self.assertEqual(sm.points, {"A": 0, "B": 0})
        # 之后己方再弹 → 门控 1 重置（不是 DOUBLE_BOUNCE）
        out = feed(sm, [ev("BOUNCE", "A", 0.3)])
        self.assertEqual(out, [])
        self.assertGreaterEqual(sm.describe()["gates"]["serve_prep_resets"], 1)

    def test_unknown_hit_does_not_become_last_hitter(self):
        """UNKNOWN 击球不得成为击球方：随后同侧二跳绝不能把分加给 "UNKNOWN"。
        击球归属缺失期间，落台只记录不判罚。"""
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
# 入口幂等（参考《状态机修改方案》§2.5/§6：迟到/重复提交不重复加分）
# ---------------------------------------------------------------------------
class TestEntryIdempotency(unittest.TestCase):
    """同一个 event_id 只提交一次：重复投递静默忽略，绝不重复判分。

    携带旧时间戳的重复投递也必须被忽略（而非触发"时间倒退"异常）——
    这正是"按事件时间与事件 ID 幂等处理，而不是按推理完成时间"的要求。
    """

    def test_duplicate_event_id_is_ignored(self):
        sm = TableTennisStateMachine()
        e = ev("RACKET_HIT", "A", 0.0)
        feed(sm, [e])
        feed(sm, [e])                                # 同一事件重发
        feed(sm, [ev("BOUNCE", "A", 0.1)])           # 新事件正常处理
        # 击球只生效一次：准入候选建立一次，落台见证一次
        self.assertEqual(sm.state, MatchState.SERVE_IN_PROGRESS)
        self.assertGreaterEqual(
            sm.describe()["gates"]["duplicate_event_ids_ignored"], 1
        )

    def test_duplicate_bounce_does_not_double_score(self):
        """重复投递一条"能得分"的事件序列 → 不得双倍加分。"""
        sm = TableTennisStateMachine()
        seq = [
            ev("RACKET_HIT", "A", 0.0),
            ev("BOUNCE", "A", 0.1),
            ev("BOUNCE", "B", 0.2),
            ev("RACKET_HIT", "B", 0.3),
            ev("BOUNCE", "A", 0.4),
            ev("RACKET_HIT", "A", 0.5),
            ev("BOUNCE", "B", 0.6),
            ev("BOUNCE", "B", 0.75),
        ]
        first = feed(sm, seq)
        self.assertEqual(len([d for d in first if d.kind == DecisionKind.POINT]), 1)
        replay = feed(sm, seq)                       # 整段重放（同 id）
        self.assertEqual(replay, [], "整段重放不得再产生任何判罚")
        self.assertEqual(sm.points, {"A": 1, "B": 0})

    def test_duplicate_with_backwards_timestamp_is_ignored_not_raised(self):
        sm = TableTennisStateMachine()
        e = ev("RACKET_HIT", "A", 5.0)
        feed(sm, [e])
        late = RefereeEvent(
            event_id=e.event_id, timestamp_s=1.0, type="RACKET_HIT", side="A",
            sources=["yolo"],
        )
        out = sm.update(late)                        # 旧时间戳 + 同 id → 忽略而非抛错
        self.assertEqual(out, [])

    def test_id_reuse_after_reset_is_accepted(self):
        """处理器重放后事件计数器重启（id 会重复）——与状态机一并 reset 后必须正常。"""
        sm = TableTennisStateMachine()
        feed(sm, [ev("RACKET_HIT", "A", 0.0)])
        sm.reset()
        feed(sm, [ev("RACKET_HIT", "A", 0.0)])       # 同 id 重新出现（合法：已 reset）
        self.assertEqual(sm.state, MatchState.SERVE_IN_PROGRESS)

    def test_id_window_is_bounded(self):
        """有界去重窗口：超过窗口的旧 id 允许被遗忘（长比赛不内存膨胀）。"""
        sm = TableTennisStateMachine()
        for i in range(600):
            feed(sm, [ev("RACKET_HIT", "A", float(i))])
        self.assertLessEqual(len(sm._seen_event_ids), 512)


# ---------------------------------------------------------------------------
# 规则统计与健壮性
# ---------------------------------------------------------------------------
class TestDiagnostics(unittest.TestCase):
    def test_rule_counts_track_which_rule_scored(self):
        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        self.assertEqual(sm.rule_counts.get("DOUBLE_BOUNCE"), 1)
        # 旧版模式下 SERVE_FAULT 仍可触发
        legacy = TableTennisStateMachine(MatchRules(enable_serve_admission=False))
        feed(legacy, [ev("RACKET_HIT", "B", POINT_INTERVAL_S), ev("BOUNCE", "A", POINT_INTERVAL_S + 0.1)])
        self.assertEqual(legacy.rule_counts.get("SERVE_FAULT"), 1)

    def test_gate_stats_exposed_in_describe(self):
        """门控触发计数随 describe() 暴露，便于统计"挡掉了什么"。"""
        import json

        sm = TableTennisStateMachine()
        serve_and_win_point_by(sm, 0.0, "A")
        d = sm.describe()
        json.dumps(d)                              # 必须可 JSON 序列化
        self.assertIn("gates", d)
        for key in (
            "serve_prep_resets",
            "serve_admission_violations",
            "serve_no_hit_blocked",
            "splice_skips",
            "no_hit_skips",
            "cooldown_ignored_events",
        ):
            self.assertIn(key, d["gates"])

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
        self.assertEqual(sm.cooldown_remaining_s, 0.0)

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


# ---------------------------------------------------------------------------
# 生活流长会话（带时钟推进的集成回归）
# ---------------------------------------------------------------------------
class TestDailyLifeSession(unittest.TestCase):
    """**接手审计新增**：按真实管线节奏（先 tick 后事件）复验四条门控的协同。

    场景取自 Test4 报告与日常生活形态：选手拿球找手感（颠球）、手抛传球、
    正式发球、真实回合得分、判分后捡球、冷却结束后继续下一分。
    预期：**准备动作与杂散事件一律不产生判罚，全场只有真实回合判分**。
    """

    def test_full_session_scores_only_real_rally(self):
        sm = TableTennisStateMachine()
        out = feed_with_ticks(
            sm,
            [
                ev("RACKET_HIT", "A", 0.5),    # 拿拍颠球（候选开启）
                ev("BOUNCE", "A", 1.2),        # 颠球1（己方落台 seen）
                ev("BOUNCE", "A", 1.5),        # 颠球2 → 准备动作重置
                ev("BOUNCE", "A", 2.0),        # 颠球3（重新 seen）
                ev("BOUNCE", "A", 2.4),        # 颠球4 → 再次重置
                ev("BOUNCE", "A", 5.0),        # 手抛传球（此时候选已超时作废）
                ev("BOUNCE", "B", 5.3),
                ev("BOUNCE", "B", 5.6),
                ev("RACKET_HIT", "A", 15.6),   # 正式发球
                ev("BOUNCE", "A", 15.9),       # 本方落台
                ev("BOUNCE", "B", 16.3),       # 对方落台 → 升级（0.7s ≤ 1.5s）
                ev("RACKET_HIT", "B", 16.6),   # 接发（回合内击球 ≥1）
                ev("BOUNCE", "A", 17.0),       # 正常交替
                ev("RACKET_HIT", "A", 17.4),
                ev("BOUNCE", "B", 17.8),
                ev("BOUNCE", "B", 18.0),       # B 未接到 → A 得分（判分于 18.0s）
                ev("BOUNCE", "A", 19.0),       # 捡球杂散事件（冷却至 20.0s）
                ev("RACKET_HIT", "B", 19.4),
            ],
        )
        points = [d for d in out if d.kind == DecisionKind.POINT]
        self.assertEqual(sm.points, {"A": 1, "B": 0}, "全场只应判出真实的 1 分")
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].rule, "DOUBLE_BOUNCE")
        self.assertEqual(points[0].winner, "A")
        gates = sm.describe()["gates"]
        self.assertGreaterEqual(gates["serve_prep_resets"], 2, "颠球应被准备动作重置吸收")
        self.assertGreaterEqual(gates["cooldown_ignored_events"], 2, "捡球杂散事件应被冷却忽略")

        # 冷却期满后第二分（发球方仍为 A），随后换发
        out2 = feed_with_ticks(
            sm,
            [
                ev("RACKET_HIT", "A", 21.0),
                ev("BOUNCE", "A", 21.10),
                ev("BOUNCE", "B", 21.30),
                ev("RACKET_HIT", "B", 21.50),
                ev("BOUNCE", "A", 21.70),
                ev("RACKET_HIT", "A", 21.90),
                ev("BOUNCE", "B", 22.10),
                ev("BOUNCE", "B", 22.25),
            ],
            t_start=19.4,
        )
        self.assertEqual(sm.points, {"A": 2, "B": 0})
        self.assertEqual(sm.server, "B", "两分后应换发")
        self.assertEqual(sm.state, MatchState.CONFIRM_POINT)

    def test_event_at_exact_cooldown_boundary_is_processed(self):
        """冷却截止时刻（t == cooldown_until）的事件应被处理，而非忽略。"""
        sm = TableTennisStateMachine()
        out = serve_and_win_point_by(sm, 1.0, "A")
        award_t = [d for d in out if d.kind == DecisionKind.POINT][0].timestamp_s
        boundary = feed_with_ticks(
            sm, [ev("RACKET_HIT", "B", award_t + 2.0)], t_start=award_t
        )
        self.assertEqual(boundary, [], "边界事件本身是击球，不应产生判罚")
        self.assertEqual(
            sm.state, MatchState.SERVE_IN_PROGRESS,
            "冷却截止时刻的事件应开启新候选而非被忽略",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
