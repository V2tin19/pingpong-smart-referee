"""
tests/test_evidence_adjudicator.py
==================================

回合级证据判决引擎单元测试集：
1. 颠球试手感在 PREP 阶段不产生任何判分；
2. 接发失误 (发球后 B 侧连续二跳) 正确判定 A 得分 1:0，不被当作死球吞掉；
3. 发球擦网重发 (LET) 准确触发且不改变比分；
4. 击球出界 (OUT) 正确判定对方得分；
5. 语音纠错撤销与改判接口测试。
"""

from __future__ import annotations

import unittest
from core.geometry import preset
from core.schemas import EventType, RefereeEvent, Side
from core.evidence_adjudicator import (
    AdjudicationOutcome,
    EvidenceStateMachine,
    HypothesisJudge,
    RallyBuffer,
    RallyPhase,
)


class TestEvidenceAdjudicator(unittest.TestCase):
    def setUp(self):
        self.calib = preset("game2")
        self.sm = EvidenceStateMachine(self.calib)

    def _event(
        self, ev_type: str, side: str, t: float,
        pos_px: tuple = (700.0, 500.0), conf: float = 0.9,
        d_edge: float = 10.0,
    ) -> RefereeEvent:
        return RefereeEvent(
            event_id=f"{ev_type.lower()}_{int(t * 1000)}",
            timestamp_s=t,
            type=ev_type,
            side=side,
            position_px=pos_px,
            confidence=conf,
            distance_to_edge_cm=d_edge,
            sources=["yolo"],
        )

    def test_prep_bounces_and_dribbles_never_score(self):
        """测试 1：发球准备期拍球/颠球找手感，绝不产生判分。"""
        # 选手在 A 侧连续颠球 5 次
        times = [1.0, 1.3, 1.6, 1.9, 2.2]
        for t in times:
            decs = self.sm.update(self._event("BOUNCE", "A", t, pos_px=(600.0, 550.0)))
            self.assertEqual(len(decs), 0)
            t_decs = self.sm.tick(t + 0.1)
            self.assertEqual(len(t_decs), 0)

        # 选手停顿 2 秒放弃发球
        t_after = self.sm.tick(5.0)
        self.assertEqual(len(t_after), 0)
        self.assertEqual(self.sm.points["A"], 0)
        self.assertEqual(self.sm.points["B"], 0)
        self.assertEqual(self.sm.phase, RallyPhase.PREP)

    def test_serve_receiver_miss_double_bounce_scores(self):
        """测试 2：发球合规，接发方未触球且在 B 侧连跳两次，准确判 P1(A) 得 1:0。"""
        # 1. 发球方 A 击球与己方落台
        self.sm.update(self._event("RACKET_HIT", "A", 1.0, pos_px=(500.0, 520.0)))
        self.sm.update(self._event("BOUNCE", "A", 1.2, pos_px=(700.0, 540.0)))

        # 2. 球越网 (net_x=929，球从 700 飞到 1100) 并落在 B 台
        self.sm.update(self._event("BOUNCE", "B", 1.6, pos_px=(1100.0, 540.0)))
        self.assertEqual(self.sm.phase, RallyPhase.RALLY)

        # 3. 接发方 B 漏接，球在 B 台连跳第 2 次
        decs = self.sm.update(self._event("BOUNCE", "B", 1.9, pos_px=(1250.0, 560.0)))

        self.assertEqual(len(decs), 1)
        dec = decs[0]
        self.assertEqual(dec.outcome, AdjudicationOutcome.POINT_A)
        self.assertEqual(dec.winner, "A")
        self.assertEqual(self.sm.points["A"], 1)
        self.assertEqual(self.sm.points["B"], 0)
        self.assertEqual(dec.tier, "HIGH_CONFIDENCE")
        self.assertGreater(dec.conf, 0.35)

    def test_serve_let_net_touch_handled(self):
        """测试 3：发球擦网后成功落在接发方 B 台，判定擦网重发 (LET)，比分不增加。"""
        # A 发球落 A 台
        self.sm.update(self._event("BOUNCE", "A", 1.0, pos_px=(700.0, 540.0)))
        # 触网 (net_x=929)
        self.sm.update(self._event("NET_TOUCH", "A", 1.25, pos_px=(930.0, 500.0)))
        # 越网落在 B 台
        self.sm.update(self._event("BOUNCE", "B", 1.5, pos_px=(1100.0, 540.0)))
        # 超时结束对打
        decs = self.sm.tick(2.5)

        self.assertEqual(len(decs), 1)
        dec = decs[0]
        self.assertEqual(dec.outcome, AdjudicationOutcome.LET)
        self.assertIsNone(dec.winner)
        self.assertEqual(self.sm.points["A"], 0)
        self.assertEqual(self.sm.points["B"], 0)

    def test_out_event_awards_opponent(self):
        """测试 4：对攻中 B 方击球后直接出界，判定 A 得分。"""
        # A 发球落 A 台后越网落 B 台
        self.sm.update(self._event("BOUNCE", "A", 1.0, pos_px=(700.0, 540.0)))
        self.sm.update(self._event("BOUNCE", "B", 1.4, pos_px=(1100.0, 540.0)))
        # B 方回击
        self.sm.update(self._event("RACKET_HIT", "B", 1.6, pos_px=(1300.0, 530.0)))
        # 球飞出台面出界 (OUT)
        decs = self.sm.update(self._event("OUT", "A", 1.9, pos_px=(400.0, 750.0), d_edge=-15.0))

        self.assertEqual(len(decs), 1)
        dec = decs[0]
        self.assertEqual(dec.outcome, AdjudicationOutcome.POINT_A)
        self.assertEqual(dec.winner, "A")
        self.assertEqual(self.sm.points["A"], 1)

    def test_voice_revoke_and_change_decision(self):
        """测试 5：预留语音纠错接口 (撤销重算与强行改判)。"""
        # 先制造一次 P1 得分 (1:0)
        self.sm.update(self._event("BOUNCE", "A", 1.0, pos_px=(700.0, 540.0)))
        self.sm.update(self._event("BOUNCE", "B", 1.4, pos_px=(1100.0, 540.0)))
        self.sm.update(self._event("BOUNCE", "B", 1.8, pos_px=(1200.0, 540.0)))
        self.assertEqual(self.sm.points["A"], 1)

        # 玩家说“不对”/“撤销”
        revoked = self.sm.revoke_last_point()
        self.assertTrue(revoked)
        self.assertEqual(self.sm.points["A"], 0)
        self.assertEqual(self.sm.points["B"], 0)

        # 制造第二次得分
        self.sm.update(self._event("BOUNCE", "A", 3.0, pos_px=(700.0, 540.0)))
        self.sm.update(self._event("BOUNCE", "B", 3.4, pos_px=(1100.0, 540.0)))
        self.sm.update(self._event("BOUNCE", "B", 3.8, pos_px=(1200.0, 540.0)))
        self.assertEqual(self.sm.points["A"], 1)

        # 玩家说“B得分” (改判)
        changed = self.sm.change_last_decision("B")
        self.assertTrue(changed)
        self.assertEqual(self.sm.points["A"], 0)
        self.assertEqual(self.sm.points["B"], 1)

    def _admit_serve(self, sm=None):
        sm = sm or self.sm
        sm.update(self._event("RACKET_HIT", "A", 0.8, pos_px=(500.0, 520.0)))
        sm.update(self._event("BOUNCE", "A", 1.0, pos_px=(700.0, 540.0)))
        sm.update(self._event("BOUNCE", "B", 1.3, pos_px=(1100.0, 540.0)))
        self.assertEqual(sm.phase, RallyPhase.RALLY)

    def test_floor_out_after_legal_serve_bounce_awards_server(self):
        """台内未回击后掉到地板的 OUT 不能反判成发球出界。"""
        self._admit_serve()
        decision = self.sm.update(self._event(
            "OUT", "B", 1.6, pos_px=(1300.0, 750.0), d_edge=-15.0,
        ))[0]
        self.assertEqual(decision.outcome, AdjudicationOutcome.POINT_A)
        self.assertIn("OUT_AFTER_LEGAL_BOUNCE", [r.rule_name for r in decision.matched_rules])

    def test_floor_out_after_legal_return_awards_last_hitter(self):
        self._admit_serve()
        self.sm.update(self._event("RACKET_HIT", "B", 1.5, pos_px=(1300.0, 520.0)))
        self.sm.update(self._event("BOUNCE", "A", 1.7, pos_px=(700.0, 540.0)))
        decision = self.sm.update(self._event(
            "OUT", "A", 2.0, pos_px=(500.0, 750.0), d_edge=-15.0,
        ))[0]
        self.assertEqual(decision.outcome, AdjudicationOutcome.POINT_B)

    def test_unknown_hit_and_bounce_do_not_score_own_court(self):
        self._admit_serve()
        self.sm.update(self._event("RACKET_HIT", "UNKNOWN", 1.5, pos_px=(1100.0, 530.0)))
        self.sm.update(self._event("BOUNCE", "UNKNOWN", 1.7, pos_px=(1100.0, 540.0)))
        decision = self.sm.tick(2.6)[0]
        self.assertEqual(decision.outcome, AdjudicationOutcome.INVALID)
        self.assertNotIn("OWN_COURT_UNKNOWN", [r.rule_name for r in decision.matched_rules])
        self.assertEqual(self.sm.points, {"A": 0, "B": 0})

    def test_game_over_ignores_more_events_until_revoke(self):
        sm = EvidenceStateMachine(self.calib, points_to_win=1, win_by=1)
        self._admit_serve(sm)
        sm.update(self._event("BOUNCE", "B", 1.6, pos_px=(1200.0, 540.0)))
        self.assertTrue(sm.game_over)
        for kind, side, t, x in [
            ("BOUNCE", "A", 3.0, 700.0),
            ("BOUNCE", "B", 3.3, 1100.0),
            ("BOUNCE", "B", 3.6, 1200.0),
        ]:
            self.assertEqual(sm.update(self._event(kind, side, t, pos_px=(x, 540.0))), [])
        self.assertEqual(sm.tick(5.0), [])
        self.assertEqual(sm.points, {"A": 1, "B": 0})
        self.assertEqual(len(sm._decisions_history), 1)
        self.assertTrue(sm.revoke_last_point())
        self.assertFalse(sm.game_over)
        self._admit_serve(sm)
        self.assertEqual(sm.phase, RallyPhase.RALLY)

    def test_expired_prep_sample_cannot_create_a_net_crossing(self):
        for use_tick in (False, True):
            with self.subTest(use_tick=use_tick):
                sm = EvidenceStateMachine(self.calib)
                sm.update(self._event("BOUNCE", "A", 1.0, pos_px=(700.0, 540.0)))
                if use_tick:
                    sm.tick(10.0)
                sm.update(self._event("BOUNCE", "B", 11.0, pos_px=(1100.0, 540.0)))
                self.assertEqual(sm.phase, RallyPhase.PREP)
                self.assertEqual(sm.tick(12.0), [])
                self.assertEqual(sm.points, {"A": 0, "B": 0})

    def test_zero_confidence_double_bounce_is_not_promoted(self):
        self._admit_serve()
        decision = self.sm.update(self._event(
            "BOUNCE", "B", 1.6, pos_px=(1200.0, 540.0), conf=0.0,
        ))[0]
        self.assertEqual(decision.outcome, AdjudicationOutcome.INVALID)
        self.assertEqual(decision.matched_rules[0].min_event_conf, 0.0)
        self.assertEqual(self.sm.points, {"A": 0, "B": 0})

    def test_zero_confidence_stays_zero_for_each_judge_rule(self):
        cases = [
            ("UNRETURNED_IN_B", [self._event("BOUNCE", "B", 1.0, conf=0.0)]),
            ("SERVE_OUT_A", [self._event("OUT", "B", 1.0, conf=0.0)]),
            ("SERVE_OFF_TABLE_A", [self._event("BOUNCE", "B", 1.0, conf=0.0, d_edge=-15.0)]),
            ("OWN_COURT_B", [self._event("RACKET_HIT", "B", 1.0, conf=0.0),
                             self._event("BOUNCE", "B", 1.3)]),
            ("SERVE_NET_LET", [self._event("NET_TOUCH", "A", 1.0, conf=0.0),
                              self._event("BOUNCE", "B", 1.3)]),
        ]
        for rule_name, events in cases:
            with self.subTest(rule=rule_name):
                buffer = RallyBuffer(round_id=1, server="A")
                for event in events:
                    buffer.append(event)
                decision = HypothesisJudge().adjudicate(buffer, (0, 0))
                rule = next(r for r in decision.matched_rules if r.rule_name == rule_name)
                self.assertEqual(rule.min_event_conf, 0.0)
                self.assertEqual(rule.score, 0.0)


if __name__ == "__main__":
    unittest.main()
