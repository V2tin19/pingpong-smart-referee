"""
core/evidence_adjudicator.py
============================

回合级证据判决引擎 (Evidence-Based Rally Adjudication Engine)

基于《乒乓球视觉裁判：回合级证据判决设计方案》（Nov 9, 2026 · @Peter Lee）实现：
1. 先收证据，后下判决：回合内只收集带物理置信度的事件，回合结束后统一竞争打分；
2. 准备期不参与判决：颠球、垫球、试手感只在滑动窗口内自然过期，等球第一次越网再回头看；
3. 缺失击球不是否决条件：接发失误与发球 Ace 靠落台与消失证据独立判分，绝不被当死球吞掉；
4. 支持擦网重发 (LET) 候选分支与出界保底 (OUT)；
5. 两档置信度 (HIGH / LOW) 动作，预留后续语音纠错对接接口 (撤销、改判、低置信播报)。
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.geometry import TableCalibration
from core.homography import TableHomography
from core.schemas import EventType, RefereeEvent, Side


class AdjudicationOutcome(str, Enum):
    """回合候选结局。"""
    POINT_A = "POINT_A"      # A 得分
    POINT_B = "POINT_B"      # B 得分
    LET = "LET"              # 擦网重发 (双方均不计分，发球方重发)
    INVALID = "INVALID"      # 无效局 / 证据不足死球 (不计分)


class RallyPhase(str, Enum):
    """回合三阶段状态机。"""
    PREP = "PREP"            # 发球准备期 (颠球/热身/找手感，不判分)
    RALLY = "RALLY"          # 回合对打中 (缓冲收集带置信度证据)
    CLOSED = "CLOSED"        # 回合结束待判决 (竞争打分，输出决策)


@dataclass
class EvidenceRuleMatch:
    """一条匹配上的证据规则及其对候选结局的支持分。"""
    rule_name: str
    target: AdjudicationOutcome
    weight: float
    min_event_conf: float
    score: float
    description: str


@dataclass
class RallyDecision:
    """回合最终判决结果。"""
    round_id: int
    timestamp_s: float
    outcome: AdjudicationOutcome
    winner: Optional[str]              # "A" / "B" / None
    conf: float                        # 相对置信度 (S1 - S2) / (S1 + S2 + eps)
    tier: str                          # "HIGH_CONFIDENCE" / "LOW_CONFIDENCE"
    score_before: Tuple[int, int]
    score_after: Tuple[int, int]
    server: str                        # 本回合发球方 ("A" / "B")
    scores_by_outcome: Dict[str, float]
    matched_rules: List[EvidenceRuleMatch]
    events_in_rally: int
    explanation: str

    def describe(self) -> str:
        tag = "【高置信】" if self.tier == "HIGH_CONFIDENCE" else "【低置信-待确认】"
        if self.outcome == AdjudicationOutcome.POINT_A:
            res = f"P1(左) 得分  {self.score_after[0]}:{self.score_after[1]}"
        elif self.outcome == AdjudicationOutcome.POINT_B:
            res = f"P2(右) 得分  {self.score_after[0]}:{self.score_after[1]}"
        elif self.outcome == AdjudicationOutcome.LET:
            res = f"擦网重发 [LET] (比分保持 {self.score_after[0]}:{self.score_after[1]})"
        else:
            res = f"无效球 [INVALID] (比分保持 {self.score_after[0]}:{self.score_after[1]})"
        return f"{tag} {res} | conf={self.conf:.2f} ({self.explanation})"


@dataclass
class RallyBuffer:
    """单回合证据缓冲区。"""
    round_id: int
    server: str
    serve_compliance: float = 1.0
    start_time: float = 0.0
    end_time: float = 0.0
    events: List[RefereeEvent] = field(default_factory=list)

    def append(self, ev: RefereeEvent) -> None:
        self.events.append(ev)
        self.end_time = max(self.end_time, ev.timestamp_s)

    @property
    def bounces(self) -> List[RefereeEvent]:
        return [e for e in self.events if e.type == EventType.BOUNCE.value]

    @property
    def hits(self) -> List[RefereeEvent]:
        return [e for e in self.events if e.type == EventType.RACKET_HIT.value]

    @property
    def net_touches(self) -> List[RefereeEvent]:
        return [e for e in self.events if e.type == EventType.NET_TOUCH.value]

    @property
    def outs(self) -> List[RefereeEvent]:
        return [e for e in self.events if e.type == EventType.OUT.value]

    def consecutive_double_bounce(self) -> Optional[Tuple[str, float, RefereeEvent, RefereeEvent]]:
        """检测同半台连续二次落台（期间无该侧击球）。

        Returns
        -------
        Optional[Tuple[side, min_conf, first_bounce, second_bounce]]
        """
        for i in range(len(self.bounces) - 1):
            b1 = self.bounces[i]
            b2 = self.bounces[i + 1]
            if b1.side in ("A", "B") and b1.side == b2.side:
                # 检查两弹之间是否有该侧击球
                intervening_hit = any(
                    h.side == b1.side and b1.timestamp_s <= h.timestamp_s <= b2.timestamp_s
                    for h in self.hits
                )
                if not intervening_hit:
                    min_c = min(
                        b1.confidence if b1.confidence is not None else 0.8,
                        b2.confidence if b2.confidence is not None else 0.8,
                    )
                    return (b1.side, min_c, b1, b2)
        return None


class HypothesisJudge:
    """假设竞争判决引擎。

    给四个候选结局打分：POINT_A, POINT_B, LET, INVALID。
    最高分胜出，并根据第一名与第二名的差比计算置信度。
    """

    def __init__(
        self,
        theta_hi: float = 0.35,
        min_score_hi: float = 1.6,
        base_invalid_score: float = 0.8,
        out_margin_cm: float = 6.0,
    ) -> None:
        self.theta_hi = float(theta_hi)
        self.min_score_hi = float(min_score_hi)
        self.base_invalid_score = float(base_invalid_score)
        self.out_margin_cm = float(out_margin_cm)

    def adjudicate(self, buffer: RallyBuffer, score_before: Tuple[int, int]) -> RallyDecision:
        scores: Dict[AdjudicationOutcome, float] = {
            AdjudicationOutcome.POINT_A: 0.0,
            AdjudicationOutcome.POINT_B: 0.0,
            AdjudicationOutcome.LET: 0.0,
            AdjudicationOutcome.INVALID: self.base_invalid_score,
        }
        matches: List[EvidenceRuleMatch] = []

        def add_rule(
            name: str, target: AdjudicationOutcome, weight: float,
            min_c: float, desc: str,
        ) -> None:
            s = weight * min_c
            scores[target] += s
            matches.append(EvidenceRuleMatch(name, target, weight, min_c, s, desc))

        # ---- 规则 1：发球擦网重发 (LET - 用户确认 3) -------------------------
        # 发球阶段有触网，且在接发方击球前，球落入了接发方半台
        receiver = "B" if buffer.server == "A" else "A"
        server = buffer.server
        receiver_hits = [h for h in buffer.hits if h.side == receiver]
        first_receiver_hit_time = receiver_hits[0].timestamp_s if receiver_hits else float("inf")
        serve_net_touches = [n for n in buffer.net_touches if n.timestamp_s < first_receiver_hit_time]

        serve_net_let_matched = False
        if serve_net_touches:
            net_ev = serve_net_touches[0]
            # 触网之后是否有落在接发方半台
            subsequent_bounce = [
                b for b in buffer.bounces
                if b.timestamp_s >= net_ev.timestamp_s and b.timestamp_s < first_receiver_hit_time
            ]
            if subsequent_bounce and subsequent_bounce[0].side == receiver:
                serve_net_let_matched = True
                c_net = min(
                    net_ev.confidence if net_ev.confidence is not None else 0.8,
                    subsequent_bounce[0].confidence if subsequent_bounce[0].confidence is not None else 0.8,
                )
                add_rule(
                    "SERVE_NET_LET", AdjudicationOutcome.LET, 3.5, c_net,
                    f"发球擦网后成功落在接发方({receiver})半台，判定擦网重发",
                )

        # ---- 规则 2：某半台连续二次落台 (DOUBLE_BOUNCE) -----------------------
        db = buffer.consecutive_double_bounce() if not serve_net_let_matched else None
        if db:
            side, min_c, b1, b2 = db
            if side == "B":
                # B 侧连跳两次，B 方失误，支持 A 得分
                add_rule(
                    "DOUBLE_BOUNCE_B", AdjudicationOutcome.POINT_A, 3.0, min_c,
                    f"球在 B 侧连跳两次且 B 方未接到 (conf={min_c:.2f})",
                )
            elif side == "A":
                # A 侧连跳两次，A 方失误，支持 B 得分
                add_rule(
                    "DOUBLE_BOUNCE_A", AdjudicationOutcome.POINT_B, 3.0, min_c,
                    f"球在 A 侧连跳两次且 A 方未接到 (conf={min_c:.2f})",
                )

        # ---- 规则 3：落入对方半台后无人回击/球消失 (UNRETURNED_BALL) ---------
        if buffer.bounces and not serve_net_let_matched:
            last_b = buffer.bounces[-1]
            last_h = buffer.hits[-1] if buffer.hits else None
            # 如果最后一次有效落台在台内
            d_edge = last_b.distance_to_edge_cm if last_b.distance_to_edge_cm is not None else 5.0
            if d_edge >= 0 and not db:
                c_b = last_b.confidence if last_b.confidence is not None else 0.8
                # 检查最后一次击球是否在最后落台之前
                hit_before_last_b = (last_h is None or last_h.timestamp_s < last_b.timestamp_s)
                if hit_before_last_b:
                    if last_b.side == "B":
                        add_rule(
                            "UNRETURNED_IN_B", AdjudicationOutcome.POINT_A, 2.5, c_b,
                            f"球合法落在 B 侧台内后未被回击 (conf={c_b:.2f})",
                        )
                    elif last_b.side == "A":
                        add_rule(
                            "UNRETURNED_IN_A", AdjudicationOutcome.POINT_B, 2.5, c_b,
                            f"球合法落在 A 侧台内后未被回击 (conf={c_b:.2f})",
                        )

        # ---- 规则 4：落台在台外出界 (OUT / OFF_TABLE) -------------------------
        last_out = buffer.outs[-1] if buffer.outs else None
        if last_out:
            c_out = last_out.confidence if last_out.confidence is not None else 0.7
            last_h = buffer.hits[-1] if buffer.hits else None
            last_b = buffer.bounces[-1] if buffer.bounces else None
            hitter = last_h.side if last_h else buffer.server
            # OUT 也可能是合法落台后无人回击、球掉到地板的反弹。
            # 此时出界并非击球方失误；必须先检查同次击球的合法落台证据。
            landed_on_receiver = (
                last_b is not None
                and hitter in ("A", "B")
                and last_b.side in ("A", "B")
                and last_b.side != hitter
                and (last_h is None or last_h.timestamp_s < last_b.timestamp_s)
                and last_b.timestamp_s <= last_out.timestamp_s
                and (last_b.distance_to_edge_cm is None or last_b.distance_to_edge_cm >= 0)
            )
            if landed_on_receiver:
                c_b = last_b.confidence if last_b.confidence is not None else 0.8
                target = AdjudicationOutcome.POINT_A if hitter == "A" else AdjudicationOutcome.POINT_B
                add_rule(
                    "OUT_AFTER_LEGAL_BOUNCE", target, 3.0, min(c_out, c_b),
                    f"球合法落在 {last_b.side} 侧后未被回击，随后落到台外",
                )
            elif last_h:
                if last_h.side == "A":
                    add_rule("OUT_BY_A", AdjudicationOutcome.POINT_B, 3.0, c_out, "A方击球出界")
                elif last_h.side == "B":
                    add_rule("OUT_BY_B", AdjudicationOutcome.POINT_A, 3.0, c_out, "B方击球出界")
            else:
                if buffer.server == "A":
                    add_rule("SERVE_OUT_A", AdjudicationOutcome.POINT_B, 3.0, c_out, "A方发球出界")
                else:
                    add_rule("SERVE_OUT_B", AdjudicationOutcome.POINT_A, 3.0, c_out, "B方发球出界")
        elif buffer.bounces:
            last_b = buffer.bounces[-1]
            d_edge = last_b.distance_to_edge_cm
            if d_edge is not None and d_edge < -self.out_margin_cm:
                c_b = last_b.confidence if last_b.confidence is not None else 0.7
                last_h = buffer.hits[-1] if buffer.hits else None
                if last_h:
                    if last_h.side == "A":
                        add_rule("OFF_TABLE_A", AdjudicationOutcome.POINT_B, 2.8, c_b, f"A方击球落台外出界 ({d_edge:.1f}cm)")
                    elif last_h.side == "B":
                        add_rule("OFF_TABLE_B", AdjudicationOutcome.POINT_A, 2.8, c_b, f"B方击球落台外出界 ({d_edge:.1f}cm)")
                else:
                    if buffer.server == "A":
                        add_rule("SERVE_OFF_TABLE_A", AdjudicationOutcome.POINT_B, 2.8, c_b, f"A方发球落台外出界 ({d_edge:.1f}cm)")
                    else:
                        add_rule("SERVE_OFF_TABLE_B", AdjudicationOutcome.POINT_A, 2.8, c_b, f"B方发球落台外出界 ({d_edge:.1f}cm)")

        # ---- 规则 5：击球后落回己方半台 (OWN_COURT，不含发球自落) ------------
        for i in range(len(buffer.hits)):
            h = buffer.hits[i]
            # UNKNOWN 是信息缺失，不能将两个 UNKNOWN 当成同一位球员。
            if h.side not in ("A", "B"):
                continue
            # 若是发球方的第一次击球 (发球)，其落在己方半台是合法发球动作，不判失误
            if i == 0 and h.side == buffer.server:
                continue
            sub_b = [b for b in buffer.bounces if b.timestamp_s > h.timestamp_s]
            if sub_b:
                first_b = sub_b[0]
                # 若期间对方已有击球，说明落台属于对方击球之后，非己方自落
                intervening_opp_hit = any(
                    h2.side != h.side and h.timestamp_s < h2.timestamp_s < first_b.timestamp_s
                    for h2 in buffer.hits
                )
                if not intervening_opp_hit and first_b.side == h.side:
                    c_own = min(
                        h.confidence if h.confidence is not None else 0.8,
                        first_b.confidence if first_b.confidence is not None else 0.8,
                    )
                    opp = AdjudicationOutcome.POINT_B if h.side == "A" else AdjudicationOutcome.POINT_A
                    add_rule(
                        f"OWN_COURT_{h.side}", opp, 2.2, c_own,
                        f"{h.side}方击球后直接落回己方半台",
                    )

        # ---- 排序选出第一名与第二名 ------------------------------------------
        sorted_outcomes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1_outcome, top1_score = sorted_outcomes[0]
        top2_outcome, top2_score = sorted_outcomes[1]

        eps = 1e-6
        conf = (top1_score - top2_score) / (top1_score + top2_score + eps)

        # 两档置信度裁定
        if conf >= self.theta_hi and top1_score >= self.min_score_hi and top1_outcome != AdjudicationOutcome.INVALID:
            tier = "HIGH_CONFIDENCE"
        else:
            tier = "LOW_CONFIDENCE"

        # 胜者与比分结算
        winner = None
        new_score = score_before
        if top1_outcome == AdjudicationOutcome.POINT_A:
            winner = "A"
            new_score = (score_before[0] + 1, score_before[1])
        elif top1_outcome == AdjudicationOutcome.POINT_B:
            winner = "B"
            new_score = (score_before[0], score_before[1] + 1)

        expl = "; ".join(m.description for m in matches if m.target == top1_outcome)
        if not expl:
            expl = "无充分确凿证据，判定为无效球"

        return RallyDecision(
            round_id=buffer.round_id,
            timestamp_s=buffer.end_time,
            outcome=top1_outcome,
            winner=winner,
            conf=conf,
            tier=tier,
            score_before=score_before,
            score_after=new_score,
            server=buffer.server,
            scores_by_outcome={k.value: round(v, 3) for k, v in scores.items()},
            matched_rules=matches,
            events_in_rally=len(buffer.events),
            explanation=expl,
        )


class EvidenceStateMachine:
    """基于回合级证据的三阶段裁判状态机。

    阶段流程：
        PREP (准备期滑动窗口) ──[球第一次越网并合规]──► RALLY (收集回合证据)
                                                              │
                                                        [双跳/出界/超时]
                                                              │
                                                              ▼
        PREP ◄──[结算比分、清空缓冲、就绪]── CLOSED (假设竞争打分判决)
    """

    def __init__(
        self,
        calibration: TableCalibration,
        judge: Optional[HypothesisJudge] = None,
        prep_window_s: float = 2.0,
        rally_timeout_s: float = 0.8,
        points_to_win: int = 11,
        win_by: int = 2,
    ) -> None:
        self.calib = calibration
        self.judge = judge or HypothesisJudge(out_margin_cm=6.0)
        self.prep_window_s = float(prep_window_s)
        self.rally_timeout_s = float(rally_timeout_s)
        self.points_to_win = int(points_to_win)
        self.win_by = int(win_by)

        self.phase = RallyPhase.PREP
        self.points = {"A": 0, "B": 0}
        self.server = "A"              # 初始默认 A，越网时自适应推导
        self.server_inferred = False
        self.round_id = 0
        self.game_over = False

        self._prep_window: collections.deque = collections.deque()
        self._current_buffer: Optional[RallyBuffer] = None
        self._last_event_time: float = -999.0
        self._last_ball_x: Optional[float] = None
        self._decisions_history: List[RallyDecision] = []

        # 语音交互预留监听器 (用户确认 1: 准备好后续对接)
        self._low_confidence_listeners: List[Callable[[RallyDecision], None]] = []

    # ------------------------------------------------------------------
    # 外部驱动接口 (与现有 runner 兼容)
    # ------------------------------------------------------------------
    def update(self, event: RefereeEvent) -> List[RallyDecision]:
        """接收一个物理事件，驱动状态转移。"""
        if self.game_over:
            return []
        t = event.timestamp_s
        self._last_event_time = t
        out_decisions: List[RallyDecision] = []

        if self.phase == RallyPhase.PREP:
            # 准备期：进滑动窗口自然过期
            self._prune_prep_window(t)
            self._prep_window.append(event)

            # 检查是否发生【球第一次越网】
            crossing = self._detect_net_crossing(event)
            if crossing is not None:
                cross_dir, server_hint = crossing
                # 越网回头看发球合规度
                is_serve, s_compliance, inferred_srv = self._lookback_serve(t, server_hint, current_event=event)
                if is_serve:
                    # 合规越网，开启正式回合！
                    self.phase = RallyPhase.RALLY
                    self.server = inferred_srv
                    self.round_id += 1
                    self._current_buffer = RallyBuffer(
                        round_id=self.round_id,
                        server=self.server,
                        serve_compliance=s_compliance,
                        start_time=t,
                        end_time=t,
                    )
                    # 将越网前相关的发球动作（发球侧落台、触网等）纳入回合缓冲
                    earlier_events = [
                        e for e in self._prep_window
                        if e is not event and e.timestamp_s <= event.timestamp_s
                    ]
                    serve_bounce_idx = -1
                    for idx, prev_ev in enumerate(earlier_events):
                        if prev_ev.type == EventType.BOUNCE.value and prev_ev.side == self.server:
                            serve_bounce_idx = idx

                    if serve_bounce_idx >= 0:
                        for prev_ev in earlier_events[serve_bounce_idx:]:
                            self._current_buffer.append(prev_ev)

                    self._current_buffer.append(event)
            return []

        elif self.phase == RallyPhase.RALLY:
            if self._current_buffer is not None:
                self._current_buffer.append(event)

            # 提前结束触发 1：同半台连续二次落台
            if self._current_buffer and self._current_buffer.consecutive_double_bounce() is not None:
                dec = self._close_and_adjudicate(t)
                if dec:
                    out_decisions.append(dec)
                return out_decisions

            # 提前结束触发 2：明确出界
            if event.type == EventType.OUT.value:
                dec = self._close_and_adjudicate(t)
                if dec:
                    out_decisions.append(dec)
                return out_decisions

            return []

        elif self.phase == RallyPhase.CLOSED:
            # 结算中
            return []

        return []

    def tick(self, t: float) -> List[RallyDecision]:
        """时钟推进：驱动准备期窗口过期与回合超时结算。"""
        if self.game_over:
            return []
        if self.phase == RallyPhase.PREP:
            self._prune_prep_window(t)
            return []

        elif self.phase == RallyPhase.RALLY:
            # 超时结束：球连续消失超过窗口 (如 0.8s 无事件)
            if (t - self._last_event_time) >= self.rally_timeout_s:
                dec = self._close_and_adjudicate(t)
                if dec:
                    return [dec]
            return []

        return []

    # ------------------------------------------------------------------
    # 核心机制：发球越网回头看 (Lookback on Net Crossing)
    # ------------------------------------------------------------------
    def _detect_net_crossing(self, event: RefereeEvent) -> Optional[Tuple[str, str]]:
        """检测小球是否发生了穿过球网的动作。

        Returns: (方向 "A_TO_B" 或 "B_TO_A", 发球方提示 "A" 或 "B")
        """
        if event.position_px is not None:
            cur_x = event.position_px[0]
            net_x = self.calib.net_x
            res = None
            if self._last_ball_x is not None:
                prev_x = self._last_ball_x
                if prev_x < net_x <= cur_x:
                    res = ("A_TO_B", "A")
                elif prev_x > net_x >= cur_x:
                    res = ("B_TO_A", "B")
            self._last_ball_x = cur_x
            if res is not None:
                return res

        # 兜底：如果本次事件在对方半台且上一次落台在己方
        if self._prep_window:
            bounces = [e for e in self._prep_window if e.type == EventType.BOUNCE.value]
            if len(bounces) >= 2:
                b_prev, b_cur = bounces[-2], bounces[-1]
                if b_prev.side in ("A", "B") and b_cur.side in ("A", "B") and b_prev.side != b_cur.side:
                    srv = b_prev.side
                    direction = "A_TO_B" if srv == "A" else "B_TO_A"
                    return (direction, srv)
        return None

    def _lookback_serve(
        self, t: float, server_hint: str, current_event: Optional[RefereeEvent] = None
    ) -> Tuple[bool, float, str]:
        """等球越网，再回头看滑动窗口，检验是否是一次合法发球。"""
        if not self._prep_window:
            return (True, 0.7, server_hint)

        earlier = [
            e for e in self._prep_window
            if (current_event is None or e is not current_event)
        ]
        bounces = [e for e in earlier if e.type == EventType.BOUNCE.value]
        # 回看项 1：优先寻找与越网方向相符的发球方半台落台
        if bounces:
            matched = [b for b in bounces if b.side == server_hint]
            if matched:
                return (True, 1.0, server_hint)
            # 未找到完全相符但存在有效落台
            last_prep_bounce = bounces[-1]
            inferred_server = last_prep_bounce.side if last_prep_bounce.side in ("A", "B") else server_hint
            compliance = 0.8 if inferred_server == server_hint else 0.5
            return (True, compliance, inferred_server)

        return (True, 0.6, server_hint)

    def _prune_prep_window(self, t: float) -> None:
        """滑动窗口清理：超过 prep_window_s 的历史事件自动丢弃。"""
        cutoff = t - self.prep_window_s
        while self._prep_window and self._prep_window[0].timestamp_s < cutoff:
            self._prep_window.popleft()
        if not self._prep_window:
            # 越网方向的前一采样也必须与准备窗口一同过期。
            self._last_ball_x = None

    # ------------------------------------------------------------------
    # 状态结算与流转
    # ------------------------------------------------------------------
    def _close_and_adjudicate(self, t: float) -> Optional[RallyDecision]:
        """关闭回合，启动假设竞争引擎判决。"""
        if self._current_buffer is None:
            self.phase = RallyPhase.PREP
            return None

        self.phase = RallyPhase.CLOSED
        score_before = (self.points["A"], self.points["B"])
        decision = self.judge.adjudicate(self._current_buffer, score_before)

        # 比分与胜负状态生效
        self.points["A"] = decision.score_after[0]
        self.points["B"] = decision.score_after[1]
        self._decisions_history.append(decision)

        # 检查是否达到胜赛条件
        if self._is_game_won():
            self.game_over = True

        # 发球轮转 (每 2 分换发，10:10 以后每分换发)
        self._rotate_serve_if_needed()

        # 触发低置信度监听 (为后续语音纠错对接)
        if decision.tier == "LOW_CONFIDENCE":
            for cb in self._low_confidence_listeners:
                try:
                    cb(decision)
                except Exception:
                    pass

        # 清空缓冲区，重置为 PREP
        self._current_buffer = None
        self._prep_window.clear()
        self._last_ball_x = None
        self.phase = RallyPhase.PREP
        return decision

    def _is_game_won(self) -> bool:
        a, b = self.points["A"], self.points["B"]
        hi, lo = max(a, b), min(a, b)
        return hi >= self.points_to_win and (hi - lo) >= self.win_by

    def _rotate_serve_if_needed(self) -> None:
        total = self.points["A"] + self.points["B"]
        deuce = min(self.points["A"], self.points["B"]) >= (self.points_to_win - 1)
        rotation = 1 if deuce else 2
        if total > 0 and total % rotation == 0:
            self.server = "B" if self.server == "A" else "A"

    # ------------------------------------------------------------------
    # 预留后续对接语音接口 (用户确认 1: 准备好后续对接)
    # ------------------------------------------------------------------
    def register_low_confidence_listener(self, cb: Callable[[RallyDecision], None]) -> None:
        """注册低置信度判决监听器 (对接语音播报)。"""
        self._low_confidence_listeners.append(cb)

    def revoke_last_point(self) -> bool:
        """语音指令：“不对” / “撤销”。撤销最近一回合判决，恢复比分与发球方。"""
        if not self._decisions_history:
            return False
        last_dec = self._decisions_history.pop()
        # 恢复比分
        self.points["A"] = last_dec.score_before[0]
        self.points["B"] = last_dec.score_before[1]
        self.server = last_dec.server
        self.game_over = False
        return True

    def change_last_decision(self, winner: str) -> bool:
        """语音指令：“A得分” / “B得分”。强行改判最近一回合。"""
        if not self._decisions_history or winner not in ("A", "B"):
            return False
        last_dec = self._decisions_history[-1]
        # 基于该回合前比分重新改判
        prev_a, prev_b = last_dec.score_before
        if winner == "A":
            self.points["A"] = prev_a + 1
            self.points["B"] = prev_b
        else:
            self.points["A"] = prev_a
            self.points["B"] = prev_b + 1
        self._rotate_serve_if_needed()
        self.game_over = self._is_game_won()
        return True

    @property
    def score_text(self) -> str:
        return f"{self.points['A']} : {self.points['B']}"

    def describe(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "server": self.server,
            "score": self.score_text,
            "rounds_completed": len(self._decisions_history),
            "game_over": self.game_over,
        }
