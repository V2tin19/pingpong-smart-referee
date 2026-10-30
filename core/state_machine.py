"""
core/state_machine.py
=====================

共用计分状态机：把**物理事件**转换成**比分**。

这是组长方案 §1 架构的最后一层：

.. code-block:: text

    适配器 → FrameObservation → 轨迹/事件处理 → RefereeEvent → 【本模块】→ 比分

状态序列（组长方案 §3）
------------------------

.. code-block:: text

    等待发球 → 发球进行中 → 对打中 → 回合结束候选 → 确认得分
                                        └→ 无法判断，等待人工确认

关于"简单逻辑"版本
------------------
组长指示：**先让 AI 写简单逻辑的状态机，跑测试数据看实际评分效果，后面再加限制。**
因此本模块当前是**第一版简单实现**，判罚规则刻意从简，每条规则都带名字，
便于后续统计"分数是哪条规则给的"，也便于逐条加限制。

已实现的判罚规则（每条都可单独关闭）
------------------------------------
====================================  ==========================================
规则                                  含义
====================================  ==========================================
``SERVE_FAULT``                       发球没先落在自己半台 → 接发方得分
``DOUBLE_BOUNCE``                     球在同一侧连跳两次 → 另一侧得分
``OWN_COURT``                         球落在击球方**自己**半台 → 对手得分
``OUT``                               出界 → 击球方的对手得分
``NO_RETURN``（简化假设，可关）        球在对方半台落台后消失 → 击球方得分
====================================  ==========================================

``NO_RETURN`` 需要特别说明：组长 §4-2 要求"不要把球消失直接当作失分"。
本条**不是**单凭消失判分——它要求先有"球合法落在对方半台"这个正向证据，
消失只是"对方没接到"的佐证。但它仍是一个**简化假设**（也可能是我们跟丢了球），
因此默认开启但可一键关闭，后续应结合更多证据收紧。

对组长 §4-1 / §4-2 的落实
-------------------------
* **§4-1**：所有超时（回合终了、消抖）一律以**秒**为单位，按 ``timestamp_s`` 判定。
* **§4-2**：回合终了先进入 ``RALLY_END_CANDIDATE`` **缓冲**，不单帧判负；
  球消失本身**从不**直接产生得分。

``side=UNKNOWN`` 的处理（§7.4）
-------------------------------
``RefereeEvent.side`` 允许 ``UNKNOWN``（文档承诺"状态机会把 UNKNOWN 当作
信息缺失处理"），本模块兑现该承诺：

* ``UNKNOWN`` 的击球：**不建立发球归属、不更新击球方**（否则后续判罚会归到
  ``"UNKNOWN"`` 头上甚至崩溃），只把状态从等待带回到对打；
* ``UNKNOWN`` 的落台：**不参与任何判罚**，也不写入 ``last_bounce_side``
  （否则两次 UNKNOWN 落台会被误判成同侧二跳）；
* 原则：**已知信息不被未知信息覆盖，未知信息绝不参与判罚**。

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import EventType, RefereeEvent, Side

__all__ = [
    "MatchState",
    "MatchRules",
    "Decision",
    "DecisionKind",
    "TableTennisStateMachine",
]


class MatchState(str, Enum):
    """组长方案 §3 的状态序列。"""

    WAITING_SERVE = "WAITING_SERVE"                  # 等待发球
    SERVE_IN_PROGRESS = "SERVE_IN_PROGRESS"          # 发球进行中
    RALLY = "RALLY"                                  # 对打中
    RALLY_END_CANDIDATE = "RALLY_END_CANDIDATE"      # 回合结束候选（缓冲）
    CONFIRM_POINT = "CONFIRM_POINT"                  # 确认得分
    MANUAL_REVIEW = "MANUAL_REVIEW"                  # 无法判断，等待人工确认


class DecisionKind(str, Enum):
    POINT = "POINT"          # 判分
    LET = "LET"              # 擦网重发，比分不变
    DEAD_BALL = "DEAD_BALL"  # 回合终了但无法判定，不计分
    GAME_OVER = "GAME_OVER"  # 一局结束
    REVIEW = "REVIEW"        # 需要人工确认


@dataclass(frozen=True)
class MatchRules:
    """比赛规则与判罚参数（全部可调，便于逐条加限制）。"""

    points_to_win: int = 11          # ITTF：一局 11 分
    win_by: int = 2                  # 需净胜 2 分
    serve_rotation: int = 2          # 每 2 分换发球
    deuce_serve_rotation: int = 1    # 10 平后每 1 分换发

    # 时间参数（秒）—— 组长 §4-1 要求，禁止帧数
    rally_end_s: float = 0.8         # 无事件多久算"回合可能结束"
    confirm_s: float = 0.4           # 候选→确认的消抖缓冲
    # 发球判定：只有**死球停顿**之后的第一次击球才算发球。
    #
    # 为什么需要这一条（实测教训）：若不要求停顿，那么"重置后的第一次击球"
    # 就会被当成发球；可正常对打中击球后球本来就该落在**对方**半台，
    # 于是被 SERVE_FAULT 误判。test_game_2 上这一条曾误报 8 次（共 12 分）。
    # 真实比赛里每分之间有数秒停顿，而对拉中两次击球间隔通常 <1s，
    # 因此用"停顿"来区分发球与对打击球是可靠的。
    serve_min_gap_s: float = 1.5

    # 判罚规则开关（便于逐条加限制 / 做消融）
    enable_serve_fault: bool = True
    enable_double_bounce: bool = True
    enable_own_court: bool = True
    enable_out: bool = True
    enable_no_return: bool = True    # 简化假设，见模块文档

    def opponent(self, side: str) -> str:
        return "B" if side == "A" else "A"


@dataclass
class Decision:
    """一次判罚。"""

    timestamp_s: float
    kind: DecisionKind
    winner: Optional[str] = None     # "A" / "B" / None
    reason: str = ""
    rule: Optional[str] = None       # 触发的规则名，便于统计与调试
    score: Tuple[int, int] = (0, 0)  # (A, B)
    state: str = ""

    def describe(self) -> str:
        if self.kind == DecisionKind.POINT:
            who = "P1(左)" if self.winner == "A" else "P2(右)"
            return f"{who} 得分 [{self.rule}] {self.reason}  {self.score[0]}:{self.score[1]}"
        if self.kind == DecisionKind.LET:
            return f"擦网重发 [{self.rule}] {self.reason}"
        if self.kind == DecisionKind.GAME_OVER:
            return f"本局结束 {self.score[0]}:{self.score[1]}"
        if self.kind == DecisionKind.REVIEW:
            return f"无法判断，待人工确认 [{self.rule}] {self.reason}"
        return f"死球 [{self.rule}] {self.reason}"


@dataclass
class _RallyContext:
    """当前回合的上下文。"""

    server: str = "A"
    last_hitter: Optional[str] = None
    last_bounce_side: Optional[str] = None
    serve_own_bounce_seen: bool = False
    events_in_rally: int = 0
    last_event_time: Optional[float] = None
    candidate_since: Optional[float] = None


class TableTennisStateMachine:
    """ITTF 风格共用计分状态机（第一版简单逻辑）。

    Parameters
    ----------
    rules:
        见 :class:`MatchRules`。
    first_server:
        首局首发方（``"A"`` / ``"B"``）。

    Examples
    --------
    >>> from core.schemas import RefereeEvent
    >>> sm = TableTennisStateMachine()
    >>> ev = RefereeEvent(event_id="bounce_1", timestamp_s=1.0, type="BOUNCE", side="B")
    >>> sm.update(ev)                       # doctest: +SKIP
    []
    """

    def __init__(self, rules: Optional[MatchRules] = None, first_server: str = "A") -> None:
        if first_server not in ("A", "B"):
            raise ValueError(f"first_server must be 'A' or 'B', got {first_server!r}")
        self.rules = rules or MatchRules()
        self._first_server = first_server
        self.reset()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.points: Dict[str, int] = {"A": 0, "B": 0}
        self.state: MatchState = MatchState.WAITING_SERVE
        self._ctx = _RallyContext(server=self._first_server)
        self._last_timestamp: Optional[float] = None
        # 机器级的"上一个事件时刻"，**不随回合重置清空** —— 发球判定要靠它
        # 判断两次击球之间是否发生了死球停顿。
        self._last_event_time: Optional[float] = None
        self.game_over = False
        self.decisions: List[Decision] = []
        # 统计：每个规则贡献了多少分
        self.rule_counts: Dict[str, int] = {}
        self.review_count = 0
        self.dead_ball_count = 0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def update(self, event: RefereeEvent) -> List[Decision]:
        """喂入一个物理事件，返回由此产生的判罚（通常 0~1 条）。"""
        if not isinstance(event, RefereeEvent):
            raise TypeError(f"event must be RefereeEvent, got {type(event).__name__}")

        t = event.timestamp_s
        if self._last_timestamp is not None and t < self._last_timestamp:
            raise ValueError(
                f"event timestamp went backwards: {t} < {self._last_timestamp}. Call reset()."
            )
        self._last_timestamp = t

        if self.game_over:
            return []

        # 回合结束候选期间来了新事件 → 说明球又出现了，取消候选、回到对打
        if self.state == MatchState.RALLY_END_CANDIDATE:
            self.state = MatchState.RALLY
            self._ctx.candidate_since = None

        self._ctx.last_event_time = t
        self._ctx.events_in_rally += 1
        prev_event_time = self._last_event_time
        self._last_event_time = t

        etype = event.type
        side = event.side

        if etype == EventType.RACKET_HIT.value:
            return self._on_racket_hit(side, t, prev_event_time)
        if etype == EventType.BOUNCE.value:
            return self._on_bounce(side, t)
        if etype == EventType.NET_TOUCH.value:
            return self._on_net_touch(t)
        if etype == EventType.OUT.value:
            return self._on_out(side, t)
        return []

    def tick(self, timestamp_s: float) -> List[Decision]:
        """时间推进。每帧调用一次，用于处理"球消失"这类**超时**判据。

        组长 §4-1：超时一律用秒，不用帧数。
        """
        if self.game_over:
            return []
        t = float(timestamp_s)
        if self._last_timestamp is not None and t < self._last_timestamp:
            raise ValueError(f"tick timestamp went backwards: {t} < {self._last_timestamp}")

        out: List[Decision] = []
        ctx = self._ctx
        last = ctx.last_event_time

        # 发球阶段长时间没有后续事件 → 发球没打完（球消失/漏检）。
        # 不能猜是发球失误，按死球处理并回到等待发球，否则状态机会永久卡住。
        if (
            self.state == MatchState.SERVE_IN_PROGRESS
            and last is not None
            and (t - last) >= (self.rules.rally_end_s + self.rules.confirm_s)
        ):
            self.dead_ball_count += 1
            decision = Decision(
                timestamp_s=t, kind=DecisionKind.DEAD_BALL, rule="SERVE_INCOMPLETE",
                reason="发球阶段球消失，证据不足，不判分", score=self._score(),
                state=self.state.value,
            )
            self.decisions.append(decision)
            self._reset_rally(keep_server=True)
            return [decision]

        # 对打中长时间没有事件 → 回合结束候选（先缓冲，不直接判分）
        if (
            self.state == MatchState.RALLY
            and last is not None
            and (t - last) >= self.rules.rally_end_s
        ):
            self.state = MatchState.RALLY_END_CANDIDATE
            ctx.candidate_since = t

        # 候选期满了 → 裁决
        if (
            self.state == MatchState.RALLY_END_CANDIDATE
            and ctx.candidate_since is not None
            and (t - ctx.candidate_since) >= self.rules.confirm_s
        ):
            out.extend(self._resolve_rally_end(t))

        return out

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------
    def _on_racket_hit(self, side: str, t: float, prev_event_time: Optional[float]) -> List[Decision]:
        ctx = self._ctx

        if side == Side.UNKNOWN.value:
            # §7.4：UNKNOWN = 信息缺失（另一分支的事件层无法判定击球方）。
            # 此时既不能建立发球归属（server 无从确定），也不能更新 last_hitter
            # ——否则后续判罚会归到 "UNKNOWN" 头上，甚至触发 points KeyError。
            # 采取"保留已知、不猜未知"：清空击球归属，让后续落台只记录不判罚，
            # 直到下一次已知方的击球重新建立归属。
            if self.state == MatchState.WAITING_SERVE:
                self.state = MatchState.RALLY
            ctx.last_hitter = None
            ctx.last_bounce_side = None
            return []

        if self.state == MatchState.WAITING_SERVE:
            gap = (t - prev_event_time) if prev_event_time is not None else float("inf")
            if gap >= self.rules.serve_min_gap_s:
                # 死球停顿之后的第一次击球 → 这才是发球
                ctx.server = side
                ctx.serve_own_bounce_seen = False
                self.state = MatchState.SERVE_IN_PROGRESS
            else:
                # 上一个回合刚结束就又有击球 → 其实是对打的延续，
                # 不能当成发球，否则会把正常的"打到对方半台"误判为发球失误。
                self.state = MatchState.RALLY

        ctx.last_hitter = side
        ctx.last_bounce_side = None
        return []

    def _on_bounce(self, side: str, t: float) -> List[Decision]:
        ctx = self._ctx
        rules = self.rules

        if side == Side.UNKNOWN.value:
            # §7.4：信息缺失的落台不参与任何判罚，也不写入 last_bounce_side
            # ——否则两次 UNKNOWN 落台会被当成"同侧二跳"误判 DOUBLE_BOUNCE，
            # 或者把已知的 last_bounce_side 覆盖成 UNKNOWN 污染后续判定。
            return []

        # ---- 发球阶段 ----
        if self.state == MatchState.SERVE_IN_PROGRESS:
            if not ctx.serve_own_bounce_seen:
                if side == ctx.server:
                    ctx.serve_own_bounce_seen = True     # 合法的发球第一跳
                    ctx.last_bounce_side = side
                    return []
                if rules.enable_serve_fault:
                    # 发球没有先落在自己半台
                    return self._award(
                        rules.opponent(ctx.server), t, "SERVE_FAULT",
                        f"发球未落在发球方({ctx.server})半台，先落在 {side}",
                    )
                ctx.serve_own_bounce_seen = True
                ctx.last_bounce_side = side
                return []
            # 己方第一跳已出现
            if side == ctx.server:
                if rules.enable_double_bounce:
                    return self._award(
                        rules.opponent(ctx.server), t, "DOUBLE_BOUNCE",
                        f"发球在己方({ctx.server})半台连跳两次",
                    )
            else:
                # 发球成功越网落到对方半台 → 进入对打
                self.state = MatchState.RALLY
                ctx.last_bounce_side = side
            return []

        # ---- 对打阶段 ----
        if self.state in (MatchState.RALLY, MatchState.RALLY_END_CANDIDATE):
            if ctx.last_hitter is None:
                ctx.last_bounce_side = side
                return []

            # 球落在击球方自己半台 → 失误
            if side == ctx.last_hitter:
                if rules.enable_own_court:
                    return self._award(
                        rules.opponent(ctx.last_hitter), t, "OWN_COURT",
                        f"球落在击球方({ctx.last_hitter})自己半台",
                    )
                ctx.last_bounce_side = side
                return []

            # 同一侧连跳两次 → 该侧球员没接到
            if side == ctx.last_bounce_side:
                if rules.enable_double_bounce:
                    return self._award(
                        ctx.last_hitter, t, "DOUBLE_BOUNCE",
                        f"球在 {side} 侧连跳两次，{side} 方未接到",
                    )
            ctx.last_bounce_side = side
            return []

        # 其它状态（等待发球等）下的落台不判分
        ctx.last_bounce_side = side
        return []

    def _on_net_touch(self, t: float) -> List[Decision]:
        """擦网。发球阶段擦网且仍落对方半台 → LET（重发，比分不变）。"""
        if self.state == MatchState.SERVE_IN_PROGRESS:
            decision = Decision(
                timestamp_s=t, kind=DecisionKind.LET, rule="LET",
                reason="发球擦网，重发", score=self._score(), state=self.state.value,
            )
            self.decisions.append(decision)
            self._reset_rally(keep_server=True)
            return [decision]
        # 对打中擦网不判分（简化）
        return []

    def _on_out(self, side: str, t: float) -> List[Decision]:
        ctx = self._ctx
        if not self.rules.enable_out:
            return []
        if self.state == MatchState.SERVE_IN_PROGRESS:
            return self._award(
                self.rules.opponent(ctx.server), t, "OUT", "发球出界"
            )
        if self.state in (MatchState.RALLY, MatchState.RALLY_END_CANDIDATE) and ctx.last_hitter:
            return self._award(
                self.rules.opponent(ctx.last_hitter), t, "OUT",
                f"击球方({ctx.last_hitter})出界",
            )
        return []

    # ------------------------------------------------------------------
    # 回合终了裁决
    # ------------------------------------------------------------------
    def _resolve_rally_end(self, t: float) -> List[Decision]:
        """候选期满：决定这一分归谁，或者承认判不出来。"""
        ctx = self._ctx
        rules = self.rules

        # 简化假设：球合法落在对方半台之后消失 → 对方没接到
        if (
            rules.enable_no_return
            and ctx.last_hitter is not None
            and ctx.last_bounce_side is not None
            and ctx.last_bounce_side != ctx.last_hitter
        ):
            return self._award(
                ctx.last_hitter, t, "NO_RETURN",
                f"球落在 {ctx.last_bounce_side} 侧后消失，判定对方未接到",
            )

        # 判不出来：**不猜**，记为死球，回到等待发球
        self.dead_ball_count += 1
        decision = Decision(
            timestamp_s=t, kind=DecisionKind.DEAD_BALL, rule="NO_EVIDENCE",
            reason="回合结束但证据不足，不判分", score=self._score(), state=self.state.value,
        )
        self.decisions.append(decision)
        self._reset_rally(keep_server=True)
        return [decision]

    # ------------------------------------------------------------------
    # 判分
    # ------------------------------------------------------------------
    def _award(self, winner: str, t: float, rule: str, reason: str) -> List[Decision]:
        # 防御性校验：points 字典只有 "A"/"B" 两个键。任何判罚路径若把
        # "UNKNOWN" / None 当成胜者传进来，宁可当场抛错也不要静默记错分。
        if winner not in ("A", "B"):
            raise ValueError(f"award winner must be 'A' or 'B', got {winner!r} (rule={rule})")
        self.points[winner] += 1
        self.rule_counts[rule] = self.rule_counts.get(rule, 0) + 1

        decision = Decision(
            timestamp_s=t, kind=DecisionKind.POINT, winner=winner, rule=rule,
            reason=reason, score=self._score(), state=self.state.value,
        )
        self.decisions.append(decision)
        self.state = MatchState.CONFIRM_POINT

        out = [decision]

        # 局结束？
        if self._is_game_won():
            self.game_over = True
            game = Decision(
                timestamp_s=t, kind=DecisionKind.GAME_OVER,
                winner="A" if self.points["A"] > self.points["B"] else "B",
                rule="GAME", reason="达到局分", score=self._score(), state=self.state.value,
            )
            self.decisions.append(game)
            out.append(game)
            return out

        self._rotate_serve_if_needed()
        self._reset_rally(keep_server=True)
        return out

    def _is_game_won(self) -> bool:
        a, b = self.points["A"], self.points["B"]
        hi, lo = max(a, b), min(a, b)
        return hi >= self.rules.points_to_win and (hi - lo) >= self.rules.win_by

    def _rotate_serve_if_needed(self) -> None:
        total = self.points["A"] + self.points["B"]
        rotation = (
            self.rules.deuce_serve_rotation
            if min(self.points["A"], self.points["B"]) >= self.rules.points_to_win - 1
            else self.rules.serve_rotation
        )
        if total % rotation == 0:
            self._ctx.server = self.rules.opponent(self._ctx.server)

    def _reset_rally(self, keep_server: bool = True) -> None:
        server = self._ctx.server
        self._ctx = _RallyContext(server=server)
        self.state = MatchState.WAITING_SERVE

    def _score(self) -> Tuple[int, int]:
        return (self.points["A"], self.points["B"])

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def server(self) -> str:
        return self._ctx.server

    @property
    def score_text(self) -> str:
        return f"{self.points['A']} : {self.points['B']}"

    def describe(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "server": self._ctx.server,
            "points": dict(self.points),
            "game_over": self.game_over,
            "rule_counts": dict(self.rule_counts),
            "dead_ball": self.dead_ball_count,
            "review": self.review_count,
            "decisions": len(self.decisions),
        }

    def __repr__(self) -> str:
        return (
            f"<TableTennisStateMachine {self.score_text} state={self.state.value} "
            f"server={self._ctx.server}>"
        )
