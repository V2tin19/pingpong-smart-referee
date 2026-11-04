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

    等待发球 → 发球准入候选 → 对打中(可判分) → 回合结束候选 → 确认得分(冷却)
                                        └→ 无法判断，等待人工确认

判罚规则（每条都可单独关闭）
----------------------------
====================================  ==========================================
规则                                  含义
====================================  ==========================================
``SERVE_FAULT``                       发球没先落在自己半台 → 接发方得分（仅旧版）
``DOUBLE_BOUNCE``                     球在同一侧连跳两次 → 另一侧得分
``OWN_COURT``                         球落在击球方**自己**半台 → 对手得分
``OUT``                               出界 → 击球方的对手得分
``NO_RETURN``（简化假设，默认关）      球在对方半台落台后消失 → 击球方得分
====================================  ==========================================

四条门控（学长 2026-11-04 布置的"参考门控"）
--------------------------------------------
针对 Test4 实测暴露的两大误判源（虚假击球洪流、发球前颠球被判 DOUBLE_BOUNCE
抢跑送分，见 ``docs/test4_evaluation_report.md``），在判罚之前加四道门：

1. **发球准入**（``enable_serve_admission``，默认开）
   必须见证 **"本方台面落台 → 对方台面落台"** 序列，回合才升为可判分；
   准入期间的同侧再弹视为**发球前的准备动作**（颠球/拍球找手感），重置候选、
   绝不判 DOUBLE_BOUNCE；序列始终无法完成的，由超时作废（死球，不判分）。
   发球阶段的 ``SERVE_FAULT`` / ``DOUBLE_BOUNCE`` 判分随之移入旧版模式
   （``enable_serve_admission=False`` 才可用）。

2. **发球前击球校验**（``pre_serve_hit_window_s``，默认 1.5s）
   准入序列完成（升级）时刻，要求最近一次 ``RACKET_HIT`` 距今不超过窗口：
   手抛传球、滚球等**没有球拍接触**的弹跳序列直接挡下，绝不升级。

3. **回合内击球要求**（``enable_rally_hit_required`` / ``enable_splice_guard``，默认开）
   * 升级后的回合内**至少出现过一次已知击球**才允许任何判分——发球得分家族
     （ace、发球二跳）在证据不足时一律放过，宁漏判不误判；
   * "同侧两弹之间夹一次击球"视为**序列误拼**（虚假击球 + 漏检拼接的典型形态，
     见 Test4 问题一），不计分。

4. **判分后冷却与默认不判分**（``score_cooldown_s`` / ``enable_no_return``）
   * 判分后进入 **2s 冷却**（``CONFIRM_POINT`` 状态），**忽略全部事件**——
     捡球、庆祝、地板反光都进不来；
   * 静默超时与视频结束**默认只清理证据、不判分**：``NO_RETURN`` 默认关闭
     （原默认开启），回合消失一律记死球。

每道门都会在 ``describe()["gates"]`` 里留下触发计数，便于统计"挡掉了什么"。

对组长 §4-1 / §4-2 的落实
-------------------------
* **§4-1**：所有超时（准入作废、回合终了、消抖、冷却）一律以**秒**为单位。
* **§4-2**：回合终了先进入 ``RALLY_END_CANDIDATE`` **缓冲**，不单帧判负；
  球消失本身**从不**直接产生得分（``NO_RETURN`` 也需要正向落台证据且默认关闭）。

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
    SERVE_IN_PROGRESS = "SERVE_IN_PROGRESS"          # 发球进行中（= 发球准入候选）
    RALLY = "RALLY"                                  # 对打中（已通过准入，可判分）
    RALLY_END_CANDIDATE = "RALLY_END_CANDIDATE"      # 回合结束候选（缓冲）
    CONFIRM_POINT = "CONFIRM_POINT"                  # 确认得分（判分后冷却期）
    MANUAL_REVIEW = "MANUAL_REVIEW"                  # 无法判断，等待人工确认


class DecisionKind(str, Enum):
    POINT = "POINT"          # 判分
    LET = "LET"              # 擦网重发，比分不变
    DEAD_BALL = "DEAD_BALL"  # 回合终了但无法判定，不计分
    GAME_OVER = "GAME_OVER"  # 一局结束
    REVIEW = "REVIEW"        # 需要人工确认


@dataclass(frozen=True)
class MatchRules:
    """比赛规则、判罚参数与门控开关（全部可调，便于逐条加限制 / 消融）。"""

    points_to_win: int = 11          # ITTF：一局 11 分
    win_by: int = 2                  # 需净胜 2 分
    serve_rotation: int = 2          # 每 2 分换发球
    deuce_serve_rotation: int = 1    # 10 平后每 1 分换发

    # 时间参数（秒）—— 组长 §4-1 要求，禁止帧数
    rally_end_s: float = 0.8         # 无事件多久算"回合可能结束"
    confirm_s: float = 0.4           # 候选→确认的消抖缓冲
    # 发球判定（**仅旧版模式 enable_serve_admission=False 时使用**）：
    # 只有死球停顿之后的第一次击球才算发球。门控 1（发球准入）上线后，
    # "发球 vs 对打击球"的区分改由落台序列承担，本参数退居兼容开关。
    serve_min_gap_s: float = 1.5

    # ---- 门控 1：发球准入 -------------------------------------------------
    # 必须见证"本方落台→对方落台"才升为可判分回合；准入期间同侧再弹视为
    # 发球前准备动作（颠球/拍球），重置候选不判罚；候选超时作废。
    # 依据：Test4 帧 828 把发球前第 2 次颠球判成 DOUBLE_BOUNCE，比赛未开
    # 就送出 0:1。关闭本门控退回旧版行为（SERVE_FAULT / 发球二跳判分）。
    enable_serve_admission: bool = True

    # ---- 门控 2：发球前击球校验 -------------------------------------------
    # 升级（本方→对方序列完成）时刻，最近一次 RACKET_HIT 必须在此窗口内：
    # 手抛传球、滚球等没有球拍接触的弹跳序列直接挡下。
    # 实测依据：真实发球从击球到对方半台落台约 0.5~1.1s，1.5s 留有余量。
    pre_serve_hit_window_s: float = 1.5

    # ---- 门控 3：回合内击球要求 -------------------------------------------
    # 3a：升级后的回合内至少出现过一次已知击球才允许任何判分。
    #     默认为 False（允许发球得分 / 接发方未触球的二跳失误正常判分；
    #     若担心强噪点场景下的虚假落台可设为 True 宁漏勿误）。
    enable_rally_hit_required: bool = True
    # 3b：同侧两弹之间夹一次击球 = 序列误拼（虚假击球 + 漏检拼接的典型形态，
    #     Test4 问题一），不计分。关闭后恢复旧版 OWN_COURT 敏感度。
    enable_splice_guard: bool = True

    # ---- 门控 4：判分后冷却与默认不判分 -------------------------------------
    # 判分后进入 CONFIRM_POINT 冷却，忽略全部事件（捡球/庆祝/地板反光进不来）。
    score_cooldown_s: float = 2.0
    # 静默超时默认只清理证据、不判分：NO_RETURN 由默认开启改为默认关闭。
    enable_no_return: bool = False

    # 判罚规则开关（便于逐条加限制 / 做消融）
    enable_serve_fault: bool = True   # 仅旧版模式（见 enable_serve_admission）
    enable_double_bounce: bool = True
    enable_own_court: bool = True
    enable_out: bool = True

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
    last_own_bounce_time: Optional[float] = None
    events_in_rally: int = 0
    last_event_time: Optional[float] = None
    candidate_since: Optional[float] = None
    # ---- 门控记账 ---------------------------------------------------------
    # 门控 3a：升级为可判分回合后，是否出现过已知方的击球
    rally_has_hit: bool = False
    # 门控 3b：上一次落台之后是否有过击球（用于识别"同侧两弹夹击球"）
    hit_since_last_bounce: bool = False
    # 门控 2：最近一次击球时刻（升级时校验"发球前必须有击球"）
    last_hit_time: Optional[float] = None


class TableTennisStateMachine:
    """ITTF 风格共用计分状态机（门控版）。

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
        # 机器级的"上一个事件时刻"，**不随回合重置清空**。
        self._last_event_time: Optional[float] = None
        # 门控 4：判分后冷却的截止时刻（None = 不在冷却期）
        self._cooldown_until: Optional[float] = None
        # 入口幂等：同一个 event_id 只处理一次（上游重发/迟到投递不得重复判分）。
        # 用有序字典实现"有界去重集合"。注意：事件 ID 由轨迹处理器的计数器生成，
        # 处理器 reset() 后计数器会从 1 重新开始 —— 因此换视频/跳转/回放时，
        # 必须把状态机与处理器**一并 reset**（run_referee / run_end_to_end 均如此）。
        self._seen_event_ids: Dict[str, None] = {}
        self.game_over = False
        self.decisions: List[Decision] = []
        # 统计：每个规则贡献了多少分
        self.rule_counts: Dict[str, int] = {}
        self.review_count = 0
        self.dead_ball_count = 0
        # 门控触发计数：每道门"挡掉了什么"，便于统计与消融对比
        self._gate_stats: Dict[str, int] = {
            "serve_prep_resets": 0,         # 门控1：同侧再弹视为准备动作，重置候选
            "serve_admission_violations": 0,  # 门控1：未先落本方半台，序列不成立
            "serve_no_hit_blocked": 0,      # 门控2：发球前无（近期）击球，挡下
            "serve_phase_ignored": 0,       # 门控1：准入期间 OUT 不判分
            "splice_skips": 0,              # 门控3b：同侧两弹夹击球，视为误拼
            "no_hit_skips": 0,              # 门控3a：回合内无击球，放弃判分
            "cooldown_ignored_events": 0,   # 门控4：冷却期忽略的事件数
            "duplicate_event_ids_ignored": 0,  # 入口幂等：重复 event_id 忽略数
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def update(self, event: RefereeEvent) -> List[Decision]:
        """喂入一个物理事件，返回由此产生的判罚（通常 0~1 条）。"""
        if not isinstance(event, RefereeEvent):
            raise TypeError(f"event must be RefereeEvent, got {type(event).__name__}")

        # 入口幂等：同一个 event_id 只提交一次。重复投递（上游重发、迟到返回、
        # 调用方 bug）静默忽略并计数，不得重复判分。刻意放在**时间单调校验之前**：
        # 携带旧时间戳的重复投递应当被忽略，而不是触发"时间倒退"异常。
        if event.event_id in self._seen_event_ids:
            self._gate_stats["duplicate_event_ids_ignored"] += 1
            return []
        self._seen_event_ids[event.event_id] = None
        if len(self._seen_event_ids) > 512:          # 有界窗口，防长比赛内存膨胀
            del self._seen_event_ids[next(iter(self._seen_event_ids))]

        t = event.timestamp_s
        if self._last_timestamp is not None and t < self._last_timestamp:
            raise ValueError(
                f"event timestamp went backwards: {t} < {self._last_timestamp}. Call reset()."
            )
        self._last_timestamp = t

        if self.game_over:
            return []

        # 门控 4：判分后冷却 —— 忽略全部事件（捡球/庆祝/反光都进不来）。
        # 冷却期内状态停在 CONFIRM_POINT；期满后的第一个事件先触发回合重置，
        # 再按正常流程处理（兼容不调用 tick 的使用方式）。
        if self.state == MatchState.CONFIRM_POINT and self._cooldown_until is not None:
            if t < self._cooldown_until:
                self._gate_stats["cooldown_ignored_events"] += 1
                return []
            self._cooldown_until = None
            self._reset_rally(keep_server=True)

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

        # 门控 4：冷却期满 → 落地回合重置（换发已在判分时完成，这里只清证据）
        if (
            self.state == MatchState.CONFIRM_POINT
            and self._cooldown_until is not None
            and t >= self._cooldown_until
        ):
            self._cooldown_until = None
            self._reset_rally(keep_server=True)

        # 发球准入候选长时间没有后续事件 → 序列没打完（球消失/漏检）。
        # 门控 1 的"候选超时作废"：不能猜是发球失误，按死球处理并回到等待发球。
        if (
            self.state == MatchState.SERVE_IN_PROGRESS
            and last is not None
            and (t - last) >= (self.rules.rally_end_s + self.rules.confirm_s)
        ):
            self.dead_ball_count += 1
            decision = Decision(
                timestamp_s=t, kind=DecisionKind.DEAD_BALL, rule="SERVE_INCOMPLETE",
                reason="发球准入候选超时作废，不判分", score=self._score(),
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

        # 候选期满了 → 裁决（门控 4：默认只清理证据，不判分）
        if (
            self.state == MatchState.RALLY_END_CANDIDATE
            and ctx.candidate_since is not None
            and (t - ctx.candidate_since) >= self.rules.confirm_s
        ):
            out.extend(self._resolve_rally_end(t))

        return out

    # ------------------------------------------------------------------
    # 事件处理：击球
    # ------------------------------------------------------------------
    def _on_racket_hit(self, side: str, t: float, prev_event_time: Optional[float]) -> List[Decision]:
        ctx = self._ctx
        rules = self.rules

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
            ctx.hit_since_last_bounce = False
            return []

        if self.state == MatchState.WAITING_SERVE:
            if rules.enable_serve_admission:
                # 门控 1+2：已知方的击球开启"发球准入候选"。
                # 是否真发球，由后续"本方落台→对方落台"序列与击球窗口共同验证；
                # 序列完成前的同侧弹跳（颠球/拍球）会被门控 1 重置，不产生判罚。
                ctx.server = side
                ctx.serve_own_bounce_seen = False
                ctx.last_bounce_side = None
                ctx.hit_since_last_bounce = False
                ctx.last_hit_time = t
                ctx.last_hitter = side
                self.state = MatchState.SERVE_IN_PROGRESS
                return []
            # 旧版模式：死球停顿区分发球与对打延续（R-015）
            gap = (t - prev_event_time) if prev_event_time is not None else float("inf")
            if gap >= rules.serve_min_gap_s:
                ctx.server = side
                ctx.serve_own_bounce_seen = False
                self.state = MatchState.SERVE_IN_PROGRESS
            else:
                # 对打的延续：直接进入可判分对打（旧行为）
                self.state = MatchState.RALLY
                ctx.rally_has_hit = True
                ctx.hit_since_last_bounce = rules.enable_splice_guard
            ctx.last_hit_time = t
            ctx.last_hitter = side
            ctx.last_bounce_side = None
            return []

        if self.state == MatchState.SERVE_IN_PROGRESS:
            # 准入候选期间的再次击球（颠球触拍 / 虚假击球 / 重新发球）
            ctx.last_hit_time = t
            ctx.last_hitter = side
            if rules.enable_serve_admission:
                # 再次击球说明上一轮击球/弹跳已作废（如发球前先颠了球再击球发球，或重新挥拍发球）。
                # 重新以最新击球定位发球方，并重置己方落台标记，防止陈旧落台污染真实发球。
                ctx.server = side
                ctx.serve_own_bounce_seen = False
                ctx.last_own_bounce_time = None
                ctx.last_bounce_side = None
            else:
                ctx.last_bounce_side = None
            return []

        # ---- 对打中的击球（RALLY / RALLY_END_CANDIDATE）----
        ctx.last_hit_time = t
        ctx.last_hitter = side
        if self.state in (MatchState.RALLY, MatchState.RALLY_END_CANDIDATE):
            ctx.rally_has_hit = True
            if rules.enable_splice_guard:
                # 门控 3b 记账：保留 last_bounce_side 作为"前弹"锚点，
                # "同侧两弹夹击球"在落台侧识别并拦截。
                ctx.hit_since_last_bounce = True
            else:
                # 旧语义：击球后清除落台锚点
                ctx.last_bounce_side = None
        else:
            ctx.last_bounce_side = None
        return []

    # ------------------------------------------------------------------
    # 事件处理：落台
    # ------------------------------------------------------------------
    def _on_bounce(self, side: str, t: float) -> List[Decision]:
        if side == Side.UNKNOWN.value:
            # §7.4：信息缺失的落台不参与任何判罚，也不写入 last_bounce_side
            # ——否则两次 UNKNOWN 落台会被当成"同侧二跳"误判 DOUBLE_BOUNCE，
            # 或者把已知的 last_bounce_side 覆盖成 UNKNOWN 污染后续判定。
            return []

        if self.state == MatchState.SERVE_IN_PROGRESS:
            if self.rules.enable_serve_admission:
                return self._on_bounce_serve_admission(side, t)
            return self._on_bounce_serve_legacy(side, t)

        if self.state in (MatchState.RALLY, MatchState.RALLY_END_CANDIDATE):
            return self._on_bounce_rally(side, t)

        # 其它状态（等待发球等）下的落台不判分
        self._ctx.last_bounce_side = side
        return []

    def _on_bounce_serve_admission(self, side: str, t: float) -> List[Decision]:
        """门控 1 发球准入：必须见证"本方落台→对方落台"才升为可判分回合。"""
        ctx = self._ctx
        rules = self.rules

        if side == ctx.server:
            if ctx.serve_own_bounce_seen:
                # 同侧再弹 = 发球前的准备动作（颠球/拍球找手感）→ 重置准入候选。
                # 这修复了 Test4 帧 828 的抢跑误判：发球前第 2 次颠球曾被直接
                # 判成 DOUBLE_BOUNCE 送分。
                self._gate_stats["serve_prep_resets"] += 1
                ctx.serve_own_bounce_seen = False
                ctx.last_bounce_side = None
                return []
            ctx.serve_own_bounce_seen = True     # 序列第一段：本方半台落台
            ctx.last_bounce_side = side
            return []

        # ---- 落在对方半台 ----
        if not ctx.serve_own_bounce_seen:
            # 序列不成立（未先落本方半台）→ 不判 SERVE_FAULT：多半是己方第一跳
            # 被漏检。候选保持等待，若始终无法完成序列则由 tick 超时作废。
            self._gate_stats["serve_admission_violations"] += 1
            ctx.last_bounce_side = None
            return []

        # 门控 2：发球前必须有（近期）击球 —— 手抛传球/滚球等无球拍接触的
        # 弹跳序列，或击球证据已经陈旧的序列，直接挡下、候选重来。
        last_hit = ctx.last_hit_time
        if last_hit is None or (t - last_hit) > rules.pre_serve_hit_window_s:
            self._gate_stats["serve_no_hit_blocked"] += 1
            ctx.serve_own_bounce_seen = False
            ctx.last_bounce_side = None
            return []

        # "本方→对方" 序列完成 → 升为可判分回合（归因重置为发球方，干净起步）
        self.state = MatchState.RALLY
        ctx.last_hitter = ctx.server
        ctx.last_bounce_side = side
        ctx.hit_since_last_bounce = False
        ctx.rally_has_hit = False
        return []

    def _on_bounce_serve_legacy(self, side: str, t: float) -> List[Decision]:
        """旧版发球阶段判分（enable_serve_admission=False 时才可达）。"""
        ctx = self._ctx
        rules = self.rules

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
            ctx.last_hitter = ctx.server
            ctx.rally_has_hit = False
            ctx.hit_since_last_bounce = False
            ctx.last_bounce_side = side
        return []

    def _on_bounce_rally(self, side: str, t: float) -> List[Decision]:
        """对打阶段落台：序列误拼守卫（门控 3b）+ 回合内击球要求（门控 3a）。"""
        ctx = self._ctx
        rules = self.rules

        # 门控 3b：同侧两弹之间夹击球 = 序误拼（虚假击球 + 漏检拼接的典型形态）
        if (
            rules.enable_splice_guard
            and ctx.hit_since_last_bounce
            and ctx.last_bounce_side == side
        ):
            self._gate_stats["splice_skips"] += 1
            ctx.last_bounce_side = side
            ctx.hit_since_last_bounce = False
            return []

        # 交替落台防误判：如果上一次落台在对面半台，而本次落台在这一侧，
        # 说明球已成功越网从 last_bounce_side 飞到了 side。
        # 仅当中间没有检测到新的击球时（说明回击挥拍因遮挡未检出）：
        # 此时物理上绝不可能发生 OWN_COURT（球刚刚从对面半台飞过来），击球方自动修正为对面半台的球员。
        if ctx.last_bounce_side is not None and ctx.last_bounce_side != side and not ctx.hit_since_last_bounce:
            if ctx.last_hitter == side:
                ctx.last_hitter = ctx.last_bounce_side

        if ctx.last_hitter is None:
            ctx.last_bounce_side = side
            ctx.hit_since_last_bounce = False
            return []

        # 门控 3a：回合内至少一次已知击球才允许判分
        hit_ok = (not rules.enable_rally_hit_required) or ctx.rally_has_hit

        # 球落在击球方自己半台 → 失误
        if side == ctx.last_hitter:
            if rules.enable_own_court:
                if hit_ok:
                    return self._award(
                        rules.opponent(ctx.last_hitter), t, "OWN_COURT",
                        f"球落在击球方({ctx.last_hitter})自己半台",
                    )
                self._gate_stats["no_hit_skips"] += 1
            ctx.last_bounce_side = side
            ctx.hit_since_last_bounce = False
            return []

        # 同一侧连跳两次（中间无击球）→ 该侧球员没接到
        if side == ctx.last_bounce_side:
            if rules.enable_double_bounce:
                if hit_ok:
                    return self._award(
                        ctx.last_hitter, t, "DOUBLE_BOUNCE",
                        f"球在 {side} 侧连跳两次，{side} 方未接到",
                    )
                self._gate_stats["no_hit_skips"] += 1
        ctx.last_bounce_side = side
        ctx.hit_since_last_bounce = False
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
        rules = self.rules
        if not rules.enable_out:
            return []
        if self.state == MatchState.SERVE_IN_PROGRESS:
            if rules.enable_serve_admission:
                # 门控 1：准入未完成 → 发球阶段不判分，出界证据随候选一起作废
                self._gate_stats["serve_phase_ignored"] += 1
                return []
            return self._award(
                rules.opponent(ctx.server), t, "OUT", "发球出界"
            )
        if self.state in (MatchState.RALLY, MatchState.RALLY_END_CANDIDATE) and ctx.last_hitter:
            hit_ok = (not rules.enable_rally_hit_required) or ctx.rally_has_hit
            if not hit_ok:
                self._gate_stats["no_hit_skips"] += 1
                return []
            return self._award(
                rules.opponent(ctx.last_hitter), t, "OUT",
                f"击球方({ctx.last_hitter})出界",
            )
        return []

    # ------------------------------------------------------------------
    # 回合终了裁决
    # ------------------------------------------------------------------
    def _resolve_rally_end(self, t: float) -> List[Decision]:
        """候选期满：决定这一分归谁，或者承认判不出来。

        门控 4：静默超时**默认只清理证据、不判分**——``NO_RETURN`` 默认关闭，
        且同样受门控 3a（回合内击球要求）约束。
        """
        ctx = self._ctx
        rules = self.rules

        # 简化假设：球合法落在对方半台之后消失 → 对方没接到
        hit_ok = (not rules.enable_rally_hit_required) or ctx.rally_has_hit
        if (
            rules.enable_no_return
            and hit_ok
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

        # 换发立即计算（比分已更新，HUD 可即时显示下一轮发球方）；
        # 回合证据的清理按门控 4 推迟到冷却期满（tick / update 内落地）。
        self._rotate_serve_if_needed()
        if self.rules.score_cooldown_s > 0:
            self._cooldown_until = t + self.rules.score_cooldown_s
        else:
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

    @property
    def cooldown_remaining_s(self) -> float:
        """判分后冷却剩余秒数（不在冷却期时为 0）。"""
        if self._cooldown_until is None or self._last_timestamp is None:
            return 0.0
        return max(0.0, self._cooldown_until - self._last_timestamp)

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
            "gates": dict(self._gate_stats),
        }

    def __repr__(self) -> str:
        return (
            f"<TableTennisStateMachine {self.score_text} state={self.state.value} "
            f"server={self._ctx.server}>"
        )
