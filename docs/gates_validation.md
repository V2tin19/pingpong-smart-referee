# 状态机四条参考门控 · 实现与验证记录

**日期**：2026-11-04
**依据**：学长布置的四条"参考门控" + `test4_evaluation_report.md` 定位的两大误判源
**实现**：`core/state_machine.py`（全部参数化，见 `MatchRules`；决策记录见 `decisions.md` R-022）
**单测**：`tests/test_state_machine.py`（全量 176 项通过）

---

## 一、四条门控的实现方式

| 门控 | 学长原话 | 实现（`MatchRules` 开关） |
|:---|:---|:---|
| 1 发球准入 | 必须"本方台面落台→对方台面落台"才升为可判分回合；同侧再弹视为准备动作并重置候选；候选超时作废 | `enable_serve_admission=True`：`SERVE_IN_PROGRESS` 变为**准入候选**，同侧再弹重置 `serve_own_bounce_seen` 并计数 `serve_prep_resets`；序列不成立不判 SERVE_FAULT（多半是己方第一跳漏检）；超时由 `SERVE_INCOMPLETE` 死球作废 |
| 2 发球前击球校验 | 发球前 1.5s 内必须有击球；手抛传球没有球拍接触，直接挡下 | `pre_serve_hit_window_s=1.5`：升级时刻校验最近一次 `RACKET_HIT` 距今 ≤1.5s，否则 `serve_no_hit_blocked` 挡下、候选重来。结构上准入只能由击球开启（无击球永无候选） |
| 3 回合内击球要求 | 回合内至少一次击球才允许判分；同侧两弹之间夹击球视为序列误拼、不计分 | `enable_rally_hit_required=True`：升级后 `rally_has_hit` 为假时所有判分（OWN_COURT/DOUBLE_BOUNCE/OUT/NO_RETURN）一律放过（`no_hit_skips`）；`enable_splice_guard=True`：`hit_since_last_bounce` 且落台侧与上次相同 → `splice_skips`、不计分 |
| 4 判分后冷却与默认不判分 | 判分后进入 2s 冷却并忽略全部事件；静默超时与视频结束默认只清理证据、不判分 | `score_cooldown_s=2.0`：判分 → `CONFIRM_POINT`（换发即时计算，证据清理推迟），冷却期内 `update()` 忽略全部事件（`cooldown_ignored_events`）；`enable_no_return=False`（默认值由开改关）：静默超时一律 `NO_EVIDENCE` 死球 |

## 二、修复验证（对照 Test4 报告的误判）

### 1. 帧 828 抢跑误判（报告问题二）——已消除

`test_4_50s.mp4` 前 12.5s（1500 帧）复放，同一权重、同一标定：

| | 门控前（报告实测） | 门控后 |
|:---|:---:|:---:|
| 比分 | **0:1**（帧 828 DOUBLE_BOUNCE 抢跑送分） | **0:0** |
| 帧 828 颠球 | 被判己方二跳违规 | `serve_prep_resets=1`，视为准备动作重置 |

### 2. 全 50s（6002 帧）门控全景

| 指标 | 值 |
|:---|:---|
| 比分 | **1:1**（规则贡献 OWN_COURT=2，死球 6 次） |
| `serve_prep_resets` | 2（发球前颠球全部被吸收） |
| `splice_skips` | **4**（"落台→虚假击球→同侧再弹"误拼全部被拦，修复前这类形态会送分） |
| `cooldown_ignored_events` | 7（判分后捡球期间的杂散事件全部忽略） |
| `no_hit_skips` / `serve_admission_violations` | 1 / 1 |

**判罚时刻与官方真值（`events_markup.json`）对照**：

* 官方发球准备期（帧 808~926 颠球）与首个真实回合（帧 1865 起）之间：**零判分**
  （只有 LET 与死球）——修复前在此区间就已送出 0:1；
* 两分分别产生于帧 3758（31.3s，恰在官方真实回合 3575~3785 之内）与 4892（40.8s）。

### 3. 其余视频回归

| 视频 | 事件层（修复前→后） | 比分（修复前→后） | 说明 |
|:---|:---|:---:|:---|
| `test_2.mp4`（连续对拉剪辑） | BOUNCE 39 / HIT 39 → **完全一致** | 1:2 → **0:0** | 感知零变化；比分变化全部来自门控——该片段从回合中段剪出、无发球序列，`serve_admission_violations=18` 全部正确挡下（旧的 1:2 里含误判分） |
| TrackNet 600 帧样例 | — | 0:1 → **0:0** | 帧 111 的 `SERVE_FAULT` 假分被门控 1 挡下（该假分曾印在 README 示例里） |

## 三、取舍与后续

* **召回换精度（有意为之）**：发球 ace / 真实发球失误不再判分；从回合中段切入的
  剪辑视频无法升级、不给分；真实 OWN_COURT（天然呈"同侧两弹夹击球"形态）在
  门控 3b 下几乎不再触发。全部可用开关消融恢复（见 `state_machine_rules.md` §五）。
* 门控挡的是**状态机层**的误判；Test4 报告"问题一"（感知层底噪/瞬移 → 虚假
  RACKET_HIT）仍需感知侧跟进：`conf=0.35` 门限、球台 ROI 掩码、单帧位移硬截断。
  本轮 `splice_skips=4` / `no_hit_skips=1` 说明仍有虚假击球漏进来，靠门控兜住了。
* **没有判分真值**（官方 `events_markup.json` 只有 bounce/net/empty_event 标签），
  "1:1 是否正确"仍无法量化——判分准确率评测依旧依赖自录带标注数据。
