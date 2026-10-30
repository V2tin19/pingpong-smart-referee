"""
tests/run_end_to_end.py
=======================

端到端串联：**适配器 → 轨迹/事件处理层 → 计分状态机 → 比分 UI 渲染**。

这是组长方案 §1 架构的全链路联调脚本：逐帧喂观测、提取事件、驱动状态机，
并可选渲染带左上角比分板的视频、导出统一观测 JSONL（供跨分支互通校验）。

用法::

    python tests/run_end_to_end.py \
        --video   data/test_2.mp4 \
        --weights weights/yolo11n_clean_best.pt --preset test2 \
        --output  output/output_referee_test2.mp4

对照基线（老脚本 ``smart_referee_yolo.py`` 在同一视频上的输出）：

===================  =========  =========  =========
视频                  落台       击球       违例
===================  =========  =========  =========
test_game_2.avi          24         30          1
test_rally.mp4            8          5          2
===================  =========  =========  =========

注意老脚本把"事件提取"和"计分"揉在一起，且事件由外推点也会触发；
本脚本的事件**只由真实观测点产生**（§7.3），因此计数不要求逐一对齐，
而是看**量级是否一致**、有无爆炸性误报。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.yolo_adapter import YOLOAdapter  # noqa: E402
from core.geometry import guess_preset_name, preset  # noqa: E402
from core.homography import TableHomography  # noqa: E402
from core.schemas import EventType  # noqa: E402
from core.state_machine import DecisionKind, MatchRules, TableTennisStateMachine  # noqa: E402
from core.trajectory_processor import TrajectoryEventProcessor  # noqa: E402
from visualizer.broadcast_hud import BroadcastHUD, HUDState  # noqa: E402

# 老脚本的对照数字
LEGACY_REFERENCE = {
    "test_game_2": {"bounce": 24, "racket_hit": 30, "fault": 1, "fps": 19.7, "coverage": 96.5},
    "test_rally": {"bounce": 8, "racket_hit": 5, "fault": 2, "fps": 24.4, "coverage": 41.7},
    "test_plos": {"bounce": 0, "racket_hit": 0, "fault": 0, "fps": 19.2, "coverage": 28.0},
}


def reference_for(video_path: str) -> Dict[str, float]:
    base = os.path.basename(video_path).lower()
    for key, val in LEGACY_REFERENCE.items():
        if key.replace("test_", "") in base:
            return val
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end: adapter -> trajectory/event processor")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--preset", default=None, help="不指定则按文件名猜（仅脚本便利，非正式链路）")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.18)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-candidates", action="store_true",
                    help="关闭带外多候选通道，退回只用 ball（消融对比）")
    ap.add_argument("--export-observations", default=None,
                    help="把逐帧观测按统一格式导出为 JSONL（供跨分支互通校验）")
    ap.add_argument("--output", default=None,
                    help="渲染带比分 UI 的视频（组长要求：左上角显示比分）")
    ap.add_argument("--no-no-return", action="store_true",
                    help="关闭 NO_RETURN 简化假设（球在对方半台落台后消失不判分）")
    ap.add_argument("--no-serve-fault", action="store_true", help="关闭发球失误判罚")
    ap.add_argument("--no-out", action="store_true", help="关闭出界判罚")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sm_rules = MatchRules(
        enable_no_return=not args.no_no_return,
        enable_serve_fault=not args.no_serve_fault,
        enable_out=not args.no_out,
    )

    preset_name = args.preset or guess_preset_name(args.video)
    calib = preset(preset_name)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return 2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(total, args.max_frames or total)

    adapter = YOLOAdapter(
        args.weights,
        calibration=calib,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        strict_time=True,
    )
    processor = TrajectoryEventProcessor(calib, source="yolo")
    state_machine = TableTennisStateMachine(sm_rules)

    print("=" * 78)
    print(f"Video      : {os.path.basename(args.video)} ({w}x{h} @ {fps:.1f}fps, {n_frames} frames)")
    print(f"Calibration: {preset_name}   ROI={calib.roi_box.to_tuple()}  net_x={calib.net_x}")
    print(f"Adapter    : imgsz={args.imgsz} conf={args.conf} "
          f"candidates={'off' if args.no_candidates else 'on'}")
    print(f"Rules      : no_return={sm_rules.enable_no_return} "
          f"serve_fault={sm_rules.enable_serve_fault} out={sm_rules.enable_out}")
    print("=" * 78)

    counts: Counter = Counter()
    events_log: List = []
    landings: List = []
    t0 = time.time()

    # §5：落点解算的唯一合法入口只接受 BOUNCE 事件
    homography = TableHomography(calib, image_size=(w, h))

    # 可视化（组长要求：左上角显示比分）
    hud = None
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            print(f"ERROR: cannot open video writer for {args.output}")
            return 2
        hud = BroadcastHUD(calib, homography)
    last_decision_text: Optional[str] = None
    last_decision_t = -999.0

    # 统一观测导出（跨分支互通：可直接喂给 TrackNet 交付的 check_observations.py）
    obs_fp = None
    if args.export_observations:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_observations)), exist_ok=True)
        obs_fp = open(args.export_observations, "w", encoding="utf-8")

    for frame_id in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_id / fps
        obs = adapter.predict_frame(frame, frame_id, t)
        if obs_fp is not None:
            obs_fp.write(obs.to_json() + "\n")
        cands = None if args.no_candidates else adapter.candidates_for(frame_id)

        # 事件层 → 状态机
        for decision in state_machine.tick(t):
            last_decision_text, last_decision_t = decision.describe(), t
            if not args.quiet:
                print(f"  [{frame_id:04d}] {t:7.3f}s  >> {decision.describe()}")

        for ev in processor.update(obs, candidates=cands):
            counts[ev.type] += 1
            events_log.append(ev)
            if ev.type == EventType.BOUNCE.value:
                landings.append(homography.locate_landing_from_event(ev))
            if not args.quiet:
                print(f"  [{frame_id:04d}] {t:7.3f}s  {ev.type:<11} side={ev.side:<7} "
                      f"conf={ev.confidence if ev.confidence is None else round(ev.confidence, 2)}  "
                      f"{ev.event_id}")
            for decision in state_machine.update(ev):
                last_decision_text, last_decision_t = decision.describe(), t
                if not args.quiet:
                    print(f"  [{frame_id:04d}] {t:7.3f}s  >> {decision.describe()}")

        if writer is not None:
            current = processor.trajectory[-1] if processor.trajectory else None
            hud_state = HUDState(
                frame_id=frame_id,
                timestamp_s=t,
                fps=fps,
                score=(state_machine.points["A"], state_machine.points["B"]),
                state=state_machine.state.value,
                server=state_machine.server,
                game_over=state_machine.game_over,
                last_decision=last_decision_text,
                decision_age_s=t - last_decision_t,
                track_points=processor.trajectory,
                landings=landings[-hud.landing_history:],
                current_pos=(current.x, current.y) if current is not None else None,
                track_method=(
                    "pred" if (current is not None and not current.is_observed) else "yolo"
                ),
            )
            writer.write(hud.render(frame.copy(), hud_state))

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n[输出] 渲染视频: {args.output}")
    if obs_fp is not None:
        obs_fp.close()
        print(f"\n[导出] 统一观测 JSONL: {args.export_observations}")
    elapsed = time.time() - t0

    st = processor.stats
    print("\n" + "-" * 78)
    print("[适配器]  frames=%d  measured=%d" % (st["frames"], st["measured"]))
    print("[处理器]  confirmed_tracks=%d  rejected_static=%d  smoothed=%d  predicted=%d"
          % (st["confirmed_tracks"], st["rejected_static"], st["smoothed"], st["predicted"]))
    print("[事件]   " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("[性能]   %.2fs  %.1f FPS" % (elapsed, n_frames / max(elapsed, 1e-9)))

    if landings:
        in_tbl = sum(1 for lp in landings if lp.in_table)
        xs = [lp.x_cm for lp in landings]
        ys = [lp.y_cm for lp in landings]
        print("\n[落点解算]  共 %d 个落点（§5：只从 BOUNCE 事件解算）" % len(landings))
        print("  台内 %d / 台外 %d" % (in_tbl, len(landings) - in_tbl))
        print("  物理坐标范围  x_cm=[%.1f, %.1f]  y_cm=[%.1f, %.1f]  (台面 274 x 152.5)"
              % (min(xs), max(xs), min(ys), max(ys)))
        print("  球网物理位置 net_x_cm=%.1f  |  标定反推=%.1f（越接近说明标定越自洽）"
              % (homography.net_x_cm, homography.calibration_net_x_cm()))
        print("  最近边界距离 min=%.1fcm  max=%.1fcm"
              % (min(lp.distance_to_edge_cm for lp in landings),
                 max(lp.distance_to_edge_cm for lp in landings)))

    ref = reference_for(args.video)
    if ref:
        print("\n[对照老脚本 smart_referee_yolo.py]")
        print("  事件          本实现    老脚本")
        print(f"  BOUNCE        {counts.get('BOUNCE', 0):<9}{ref.get('bounce', '-')}")
        print(f"  RACKET_HIT    {counts.get('RACKET_HIT', 0):<9}{ref.get('racket_hit', '-')}")
        print(f"  OUT           {counts.get('OUT', 0):<9}{ref.get('fault', '-')}")
        print(f"  NET_TOUCH     {counts.get('NET_TOUCH', 0)}")
        print(f"  端到端 FPS    {n_frames / max(elapsed, 1e-9):<9.1f}{ref.get('fps', '-')}")
        print("\n  说明：老脚本的事件会被卡尔曼外推点触发（is_pred 参数从未生效），")
        print("        本实现只认真实观测点，故计数不要求逐一对齐。")

    # ---- 计分结果（组长要看的就是这一段） ----
    smd = state_machine.describe()
    print("\n" + "=" * 78)
    print("[计分]  最终比分  P1 %d : %d P2" % (smd["points"]["A"], smd["points"]["B"]))
    print("  规则贡献: " + (", ".join(f"{k}={v}" for k, v in sorted(smd["rule_counts"].items())) or "无"))
    print("  死球(未判分) %d 次   待人工确认 %d 次   判罚总数 %d"
          % (smd["dead_ball"], smd["review"], smd["decisions"]))
    print("  状态机最终状态: %s   发球方: %s" % (smd["state"], smd["server"]))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
