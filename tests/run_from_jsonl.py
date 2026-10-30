"""
tests/run_from_jsonl.py
=======================

从导出的统一观测 JSONL 文件（如 TrackNet 产出的 sample_test2_video_600f.jsonl）
直接驱动【轨迹事件层 + 计分状态机】，并可选渲染带比分板的视频。

用法示例：
    python tests/run_from_jsonl.py \
        --jsonl "E:/2026Autumn/602Lab/TrackNET适配器/examples/sample_test2_video_600f.jsonl" \
        --preset test2 \
        --video data/test_2.mp4 \
        --output output/tracknet_referee_demo.mp4
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import preset
from core.homography import TableHomography
from core.schemas import EventType, FrameObservation
from core.state_machine import MatchRules, TableTennisStateMachine
from core.trajectory_processor import TrajectoryEventProcessor
from visualizer.broadcast_hud import BroadcastHUD, HUDState


def main():
    parser = argparse.ArgumentParser(description="Run trajectory & state machine from JSONL observations")
    parser.add_argument("--jsonl", required=True, help="统一观测 JSONL 文件路径")
    parser.add_argument("--preset", default="test2", help="球台机位预设 (如 test2 / game2)")
    parser.add_argument("--video", default=None, help="可选：原始视频路径（用于渲染 HUD 比分板）")
    parser.add_argument("--output", default=None, help="可选：输出渲染视频路径")
    args = parser.parse_args()

    if not os.path.isfile(args.jsonl):
        print(f"Error: JSONL file not found: {args.jsonl}")
        return 1

    calib = preset(args.preset)
    homography = TableHomography(calib, image_size=(1920, 1080))
    sm_rules = MatchRules()
    state_machine = TableTennisStateMachine(sm_rules)

    # 视频与渲染准备
    cap = None
    writer = None
    hud = None
    fps = 120.0
    w, h = 1920, 1080

    if args.video and os.path.isfile(args.video):
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        homography = TableHomography(calib, image_size=(w, h))

    if args.output and cap:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        hud = BroadcastHUD(calib, homography)

    counts = Counter()
    landings = []
    last_decision_text = None
    last_decision_t = -999.0
    processor = None

    print("=" * 70)
    print(f"输入观测文件: {os.path.basename(args.jsonl)}")
    print(f"球台几何预设: {args.preset} (net_x={calib.net_x})")
    print("=" * 70)

    obs_total = 0
    t0 = time.time()

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            obs = FrameObservation.from_dict(data)
            obs_total += 1

            if processor is None:
                source_str = obs.source.value if hasattr(obs.source, "value") else str(obs.source)
                processor = TrajectoryEventProcessor(calib, source=source_str)

            t = obs.timestamp_s

            # 状态机心跳
            for decision in state_machine.tick(t):
                last_decision_text, last_decision_t = decision.describe(), t
                print(f"  [{obs.frame_id:04d}] {t:7.3f}s  >> {decision.describe()}")

            # 轨迹事件提取与状态机推进
            for ev in processor.update(obs):
                counts[ev.type] += 1
                if ev.type == EventType.BOUNCE.value:
                    landings.append(homography.locate_landing_from_event(ev))
                print(f"  [{obs.frame_id:04d}] {t:7.3f}s  {ev.type:<11} side={ev.side:<7} id={ev.event_id}")
                for decision in state_machine.update(ev):
                    last_decision_text, last_decision_t = decision.describe(), t
                    print(f"  [{obs.frame_id:04d}] {t:7.3f}s  >> {decision.describe()}")

            # 视频渲染
            if cap and writer:
                ok, frame = cap.read()
                if ok:
                    current = processor.trajectory[-1] if processor.trajectory else None
                    hud_state = HUDState(
                        frame_id=obs.frame_id,
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
                        track_method="tracknet",
                    )
                    writer.write(hud.render(frame, hud_state))

    if cap:
        cap.release()
    if writer:
        writer.release()
        print(f"\n[输出] 渲染完成视频保存在: {args.output}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"处理完成！总观测帧数: {obs_total}  耗时: {elapsed:.2f}s ({obs_total/elapsed:.1f} FPS)")
    print(f"提取事件统计: {dict(counts)}")
    print(f"最终比分: P1 (半台A) {state_machine.points['A']} : {state_machine.points['B']} P2 (半台B)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
