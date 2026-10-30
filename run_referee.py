"""
run_referee.py
==============

乒乓球视觉智能裁判系统 · 统一运行入口 (Smart Table Tennis Referee Engine)

支持两种输入模式：
  1. 观测流模式 (--jsonl)：读取由前端感知模块（TrackNet / YOLO / 边缘计算盒）生成的统一逐帧观测 JSONL。
     零深度学习依赖，纯 CPU 秒级运行（15,000+ FPS）。
  2. 视频/相机模式 (--video)：使用 YOLO 视觉适配器实时推理视频或摄像头视频流。

命令行示例：
  # 模式 1：从统一观测文件运行（推荐用于跨分支联调与算法验证）
  python run_referee.py --jsonl examples/sample_tracknet_600f.jsonl --preset test2

  # 模式 2：从视频运行并渲染转播级比分 HUD
  python run_referee.py --video data/test_2.mp4 --weights weights/yolo11n_clean_best.pt --preset test2 --output output/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Optional

import cv2

# 将当前根目录加入 sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.geometry import guess_preset_name, preset
from core.homography import TableHomography
from core.schemas import EventType, FrameObservation
from core.state_machine import MatchRules, TableTennisStateMachine
from core.trajectory_processor import TrajectoryEventProcessor
from visualizer.broadcast_hud import BroadcastHUD, HUDState


def run_from_jsonl(
    jsonl_path: str,
    preset_name: str,
    video_path: Optional[str] = None,
    output_path: Optional[str] = None,
    rules: Optional[MatchRules] = None,
) -> int:
    if not os.path.isfile(jsonl_path):
        print(f"[错误] 找不到观测文件: {jsonl_path}")
        return 1

    calib = preset(preset_name)
    homography = TableHomography(calib, image_size=(1920, 1080))
    state_machine = TableTennisStateMachine(rules or MatchRules())

    cap = None
    writer = None
    hud = None
    fps = 120.0
    w, h = 1920, 1080

    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        homography = TableHomography(calib, image_size=(w, h))

    if output_path and cap:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        hud = BroadcastHUD(calib, homography)

    counts = Counter()
    landings = []
    last_decision_text = None
    last_decision_t = -999.0
    processor = None
    obs_total = 0
    t0 = time.time()

    print("=" * 72)
    print("【乒乓球智能裁判】观测流评测模式 (JSONL Stream)")
    print(f"观测文件: {os.path.basename(jsonl_path)}")
    print(f"机位预设: {preset_name} (球台ROI net_x={calib.net_x})")
    print("=" * 72)

    with open(jsonl_path, "r", encoding="utf-8") as f:
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

            # 状态机时钟推进（处理超时等）
            for decision in state_machine.tick(t):
                last_decision_text, last_decision_t = decision.describe(), t
                print(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] >> 判罚: {decision.describe()}")

            # 物理事件提取
            for ev in processor.update(obs):
                counts[ev.type] += 1
                if ev.type == EventType.BOUNCE.value:
                    landings.append(homography.locate_landing_from_event(ev))
                print(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] 事件: {ev.type:<11} 半台: {ev.side:<7} ID: {ev.event_id}")

                # 驱动状态机
                for decision in state_machine.update(ev):
                    last_decision_text, last_decision_t = decision.describe(), t
                    print(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] >> 判罚: {decision.describe()}")

            # 渲染视频 HUD
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
        print(f"\n[输出] 渲染完成视频已保存: {output_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"处理完成！总帧数: {obs_total} 帧 | 耗时: {elapsed:.3f}s ({obs_total/max(elapsed, 1e-4):.1f} FPS)")
    print(f"物理事件统计: {dict(counts)}")
    print(f"最终比分: P1 (半台A) {state_machine.points['A']} : {state_machine.points['B']} P2 (半台B)")
    print("=" * 72)
    return 0


def run_from_video(
    video_path: str,
    weights_path: str,
    preset_name: str,
    imgsz: int = 960,
    conf: float = 0.18,
    output_path: Optional[str] = None,
    export_jsonl: Optional[str] = None,
    max_frames: Optional[int] = None,
    rules: Optional[MatchRules] = None,
) -> int:
    from adapters.yolo_adapter import YOLOAdapter

    if not os.path.isfile(video_path):
        print(f"[错误] 找不到视频文件: {video_path}")
        return 1
    if not os.path.isfile(weights_path):
        print(f"[错误] 找不到权重文件: {weights_path}")
        return 1

    calib = preset(preset_name)
    state_machine = TableTennisStateMachine(rules or MatchRules())
    processor = TrajectoryEventProcessor(calib, source="yolo")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(total_frames, max_frames or total_frames)

    homography = TableHomography(calib, image_size=(w, h))

    adapter = YOLOAdapter(
        weights_path,
        calibration=calib,
        imgsz=imgsz,
        conf_threshold=conf,
        strict_time=True,
    )

    writer = None
    hud = None
    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        hud = BroadcastHUD(calib, homography)

    jsonl_fp = None
    if export_jsonl:
        out_dir = os.path.dirname(os.path.abspath(export_jsonl))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        jsonl_fp = open(export_jsonl, "w", encoding="utf-8")

    counts = Counter()
    landings = []
    last_decision_text = None
    last_decision_t = -999.0
    t0 = time.time()

    print("=" * 72)
    print("【乒乓球智能裁判】实时视觉推理模式 (YOLO Video)")
    print(f"视频文件: {os.path.basename(video_path)} ({w}x{h} @ {fps:.1f}fps, {n_frames} 帧)")
    print(f"机位预设: {preset_name} (球网X像素: {calib.net_x})")
    print(f"模型权重: {os.path.basename(weights_path)}")
    print("=" * 72)

    for frame_id in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_id / fps

        obs = adapter.predict_frame(frame, frame_id, t)
        if jsonl_fp:
            jsonl_fp.write(obs.to_json() + "\n")

        cands = adapter.candidates_for(frame_id)

        # 状态机心跳
        for decision in state_machine.tick(t):
            last_decision_text, last_decision_t = decision.describe(), t
            print(f"  [帧 {frame_id:04d} | {t:6.3f}s] >> 判罚: {decision.describe()}")

        # 轨迹事件提取与状态转移
        for ev in processor.update(obs, candidates=cands):
            counts[ev.type] += 1
            if ev.type == EventType.BOUNCE.value:
                landings.append(homography.locate_landing_from_event(ev))
            print(f"  [帧 {frame_id:04d} | {t:6.3f}s] 事件: {ev.type:<11} 半台: {ev.side:<7} ID: {ev.event_id}")

            for decision in state_machine.update(ev):
                last_decision_text, last_decision_t = decision.describe(), t
                print(f"  [帧 {frame_id:04d} | {t:6.3f}s] >> 判罚: {decision.describe()}")

        # HUD 渲染
        if writer and hud:
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
                track_method="yolo",
            )
            writer.write(hud.render(frame, hud_state))

    cap.release()
    if writer:
        writer.release()
        print(f"\n[输出] 渲染完成视频已保存: {output_path}")
    if jsonl_fp:
        jsonl_fp.close()
        print(f"[输出] 导出逐帧观测已保存: {export_jsonl}")

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"处理完成！处理帧数: {n_frames} | 耗时: {elapsed:.2f}s ({n_frames/max(elapsed, 1e-4):.1f} FPS)")
    print(f"物理事件统计: {dict(counts)}")
    print(f"最终比分: P1 (半台A) {state_machine.points['A']} : {state_machine.points['B']} P2 (半台B)")
    print("=" * 72)
    return 0


def main():
    parser = argparse.ArgumentParser(description="乒乓球视觉智能裁判系统 · 统一入口")
    parser.add_argument("--jsonl", default=None, help="输入统一逐帧观测 JSONL 文件路径")
    parser.add_argument("--video", default=None, help="输入视频或摄像头设备路径")
    parser.add_argument("--weights", default="weights/yolo11n_clean_best.pt", help="YOLO 权重路径 (视频模式必填)")
    parser.add_argument("--preset", default=None, help="球台机位预设 (test2, game2, rally, plos)")
    parser.add_argument("--output", default=None, help="[可选] 渲染带 HUD 比分板的视频保存路径")
    parser.add_argument("--export-jsonl", default=None, help="[可选] 将推理观测导出为 JSONL 文件")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数限制")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO 推理尺寸 (默认: 960)")
    parser.add_argument("--conf", type=float, default=0.18, help="YOLO 检测阈值 (默认: 0.18)")

    args = parser.parse_args()

    if not args.jsonl and not args.video:
        print("[错误] 请至少指定 --jsonl 或 --video 其中一种输入！")
        parser.print_help()
        return 1

    preset_name = args.preset
    if not preset_name:
        ref_name = args.jsonl or args.video or "test2"
        preset_name = guess_preset_name(ref_name)

    if args.jsonl:
        return run_from_jsonl(
            jsonl_path=args.jsonl,
            preset_name=preset_name,
            video_path=args.video,
            output_path=args.output,
        )
    else:
        return run_from_video(
            video_path=args.video,
            weights_path=args.weights,
            preset_name=preset_name,
            imgsz=args.imgsz,
            conf=args.conf,
            output_path=args.output,
            export_jsonl=args.export_jsonl,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    sys.exit(main())
