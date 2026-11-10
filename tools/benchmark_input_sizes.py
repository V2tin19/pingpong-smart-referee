"""Compare YOLO input sizes through the production adapter, on sparse point labels.

Unlisted frames are UNKNOWN, never negative examples. Negative coordinates mean
annotated invisible/unlocalizable, not proof of physical absence. Timing covers
adapter inference + conversion, excludes decoding/HUD, and is NOT live-camera FPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import zipfile
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.yolo_adapter import YOLOAdapter
from core.geometry import guess_preset_name, preset

Point = Optional[Tuple[float, float]]


def load_annotations(path: str) -> Dict[int, Point]:
    """Preserve annotated invisible frames; absent keys remain unknown."""
    if str(path).lower().endswith('.zip'):
        with zipfile.ZipFile(path) as archive:
            raw = json.loads(archive.read('ball_markup.json').decode('utf-8'))
    else:
        with open(path, encoding='utf-8') as handle:
            raw = json.load(handle)
    result = {}
    for key, value in raw.items():
        frame_id = int(key)
        if frame_id < 0:
            raise ValueError('Annotation frame IDs must be nonnegative')
        if not value:
            result[frame_id] = None
            continue
        x, y = float(value['x']), float(value['y'])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f'Nonfinite annotation at frame {frame_id}')
        result[frame_id] = (x, y) if x >= 0 and y >= 0 else None
    return result


def select_frames(annotations, frame_count, max_frames=0, sample_count=0):
    """Deterministic spread over positive labels; retain all explicit negatives."""
    limit = min(frame_count, max_frames) if max_frames else frame_count
    available = {fid: point for fid, point in annotations.items() if fid < limit}
    positives = sorted(fid for fid, point in available.items() if point is not None)
    if sample_count and len(positives) > sample_count:
        indices = np.linspace(0, len(positives) - 1, sample_count, dtype=int)
        positives = [positives[i] for i in indices]
    selected = set(positives) | {fid for fid, point in available.items() if point is None}
    return {fid: available[fid] for fid in sorted(selected)}


def summarize(rows):
    positives = [row for row in rows if row['label'] == 'visible']
    negatives = [row for row in rows if row['label'] == 'invisible']
    errors = [row['error_px'] for row in positives if row['error_px'] is not None]
    times = [row['adapter_ms'] for row in rows]
    detections = sum(row['detected'] for row in rows)
    result = {
        'evaluated_frames': len(rows), 'positive_frames': len(positives),
        'annotated_invisible_frames': len(negatives),
        'detected_positive_frames': len(errors),
        'detections_on_annotated_invisible': sum(row['detected'] for row in negatives),
        'mean_center_error_px': float(np.mean(errors)) if errors else None,
        'rmse_px': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        'adapter_mean_ms': float(np.mean(times)) if times else None,
        'adapter_p50_ms': float(np.percentile(times, 50)) if times else None,
        'adapter_p95_ms': float(np.percentile(times, 95)) if times else None,
    }
    result['adapter_fps'] = 1000 / result['adapter_mean_ms'] if times and result['adapter_mean_ms'] > 0 else None
    for threshold in (5, 10, 25):
        hits = sum(error <= threshold for error in errors)
        result[f'hits_{threshold}px'] = hits
        result[f'recall_{threshold}px_pct'] = 100 * hits / len(positives) if positives else None
        # Precision is defined ONLY over evaluated annotated frames.
        result[f'annotated_precision_{threshold}px_pct'] = 100 * hits / detections if detections else None
    return result


def compare_to_baseline(baseline, candidate, max_recall_drop_pp=0.0, max_error_increase_px=0.0):
    """An observed no-regression screen, never a guarantee of equal accuracy."""
    reasons = []
    for threshold in (5, 10, 25):
        key = f'recall_{threshold}px_pct'
        if baseline[key] is None or candidate[key] is None:
            reasons.append('missing positive labels')
        elif candidate[key] + max_recall_drop_pp + 1e-9 < baseline[key]:
            reasons.append(f'recall@{threshold}px regressed')
    b_error, c_error = baseline['mean_center_error_px'], candidate['mean_center_error_px']
    if b_error is None or c_error is None:
        reasons.append('missing localization measurements')
    elif c_error > b_error + max_error_increase_px + 1e-9:
        reasons.append('conditional center error increased')
    if candidate['detections_on_annotated_invisible'] > baseline['detections_on_annotated_invisible']:
        reasons.append('more detections on annotated invisible frames')
    faster = candidate['adapter_mean_ms'] < baseline['adapter_mean_ms']
    return {'observed_accuracy_screen_pass': not reasons,
            'faster_than_baseline': faster, 'reasons': sorted(set(reasons)),
            'speedup': baseline['adapter_mean_ms'] / candidate['adapter_mean_ms']}


def evaluate(args, size, selected, video_fps):
    adapter = YOLOAdapter(args.weights, calibration=preset(args.preset), imgsz=size,
                          conf_threshold=args.conf, iou_threshold=args.iou,
                          device=args.device, half=args.half, strict_time=True)
    cap = cv2.VideoCapture(args.video)
    rows = []
    cold_ms = None
    warmup_total_ms = None
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {args.video}')
        last_id = max(selected)
        for fid in range(last_id + 1):
            if not cap.grab():
                raise RuntimeError(f'Video ended before expected frame {fid}; incomplete evaluation rejected')
            if fid not in selected:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f'Cannot decode selected frame {fid}')
            if cold_ms is None:
                start = time.perf_counter()
                adapter.predict_frame(frame, fid, fid / video_fps)
                cold_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                for _ in range(args.warmup):
                    adapter.predict_frame(frame, fid, fid / video_fps)
                warmup_total_ms = (time.perf_counter() - start) * 1000
                adapter.reset()
            start = time.perf_counter()
            obs = adapter.predict_frame(frame, fid, fid / video_fps)
            duration_ms = (time.perf_counter() - start) * 1000
            point = selected[fid]
            prediction = obs.ball.center_px if obs.ball.visible else None
            error = math.dist(prediction, point) if prediction is not None and point is not None else None
            rows.append({'frame_id': fid, 'label': 'visible' if point is not None else 'invisible',
                         'detected': prediction is not None, 'error_px': error,
                         'center_px': prediction, 'adapter_ms': duration_ms})
        return {'imgsz': size, 'cold_first_call_ms': cold_ms, 'warmup_total_ms': warmup_total_ms,
                'adapter': adapter.describe(), 'crop_size': adapter.last_crop_size,
                'metrics': summarize(rows), 'frames': rows}
    finally:
        cap.release()
        adapter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--annotations', required=True, help='ball_markup.json or ZIP containing it')
    parser.add_argument('--weights', default='weights/yolo11n_game1_best.pt')
    parser.add_argument('--preset')
    parser.add_argument('--sizes', nargs='+', type=int, default=[960, 832, 768, 640, 512])
    parser.add_argument('--baseline', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.18)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', default=None)
    parser.add_argument('--half', action='store_true', help='FP16; compare separately from FP32 size sweep')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--max-frames', type=int, default=0, help='0 = complete video')
    parser.add_argument('--sample-count', type=int, default=0, help='0 = all positive labels; otherwise deterministic smoke subset')
    parser.add_argument('--max-recall-drop-pp', type=float, default=0.0)
    parser.add_argument('--max-error-increase-px', type=float, default=0.0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.baseline not in args.sizes:
        parser.error('--baseline must be included in --sizes')
    if any(size <= 0 or size % 32 for size in args.sizes):
        parser.error('sizes must be positive multiples of 32 (avoid silent stride rounding)')
    if len(set(args.sizes)) != len(args.sizes):
        parser.error('sizes must be unique')
    if min(args.max_frames, args.sample_count, args.warmup, args.max_recall_drop_pp, args.max_error_increase_px) < 0:
        parser.error('frame limits, warmup, and regression tolerances must be nonnegative')
    if Path(args.output).resolve() in {Path(p).resolve() for p in (args.video, args.annotations, args.weights)}:
        parser.error('--output must not overwrite an input')
    args.preset = args.preset or guess_preset_name(args.video)
    annotations = load_annotations(args.annotations)
    cap = cv2.VideoCapture(args.video)
    try:
        if not cap.isOpened():
            parser.error('video cannot be opened')
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        image_size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0:
        parser.error('video has invalid FPS')
    selected = select_frames(annotations, frame_count, args.max_frames, args.sample_count)
    if not selected or not any(point is not None for point in selected.values()):
        parser.error('no positive annotated frames in selected video interval')
    # Heavy dependencies are loaded only when actually benchmarking, not for --help/unit tests.
    import torch
    import ultralytics
    report = {
        'schema_version': 1, 'config': vars(args),
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                        'opencv': cv2.__version__, 'cuda_available': torch.cuda.is_available(),
                        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'weights_sha256': hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        'video': {'frame_count': frame_count, 'fps': fps, 'image_size': image_size},
        'selected_frame_ids': list(selected),
        'limitations': ['Unlisted frames excluded; invisible annotations are not certified no-ball negatives.',
                       'Sparse-frame adapter timing excludes decode, tracking, HUD, and camera queue latency.',
                       'Top-confidence candidate evaluated; not temporal association or rally adjudication.',
                       'No observed regression on this sample is not statistical equivalence or deployment approval.'],
        'results': [],
    }
    for size in args.sizes:
        print(f'Evaluating imgsz={size}, {len(selected)} annotated frames...', flush=True)
        result = evaluate(args, size, selected, fps)
        report['results'].append(result)
        print(json.dumps(result['metrics'], ensure_ascii=False), flush=True)
    baseline = next(result for result in report['results'] if result['imgsz'] == args.baseline)
    for result in report['results']:
        result['comparison'] = compare_to_baseline(baseline['metrics'], result['metrics'],
                                                    args.max_recall_drop_pp, args.max_error_increase_px)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Report saved: {output}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
