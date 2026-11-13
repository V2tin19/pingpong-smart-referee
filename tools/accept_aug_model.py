"""Gate the augmentation candidate against the production baseline.

Consumes two reports produced by tools/benchmark_input_sizes.py (one per weight
file) and screens recall@5/10/25, conditional center error, and detections on
annotated-invisible frames at a fixed imgsz. This is an observed no-regression
screen on sparse point labels, never a guarantee of equal accuracy or a
deployment approval. Run the benchmark once per weight, then compare here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pick_result(report, imgsz):
    for result in report['results']:
        if result['imgsz'] == imgsz:
            return result
    raise ValueError(f'no result at imgsz={imgsz}; available: '
                     f'{[r["imgsz"] for r in report["results"]]}')


def compare(baseline, candidate, max_recall_drop_pp, max_error_increase_px):
    reasons = []
    rows = []
    for threshold in (5, 10, 25):
        key = f'recall_{threshold}px_pct'
        b, c = baseline[key], candidate[key]
        delta = None if b is None or c is None else c - b
        rows.append((f'recall@{threshold}px (%)', b, c, delta))
        if b is None or c is None:
            reasons.append(f'{key}: missing positive labels')
        elif c + max_recall_drop_pp + 1e-9 < b:
            reasons.append(f'recall@{threshold}px regressed by {b - c:.2f} pp')
    for key, label in (('mean_center_error_px', 'MAE (px)'), ('rmse_px', 'RMSE (px)')):
        b, c = baseline[key], candidate[key]
        delta = None if b is None or c is None else c - b
        rows.append((label, b, c, delta))
    b, c = baseline['mean_center_error_px'], candidate['mean_center_error_px']
    if b is None or c is None:
        reasons.append('missing localization measurements')
    elif c > b + max_error_increase_px + 1e-9:
        reasons.append(f'conditional center error increased by {c - b:.3f} px')
    b_fp = baseline['detections_on_annotated_invisible']
    c_fp = candidate['detections_on_annotated_invisible']
    rows.append(('FP on annotated-invisible', b_fp, c_fp, c_fp - b_fp))
    if c_fp > b_fp:
        reasons.append(f'more detections on annotated-invisible frames ({b_fp} -> {c_fp})')
    return {'gate_pass': not reasons, 'reasons': sorted(set(reasons)), 'rows': rows}


def fmt(value):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', required=True)
    parser.add_argument('--candidate-report', required=True)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--max-recall-drop-pp', type=float, default=0.0)
    parser.add_argument('--max-error-increase-px', type=float, default=0.0)
    parser.add_argument('--output', help='optional JSON verdict path')
    args = parser.parse_args(argv)

    baseline_report = json.loads(Path(args.baseline_report).read_text(encoding='utf-8'))
    candidate_report = json.loads(Path(args.candidate_report).read_text(encoding='utf-8'))
    baseline = pick_result(baseline_report, args.imgsz)['metrics']
    candidate = pick_result(candidate_report, args.imgsz)['metrics']
    verdict = compare(baseline, candidate, args.max_recall_drop_pp, args.max_error_increase_px)

    print(f'imgsz={args.imgsz}  baseline={baseline_report["config"]["weights"]}')
    print(f'              candidate={candidate_report["config"]["weights"]}')
    print(f'{"metric":32} {"baseline":>12} {"candidate":>12} {"delta":>12}')
    for label, b, c, delta in verdict['rows']:
        sign = '' if delta is None else ('+' if delta > 0 else '')
        print(f'{label:32} {fmt(b):>12} {fmt(c):>12} {sign + fmt(delta):>12}')
    print(f'positive frames: baseline={baseline["positive_frames"]} '
          f'candidate={candidate["positive_frames"]}')
    print(f'annotated-invisible frames: baseline={baseline["annotated_invisible_frames"]} '
          f'candidate={candidate["annotated_invisible_frames"]}')
    if verdict['gate_pass']:
        print('GATE: PASS (observed no-regression on this sample)')
    else:
        print('GATE: FAIL')
        for reason in verdict['reasons']:
            print(f'  - {reason}')

    if args.output:
        payload = {'imgsz': args.imgsz,
                   'baseline_weights': baseline_report['config']['weights'],
                   'candidate_weights': candidate_report['config']['weights'],
                   'baseline_metrics': baseline, 'candidate_metrics': candidate,
                   'gate_pass': verdict['gate_pass'], 'reasons': verdict['reasons']}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                       encoding='utf-8')
        print(f'Verdict saved: {out}')
    return 0 if verdict['gate_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())