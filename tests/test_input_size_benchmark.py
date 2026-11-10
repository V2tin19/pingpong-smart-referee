"""Pure metric/selection regressions; no torch or real inference required."""
import json
import unittest
from unittest.mock import mock_open, patch

from tools.benchmark_input_sizes import (
    compare_to_baseline, load_annotations, select_frames, summarize,
)


class TestInputSizeMetrics(unittest.TestCase):
    def test_sparse_annotations_do_not_become_negatives(self):
        contents = json.dumps({'1': {'x': 10, 'y': 20}, '3': {'x': -1, 'y': -1}})
        with patch('builtins.open', mock_open(read_data=contents)):
            labels = load_annotations('ball_markup.json')
        self.assertEqual(labels, {1: (10, 20), 3: None})
        self.assertNotIn(2, select_frames(labels, 10))

    def test_sampling_spans_positive_range_and_keeps_invisible(self):
        labels = {i: (1, 2) for i in range(100)}
        labels[101] = None
        selected = select_frames(labels, 102, sample_count=3)
        self.assertEqual(list(selected), [0, 49, 99, 101])

    def test_denominator_is_actual_selected_positives(self):
        rows = [
            {'label': 'visible', 'detected': True, 'error_px': 3, 'adapter_ms': 10},
            {'label': 'visible', 'detected': False, 'error_px': None, 'adapter_ms': 20},
            {'label': 'invisible', 'detected': True, 'error_px': None, 'adapter_ms': 30},
        ]
        result = summarize(rows)
        self.assertEqual(result['recall_5px_pct'], 50)
        self.assertEqual(result['annotated_precision_5px_pct'], 50)
        self.assertEqual(result['detections_on_annotated_invisible'], 1)
        self.assertEqual(result['mean_center_error_px'], 3)
        self.assertEqual(result['adapter_fps'], 50)

    def test_no_detections_error_is_null_not_perfect_zero(self):
        result = summarize([{'label': 'visible', 'detected': False, 'error_px': None, 'adapter_ms': 10}])
        self.assertIsNone(result['mean_center_error_px'])
        self.assertEqual(result['recall_5px_pct'], 0)
        self.assertFalse(compare_to_baseline(result, result)['observed_accuracy_screen_pass'])

    def test_faster_candidate_with_recall_loss_fails_screen(self):
        baseline = summarize([{'label': 'visible', 'detected': True, 'error_px': 1, 'adapter_ms': 20}])
        candidate = summarize([{'label': 'visible', 'detected': True, 'error_px': 6, 'adapter_ms': 10}])
        comparison = compare_to_baseline(baseline, candidate)
        self.assertFalse(comparison['observed_accuracy_screen_pass'])
        self.assertTrue(comparison['faster_than_baseline'])
        self.assertEqual(comparison['speedup'], 2)


class TestLegacyBenchmarkRegression(unittest.TestCase):
    def test_unknown_frames_never_count_as_negative_and_recall_is_in_range(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock
        import numpy as np
        from tools.compare_weights_benchmark import evaluate_model
        model = Mock()
        model.predict.return_value = []
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {3: 1920, 4: 1080, 7: 4}.get(prop, 0)
        cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=model))}), patch(
            'tools.compare_weights_benchmark.cv2.VideoCapture', return_value=cap
        ):
            result = evaluate_model('dummy.pt', 'dummy.mp4', {1: (100, 100), 3: None, 10: (100, 100)}, max_frames=4)
        self.assertEqual(result['total_gt'], 1)
        self.assertEqual(result['gt_frames'], 1)
        self.assertEqual(result['non_gt_frames'], 1)
        import math
        self.assertTrue(math.isnan(result['mae']))
        self.assertTrue(math.isnan(result['rmse']))
        self.assertEqual(model.predict.call_count, 3)  # one warmup + two annotated frames


if __name__ == '__main__':
    unittest.main()
