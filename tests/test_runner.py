import unittest

from run_referee import render_config, should_render_frame


class TestRenderSampling(unittest.TestCase):
    def test_noninteger_ratio_preserves_duration(self):
        count = sum(should_render_frame(fid, 120, 50) for fid in range(1200))
        self.assertEqual(count, 500)
        self.assertEqual(count / 50, 1200 / 120)

    def test_integer_ratio_unchanged(self):
        selected = [fid for fid in range(120) if should_render_frame(fid, 120, 60)]
        self.assertEqual(selected, list(range(0, 120, 2)))

    def test_full_rate(self):
        self.assertTrue(all(should_render_frame(fid, 120, 120) for fid in range(120)))

    def test_render_config(self):
        self.assertEqual(render_config(1920, 1080, 120, 0.5, 50), (960, 540, 50))
        self.assertEqual(render_config(1920, 1080, 120), (1920, 1080, 120))

    def test_invalid_render_parameters(self):
        for scale, fps in [(0, 60), (-1, 60), (float('nan'), 60), (1, 0),
                           (1, -1), (1, float('inf')), (1, 121), (0.00001, 60)]:
            with self.subTest(scale=scale, fps=fps), self.assertRaises(ValueError):
                render_config(1920, 1080, 120, scale, fps)


if __name__ == '__main__':
    unittest.main()
