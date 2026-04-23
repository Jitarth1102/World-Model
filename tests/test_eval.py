from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset
from world_model.data.synthetic import make_synthetic_clip
from world_model.eval.benchmark_slices import build_default_slices, collect_window_stats, compute_slice_memberships
from world_model.eval.metrics_memory import memory_covered_l1


class EvalUtilitiesTest(unittest.TestCase):
    def test_memory_covered_l1_matches_masked_pixels(self) -> None:
        prediction = torch.tensor([[[[[0.0, 1.0], [0.0, 1.0]]]]], dtype=torch.float32)
        target = torch.tensor([[[[[1.0, 1.0], [0.0, 0.0]]]]], dtype=torch.float32)
        mask = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]], dtype=torch.float32)
        value = float(memory_covered_l1(prediction, target, mask))
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_collect_window_stats_and_slices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_path = Path(tmpdir) / "clip.npz"
            make_synthetic_clip(num_frames=8, image_size=64).save_npz(clip_path)
            dataset = MemoryConditionedClipWindowDataset(
                clip_paths=[clip_path],
                context_frames=4,
                predict_frames=2,
                image_size=64,
                memory_grid_resolution=(24, 20, 24),
            )
            window_stats = collect_window_stats(dataset)
            self.assertEqual(len(window_stats), len(dataset))
            self.assertGreaterEqual(window_stats[0].motion_fraction, 0.0)
            self.assertGreaterEqual(window_stats[0].memory_render_coverage, 0.0)
            slice_definitions = build_default_slices(window_stats)
            memberships = compute_slice_memberships(window_stats, slice_definitions)
            self.assertIn("all", memberships)
            self.assertEqual(len(memberships["all"]), len(dataset))


if __name__ == "__main__":
    unittest.main()
