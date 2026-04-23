from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from world_model.data.clip_dataset import ExportedClipWindowDataset, MemoryConditionedClipWindowDataset, split_clip_paths
from world_model.data.synthetic import make_synthetic_clip


class ExportedClipDatasetTest(unittest.TestCase):
    def test_dataset_builds_windows_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_path = Path(tmpdir) / "clip.npz"
            make_synthetic_clip(num_frames=8, image_size=64).save_npz(clip_path)
            dataset = ExportedClipWindowDataset(
                clip_paths=[clip_path],
                context_frames=4,
                predict_frames=2,
                image_size=64,
            )
            self.assertEqual(len(dataset), 3)
            sample = dataset[0]
            self.assertEqual(tuple(sample["context_rgb"].shape), (4, 3, 64, 64))
            self.assertEqual(tuple(sample["target_rgb"].shape), (2, 3, 64, 64))
            self.assertEqual(tuple(sample["context_poses"].shape), (4, 4, 4))
            self.assertEqual(tuple(sample["target_poses"].shape), (2, 4, 4))

    def test_split_clip_paths_keeps_both_sides(self) -> None:
        clip_paths = [Path(f"{idx:05d}.npz") for idx in range(5)]
        train_paths, val_paths = split_clip_paths(clip_paths, val_ratio=0.2, seed=0)
        self.assertTrue(train_paths)
        self.assertTrue(val_paths)
        self.assertEqual(sorted(train_paths + val_paths), sorted(clip_paths))

    def test_memory_dataset_builds_conditions(self) -> None:
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
            sample = dataset[0]
            self.assertEqual(tuple(sample["memory_condition"].shape), (2, 5, 64, 64))
            self.assertEqual(tuple(sample["target_depth"].shape), (2, 1, 64, 64))
            self.assertGreater(sample["memory_render_coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
