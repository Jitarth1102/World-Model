from __future__ import annotations

import unittest

import numpy as np

from world_model.data.synthetic import make_synthetic_clip
from world_model.memory.oracle_writer import accumulate_clip_into_memory
from world_model.memory.renderer import render_memory_view
from world_model.memory.voxel_grid import VoxelGridSpec
from world_model.types import ClipSample


class OracleMemoryTest(unittest.TestCase):
    def test_persistent_memory_has_coverage(self) -> None:
        clip = make_synthetic_clip(num_frames=6, image_size=96)
        spec = VoxelGridSpec(bounds_min=(-2.0, -1.5, -2.0), bounds_max=(2.0, 1.8, 2.0), resolution=(48, 40, 48))
        memory, _ = accumulate_clip_into_memory(clip, context_frames=4, memory_spec=spec, stride=1)
        rendered = render_memory_view(memory, clip.poses[5], clip.intrinsics, splat_radius=1)
        self.assertGreater(float(np.mean(rendered.mask)), 0.02)

    def test_persistent_memory_beats_single_frame_coverage(self) -> None:
        clip = make_synthetic_clip(num_frames=6, image_size=96)
        spec = VoxelGridSpec(bounds_min=(-2.0, -1.5, -2.0), bounds_max=(2.0, 1.8, 2.0), resolution=(48, 40, 48))
        persistent_memory, _ = accumulate_clip_into_memory(clip, context_frames=4, memory_spec=spec, stride=1)
        persistent_render = render_memory_view(persistent_memory, clip.poses[5], clip.intrinsics, splat_radius=1)

        single_clip = ClipSample(
            video=clip.video[3:4],
            depth=clip.depth[3:4],
            poses=clip.poses[3:4],
            intrinsics=clip.intrinsics,
            segmentations=None if clip.segmentations is None else clip.segmentations[3:4],
            metadata=clip.metadata,
        )
        last_frame_memory, _ = accumulate_clip_into_memory(single_clip, context_frames=1, memory_spec=spec, stride=1)
        last_frame_render = render_memory_view(last_frame_memory, clip.poses[5], clip.intrinsics, splat_radius=1)

        self.assertGreaterEqual(float(np.mean(persistent_render.mask)), float(np.mean(last_frame_render.mask)))


if __name__ == "__main__":
    unittest.main()
