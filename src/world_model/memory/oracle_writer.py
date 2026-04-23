from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_model.geometry.camera import depth_to_world_points
from world_model.memory.voxel_grid import VoxelGrid, VoxelGridSpec
from world_model.types import ClipSample


@dataclass
class WriteStats:
    num_pixels_considered: int
    num_points_written: int


def write_frame_to_memory(
    memory: VoxelGrid,
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    pose: np.ndarray,
    intrinsics,
    segmentation: np.ndarray | None = None,
    stride: int = 1,
    ignore_background: bool = True,
) -> WriteStats:
    rgb = rgb_frame[::stride, ::stride].astype(np.float32) / 255.0
    depth = depth_frame[::stride, ::stride].astype(np.float32)
    valid = depth > 0.0
    if segmentation is not None and ignore_background:
        valid &= segmentation[::stride, ::stride] > 0

    world_points = depth_to_world_points(depth, pose, intrinsics)
    points = world_points[valid]
    colors = rgb[valid]
    written = memory.splat_rgb(points, colors)
    return WriteStats(num_pixels_considered=int(valid.size), num_points_written=written)


def accumulate_clip_into_memory(
    clip: ClipSample,
    context_frames: int,
    memory_spec: VoxelGridSpec,
    stride: int = 1,
    ignore_background: bool = True,
) -> tuple[VoxelGrid, list[WriteStats]]:
    memory = VoxelGrid(memory_spec)
    stats: list[WriteStats] = []
    for frame_idx in range(context_frames):
        segmentation = None if clip.segmentations is None else clip.segmentations[frame_idx]
        stats.append(
            write_frame_to_memory(
                memory=memory,
                rgb_frame=clip.video[frame_idx],
                depth_frame=clip.depth[frame_idx],
                pose=clip.poses[frame_idx],
                intrinsics=clip.intrinsics,
                segmentation=segmentation,
                stride=stride,
                ignore_background=ignore_background,
            )
        )
    return memory, stats
