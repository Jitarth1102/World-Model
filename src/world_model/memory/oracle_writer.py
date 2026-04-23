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


def collect_context_points(
    clip: ClipSample,
    context_frames: int,
    stride: int = 1,
    ignore_background: bool = True,
) -> np.ndarray:
    point_batches: list[np.ndarray] = []
    for frame_idx in range(context_frames):
        depth = clip.depth[frame_idx, ::stride, ::stride].astype(np.float32)
        valid = depth > 0.0
        if clip.segmentations is not None and ignore_background:
            valid &= clip.segmentations[frame_idx, ::stride, ::stride] > 0
        world_points = depth_to_world_points(depth, clip.poses[frame_idx], clip.intrinsics)
        if np.any(valid):
            point_batches.append(world_points[valid])
    if not point_batches:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(point_batches, axis=0)


def estimate_memory_spec_from_clip(
    clip: ClipSample,
    context_frames: int,
    resolution: tuple[int, int, int],
    stride: int = 1,
    ignore_background: bool = True,
    margin_fraction: float = 0.1,
    min_margin: float = 0.5,
) -> VoxelGridSpec:
    points = collect_context_points(
        clip=clip,
        context_frames=context_frames,
        stride=stride,
        ignore_background=ignore_background,
    )
    if len(points) == 0:
        return VoxelGridSpec(bounds_min=(-2.0, -2.0, -2.0), bounds_max=(2.0, 2.0, 2.0), resolution=resolution)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    extent = np.maximum(bounds_max - bounds_min, 1e-3)
    margin = np.maximum(extent * margin_fraction, min_margin)
    return VoxelGridSpec(
        bounds_min=tuple((bounds_min - margin).tolist()),
        bounds_max=tuple((bounds_max + margin).tolist()),
        resolution=resolution,
    )


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
