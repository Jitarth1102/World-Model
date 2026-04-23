from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from world_model.geometry.camera import depth_to_world_points
from world_model.memory.oracle_writer import accumulate_clip_into_memory, estimate_memory_spec_from_clip
from world_model.memory.renderer import render_memory_view
from world_model.memory.voxel_grid import VoxelGrid
from world_model.types import CameraIntrinsics, ClipSample
from world_model.uncertainty.confidence import confidence_to_write_mask


@dataclass(frozen=True)
class GeneratedWriteStats:
    valid_fraction: float
    write_fraction: float
    mean_confidence: float
    mean_written_confidence: float
    num_points_written: int


def depth_scale_from_clip(clip: ClipSample) -> float:
    depth_range = clip.metadata.get("depth_range")
    if isinstance(depth_range, (list, tuple)) and len(depth_range) >= 2:
        return max(float(depth_range[1]), 1e-6)
    max_depth = float(np.max(clip.depth)) if clip.depth.size else 1.0
    return max(max_depth, 1e-6)


def scale_intrinsics(intrinsics: CameraIntrinsics, target_width: int, target_height: int) -> CameraIntrinsics:
    scale_x = target_width / intrinsics.width
    scale_y = target_height / intrinsics.height
    return CameraIntrinsics(
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=(intrinsics.cx + 0.5) * scale_x - 0.5,
        cy=(intrinsics.cy + 0.5) * scale_y - 0.5,
        width=target_width,
        height=target_height,
        depth_is_radial=intrinsics.depth_is_radial,
    )


def build_initial_memory_from_clip(
    clip: ClipSample,
    *,
    start_frame: int,
    context_frames: int,
    grid_resolution: tuple[int, int, int],
    stride: int,
    ignore_background: bool = True,
) -> VoxelGrid:
    spec = estimate_memory_spec_from_clip(
        clip=clip,
        context_frames=context_frames,
        start_frame=start_frame,
        resolution=grid_resolution,
        stride=stride,
        ignore_background=ignore_background,
    )
    memory, _ = accumulate_clip_into_memory(
        clip=clip,
        context_frames=context_frames,
        memory_spec=spec,
        start_frame=start_frame,
        stride=stride,
        ignore_background=ignore_background,
    )
    return memory


def render_memory_condition(
    memory: VoxelGrid,
    target_pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    splat_radius: int,
) -> dict[str, np.ndarray]:
    rendered = render_memory_view(
        memory=memory,
        target_pose=target_pose,
        intrinsics=intrinsics,
        splat_radius=splat_radius,
    )
    mask = rendered.mask.astype(np.float32)
    depth = np.where(rendered.mask, rendered.radial_depth / max(depth_scale, 1e-6), 0.0).astype(np.float32)
    condition = np.concatenate(
        [
            rendered.rgb.astype(np.float32).transpose(2, 0, 1),
            depth[None, ...],
            mask[None, ...],
        ],
        axis=0,
    )
    return {
        "rgb": rendered.rgb.astype(np.float32),
        "mask": mask,
        "depth": depth,
        "condition": condition,
        "coverage": float(mask.mean()),
    }


def write_prediction_to_memory(
    memory: VoxelGrid,
    *,
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    confidence_map: np.ndarray,
    confidence_threshold: float | None = None,
    stride: int = 1,
    min_depth: float = 1e-4,
) -> tuple[np.ndarray, GeneratedWriteStats]:
    rgb = rgb_frame[::stride, ::stride].astype(np.float32)
    depth = depth_frame[::stride, ::stride].astype(np.float32)
    confidence = confidence_map[::stride, ::stride].astype(np.float32)
    valid = np.isfinite(depth) & (depth > min_depth)
    mask = confidence_to_write_mask(torch.from_numpy(confidence), threshold=confidence_threshold).numpy().astype(np.float32)
    write_mask = valid & (mask > 0.0)
    points = depth_to_world_points(depth, pose, intrinsics)
    weights = confidence[write_mask]
    num_points_written = memory.splat_rgb(points[write_mask], rgb[write_mask], weights=weights) if np.any(write_mask) else 0
    stats = GeneratedWriteStats(
        valid_fraction=float(valid.mean()),
        write_fraction=float(write_mask.mean()),
        mean_confidence=float(confidence.mean()),
        mean_written_confidence=float(weights.mean()) if weights.size else 0.0,
        num_points_written=int(num_points_written),
    )
    return write_mask.astype(np.float32), stats
