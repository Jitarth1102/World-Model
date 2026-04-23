from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_model.memory.oracle_writer import accumulate_clip_into_memory, estimate_memory_spec_from_clip
from world_model.memory.renderer import render_memory_view
from world_model.types import ClipSample


@dataclass(frozen=True)
class MemoryConditionSequence:
    rgb: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    occupancy_fraction: float
    render_coverage: float
    render_rgb_l1_covered: float


def _depth_scale_from_clip(clip: ClipSample) -> float:
    depth_range = clip.metadata.get("depth_range")
    if isinstance(depth_range, (list, tuple)) and len(depth_range) >= 2:
        return max(float(depth_range[1]), 1e-6)
    max_depth = float(np.max(clip.depth)) if clip.depth.size else 1.0
    return max(max_depth, 1e-6)


def build_memory_condition_sequence(
    clip: ClipSample,
    start_frame: int,
    context_frames: int,
    predict_frames: int,
    grid_resolution: tuple[int, int, int] = (48, 40, 48),
    stride: int = 1,
    ignore_background: bool = True,
    splat_radius: int = 1,
) -> MemoryConditionSequence:
    target_start = start_frame + context_frames
    target_end = min(target_start + predict_frames, clip.num_frames)
    if target_start >= target_end:
        raise ValueError("Requested memory sequence has no target frames.")

    memory_spec = estimate_memory_spec_from_clip(
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
        start_frame=start_frame,
        memory_spec=memory_spec,
        stride=stride,
        ignore_background=ignore_background,
    )

    depth_scale = _depth_scale_from_clip(clip)
    target_rgb = clip.video[target_start:target_end].astype(np.float32) / 255.0
    render_rgbs: list[np.ndarray] = []
    render_masks: list[np.ndarray] = []
    render_depths: list[np.ndarray] = []
    covered_errors: list[float] = []

    for frame_idx in range(target_start, target_end):
        rendered = render_memory_view(
            memory=memory,
            target_pose=clip.poses[frame_idx],
            intrinsics=clip.intrinsics,
            splat_radius=splat_radius,
        )
        render_rgbs.append(rendered.rgb.astype(np.float32))
        render_masks.append(rendered.mask)
        render_depths.append(np.where(rendered.mask, rendered.radial_depth / depth_scale, 0.0).astype(np.float32))
        if np.any(rendered.mask):
            target_frame_rgb = target_rgb[frame_idx - target_start]
            covered_errors.append(float(np.mean(np.abs(rendered.rgb[rendered.mask] - target_frame_rgb[rendered.mask]))))

    mask_array = np.stack(render_masks, axis=0)
    return MemoryConditionSequence(
        rgb=np.stack(render_rgbs, axis=0),
        mask=mask_array,
        depth=np.stack(render_depths, axis=0),
        occupancy_fraction=float(memory.occupied_mask().mean()),
        render_coverage=float(mask_array.mean()),
        render_rgb_l1_covered=float(np.mean(covered_errors)) if covered_errors else float("inf"),
    )
