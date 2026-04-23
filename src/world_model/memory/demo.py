from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from world_model.memory.oracle_writer import accumulate_clip_into_memory, estimate_memory_spec_from_clip
from world_model.memory.renderer import render_memory_view
from world_model.memory.voxel_grid import VoxelGridSpec
from world_model.types import ClipSample


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def full_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def save_strip(frames: list[np.ndarray], path: Path, title: str | None = None) -> None:
    height, width = frames[0].shape[:2]
    canvas = np.zeros((height, width * len(frames), 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        canvas[:, idx * width : (idx + 1) * width] = frame
    Image.fromarray(canvas).save(path)
    if title:
        print(f"{title}: {path}")


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def plot_summary(
    clip: ClipSample,
    target_frame: int,
    persistent_rgb: np.ndarray,
    persistent_mask: np.ndarray,
    last_rgb: np.ndarray,
    last_mask: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes[0, 0].imshow(clip.video[target_frame])
    axes[0, 0].set_title("GT target")
    axes[0, 1].imshow(np.clip(persistent_rgb, 0.0, 1.0))
    axes[0, 1].set_title("Persistent memory")
    axes[0, 2].imshow(persistent_mask.astype(np.float32), cmap="gray")
    axes[0, 2].set_title("Persistent mask")
    axes[1, 0].imshow(clip.video[target_frame - 1])
    axes[1, 0].set_title("Last context frame")
    axes[1, 1].imshow(np.clip(last_rgb, 0.0, 1.0))
    axes[1, 1].set_title("Last-frame only")
    axes[1, 2].imshow(last_mask.astype(np.float32), cmap="gray")
    axes[1, 2].set_title("Last-frame mask")
    for axis in axes.ravel():
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _single_frame_clip(clip: ClipSample, frame_idx: int) -> ClipSample:
    return ClipSample(
        video=clip.video[frame_idx : frame_idx + 1],
        depth=clip.depth[frame_idx : frame_idx + 1],
        poses=clip.poses[frame_idx : frame_idx + 1],
        intrinsics=clip.intrinsics,
        segmentations=None if clip.segmentations is None else clip.segmentations[frame_idx : frame_idx + 1],
        visibility=clip.visibility,
        metadata=clip.metadata,
    )


def run_oracle_memory_demo(
    clip: ClipSample,
    output_dir: Path,
    context_frames: int = 4,
    target_frame: int = 5,
    stride: int = 1,
    grid_resolution: tuple[int, int, int] = (48, 40, 48),
    bounds_min: tuple[float, float, float] | None = None,
    bounds_max: tuple[float, float, float] | None = None,
    auto_bounds: bool = True,
    splat_radius: int = 1,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_frames = min(context_frames, clip.num_frames - 1)
    target_frame = min(max(context_frames, target_frame), clip.num_frames - 1)

    if auto_bounds and (bounds_min is None or bounds_max is None):
        memory_spec = estimate_memory_spec_from_clip(
            clip=clip,
            context_frames=context_frames,
            resolution=grid_resolution,
            stride=max(1, stride),
        )
    else:
        if bounds_min is None or bounds_max is None:
            raise ValueError("bounds_min and bounds_max are required when auto_bounds is disabled")
        memory_spec = VoxelGridSpec(bounds_min=bounds_min, bounds_max=bounds_max, resolution=grid_resolution)

    persistent_memory, write_stats = accumulate_clip_into_memory(
        clip=clip,
        context_frames=context_frames,
        memory_spec=memory_spec,
        stride=max(1, stride),
    )
    last_frame_memory, _ = accumulate_clip_into_memory(
        clip=_single_frame_clip(clip, context_frames - 1),
        context_frames=1,
        memory_spec=memory_spec,
        stride=max(1, stride),
    )

    persistent_render = render_memory_view(
        persistent_memory,
        clip.poses[target_frame],
        clip.intrinsics,
        splat_radius=splat_radius,
    )
    last_frame_render = render_memory_view(
        last_frame_memory,
        clip.poses[target_frame],
        clip.intrinsics,
        splat_radius=splat_radius,
    )

    gt_rgb = clip.video[target_frame].astype(np.float32) / 255.0
    persistent_error_masked = masked_mae(persistent_render.rgb, gt_rgb, persistent_render.mask)
    last_error_masked = masked_mae(last_frame_render.rgb, gt_rgb, last_frame_render.mask)
    persistent_error_full = full_mae(persistent_render.rgb, gt_rgb)
    last_error_full = full_mae(last_frame_render.rgb, gt_rgb)
    persistent_coverage = float(np.mean(persistent_render.mask))
    last_coverage = float(np.mean(last_frame_render.mask))

    metrics: dict[str, object] = {
        "context_frames": context_frames,
        "target_frame": target_frame,
        "writes_per_frame": [stat.num_points_written for stat in write_stats],
        "persistent_rgb_mae_masked": persistent_error_masked,
        "last_frame_rgb_mae_masked": last_error_masked,
        "persistent_rgb_mae_full": persistent_error_full,
        "last_frame_rgb_mae_full": last_error_full,
        "persistent_coverage": persistent_coverage,
        "last_frame_coverage": last_coverage,
        "persistent_beats_last_on_full_mae": persistent_error_full < last_error_full,
        "persistent_beats_last_on_coverage": persistent_coverage > last_coverage,
        "bounds_min": list(memory_spec.bounds_min),
        "bounds_max": list(memory_spec.bounds_max),
        "grid_resolution": list(memory_spec.resolution),
    }

    save_strip(
        [clip.video[idx] for idx in range(context_frames)],
        output_dir / "context_strip.png",
        title="context strip",
    )
    Image.fromarray(clip.video[target_frame]).save(output_dir / "target_gt.png")
    save_rgb(persistent_render.rgb, output_dir / "target_render.png")
    save_rgb(last_frame_render.rgb, output_dir / "last_frame_render.png")
    Image.fromarray((persistent_render.mask.astype(np.uint8) * 255)).save(output_dir / "persistent_mask.png")
    Image.fromarray((last_frame_render.mask.astype(np.uint8) * 255)).save(output_dir / "last_frame_mask.png")
    plot_summary(
        clip=clip,
        target_frame=target_frame,
        persistent_rgb=persistent_render.rgb,
        persistent_mask=persistent_render.mask,
        last_rgb=last_frame_render.rgb,
        last_mask=last_frame_render.mask,
        output_path=output_dir / "summary.png",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    persistent_memory.save_npz(output_dir / "persistent_memory.npz")
    return metrics
