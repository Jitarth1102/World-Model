#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from world_model.data.synthetic import make_synthetic_clip
from world_model.memory.oracle_writer import accumulate_clip_into_memory
from world_model.memory.renderer import render_memory_view
from world_model.memory.voxel_grid import VoxelGridSpec
from world_model.types import ClipSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end oracle persistent memory demo.")
    parser.add_argument("--source", choices=["npz", "synthetic"], default="synthetic")
    parser.add_argument("--clip", type=Path, help="Prepared clip path when source=npz.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--target-frame", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--bounds-min", type=float, nargs=3, default=(-2.0, -1.5, -2.0))
    parser.add_argument("--bounds-max", type=float, nargs=3, default=(2.0, 1.8, 2.0))
    parser.add_argument("--splat-radius", type=int, default=1)
    return parser.parse_args()


def load_clip(args: argparse.Namespace) -> ClipSample:
    if args.source == "synthetic":
        return make_synthetic_clip()
    if args.clip is None:
        raise ValueError("--clip is required when --source=npz")
    return ClipSample.load_npz(args.clip)


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def save_strip(frames: list[np.ndarray], path: Path, title: str | None = None) -> None:
    height, width = frames[0].shape[:2]
    canvas = np.zeros((height, width * len(frames), 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        canvas[:, idx * width : (idx + 1) * width] = frame
    Image.fromarray(canvas).save(path)
    if title:
        print(f"{title}: {path}")


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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clip = load_clip(args)

    context_frames = min(args.context_frames, clip.num_frames - 1)
    target_frame = min(max(context_frames, args.target_frame), clip.num_frames - 1)
    spec = VoxelGridSpec(
        bounds_min=tuple(args.bounds_min),
        bounds_max=tuple(args.bounds_max),
        resolution=tuple(args.grid_resolution),
    )

    persistent_memory, write_stats = accumulate_clip_into_memory(
        clip=clip,
        context_frames=context_frames,
        memory_spec=spec,
        stride=max(1, args.stride),
    )
    last_frame_memory, _ = accumulate_clip_into_memory(
        clip=clip,
        context_frames=context_frames,
        memory_spec=spec,
        stride=max(1, args.stride),
    )
    # Keep only the last context write for the baseline.
    last_frame_memory.color_sum.fill(0.0)
    last_frame_memory.weight.fill(0.0)
    last_frame_memory.occupancy.fill(0)
    last_frame_memory.confidence.fill(0.0)
    single_clip = ClipSample(
        video=clip.video[context_frames - 1 : context_frames],
        depth=clip.depth[context_frames - 1 : context_frames],
        poses=clip.poses[context_frames - 1 : context_frames],
        intrinsics=clip.intrinsics,
        segmentations=None if clip.segmentations is None else clip.segmentations[context_frames - 1 : context_frames],
        metadata=clip.metadata,
    )
    last_frame_memory, _ = accumulate_clip_into_memory(single_clip, context_frames=1, memory_spec=spec, stride=max(1, args.stride))

    persistent_render = render_memory_view(
        persistent_memory,
        clip.poses[target_frame],
        clip.intrinsics,
        splat_radius=args.splat_radius,
    )
    last_frame_render = render_memory_view(
        last_frame_memory,
        clip.poses[target_frame],
        clip.intrinsics,
        splat_radius=args.splat_radius,
    )

    gt_rgb = clip.video[target_frame].astype(np.float32) / 255.0
    persistent_error = masked_mae(persistent_render.rgb, gt_rgb, persistent_render.mask)
    last_error = masked_mae(last_frame_render.rgb, gt_rgb, last_frame_render.mask)
    persistent_coverage = float(np.mean(persistent_render.mask))
    last_coverage = float(np.mean(last_frame_render.mask))

    metrics = {
        "context_frames": context_frames,
        "target_frame": target_frame,
        "writes_per_frame": [stat.num_points_written for stat in write_stats],
        "persistent_rgb_mae": persistent_error,
        "last_frame_rgb_mae": last_error,
        "persistent_coverage": persistent_coverage,
        "last_frame_coverage": last_coverage,
    }

    save_strip(
        [clip.video[idx] for idx in range(context_frames)],
        output_dir / "context_strip.png",
        title="context strip",
    )
    Image.fromarray((np.clip(persistent_render.rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output_dir / "target_render.png")
    Image.fromarray((np.clip(last_frame_render.rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(output_dir / "last_frame_render.png")
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

    print(json.dumps(metrics, indent=2))
    print(f"saved demo artifacts to {output_dir}")


if __name__ == "__main__":
    main()
