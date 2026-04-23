#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from world_model.data.synthetic import make_synthetic_clip
from world_model.memory.renderer import render_memory_view
from world_model.memory.voxel_grid import VoxelGrid
from world_model.types import ClipSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a target view from an oracle voxel memory.")
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--clip", type=Path, help="Prepared .npz clip path for target pose.")
    parser.add_argument("--source", choices=["npz", "synthetic"], default="npz")
    parser.add_argument("--target-frame", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splat-radius", type=int, default=1)
    return parser.parse_args()


def load_clip(args: argparse.Namespace) -> ClipSample:
    if args.source == "synthetic":
        return make_synthetic_clip()
    if args.clip is None:
        raise ValueError("--clip is required when --source=npz")
    return ClipSample.load_npz(args.clip)


def main() -> None:
    args = parse_args()
    clip = load_clip(args)
    memory = VoxelGrid.load_npz(args.memory)
    target_frame = min(args.target_frame, clip.num_frames - 1)
    rendered = render_memory_view(memory, clip.poses[target_frame], clip.intrinsics, splat_radius=args.splat_radius)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(clip.video[target_frame])
    axes[0].set_title("GT target")
    axes[1].imshow(np.clip(rendered.rgb, 0.0, 1.0))
    axes[1].set_title("Memory render")
    axes[2].imshow(rendered.mask.astype(np.float32), cmap="gray")
    axes[2].set_title("Valid mask")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved render summary to {args.output}")


if __name__ == "__main__":
    main()
