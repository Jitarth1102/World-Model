#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from world_model.data.synthetic import make_synthetic_clip
from world_model.memory.oracle_writer import accumulate_clip_into_memory
from world_model.memory.voxel_grid import VoxelGridSpec
from world_model.types import ClipSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an oracle persistent voxel memory from a clip.")
    parser.add_argument("--clip", type=Path, help="Prepared .npz clip path.")
    parser.add_argument("--source", choices=["npz", "synthetic"], default="npz")
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz voxel memory path.")
    parser.add_argument("--grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--bounds-min", type=float, nargs=3, default=(-2.0, -1.5, -2.0))
    parser.add_argument("--bounds-max", type=float, nargs=3, default=(2.0, 1.8, 2.0))
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
    spec = VoxelGridSpec(
        bounds_min=tuple(args.bounds_min),
        bounds_max=tuple(args.bounds_max),
        resolution=tuple(args.grid_resolution),
    )
    memory, stats = accumulate_clip_into_memory(
        clip=clip,
        context_frames=min(args.context_frames, clip.num_frames - 1),
        memory_spec=spec,
        stride=max(1, args.stride),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    memory.save_npz(args.output)
    for frame_idx, frame_stats in enumerate(stats):
        print(f"context frame {frame_idx}: wrote {frame_stats.num_points_written} points")
    print(f"saved memory to {args.output}")


if __name__ == "__main__":
    main()
