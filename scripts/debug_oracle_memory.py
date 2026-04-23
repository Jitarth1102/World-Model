#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model.data.synthetic import make_synthetic_clip
from world_model.memory.demo import run_oracle_memory_demo
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
    parser.add_argument("--auto-bounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bounds-min", type=float, nargs=3, default=None)
    parser.add_argument("--bounds-max", type=float, nargs=3, default=None)
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
    output_dir = args.output_dir
    clip = load_clip(args)
    metrics = run_oracle_memory_demo(
        clip=clip,
        output_dir=output_dir,
        context_frames=args.context_frames,
        target_frame=args.target_frame,
        stride=max(1, args.stride),
        grid_resolution=tuple(args.grid_resolution),
        bounds_min=None if args.bounds_min is None else tuple(args.bounds_min),
        bounds_max=None if args.bounds_max is None else tuple(args.bounds_max),
        auto_bounds=args.auto_bounds,
        splat_radius=args.splat_radius,
    )

    print(json.dumps(metrics, indent=2))
    print(f"saved demo artifacts to {output_dir}")


if __name__ == "__main__":
    main()
