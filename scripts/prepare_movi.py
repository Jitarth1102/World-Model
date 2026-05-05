#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from world_model.data.movi import iter_movi_split, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MOVi examples into local ClipSample .npz shards.")
    parser.add_argument("--dataset", default="movi_a", help="Dataset variant, e.g. movi_a or movi_c.")
    parser.add_argument("--resolution", default="128x128", help="Resolution string, e.g. 128x128.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of clips to export. Use 0 for the full split (no cap).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for .npz clips.")
    parser.add_argument("--data-dir", default=None, help="Optional local tfds data dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_limit = None if args.limit <= 0 else args.limit
    saved_paths = []
    for idx, clip in enumerate(
        iter_movi_split(
            dataset=args.dataset,
            split=args.split,
            resolution=args.resolution,
            data_dir=args.data_dir,
            limit=clip_limit,
        )
    ):
        clip_path = output_dir / f"{idx:05d}.npz"
        clip.save_npz(clip_path)
        saved_paths.append(clip_path)
        print(f"saved {clip_path}")

    manifest_path = output_dir / "manifest.json"
    save_manifest(saved_paths, manifest_path)
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
