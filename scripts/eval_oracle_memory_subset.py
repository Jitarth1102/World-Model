#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model.memory.demo import run_oracle_memory_demo
from world_model.types import ClipSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the oracle persistent-memory demo across a local subset of clips.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON written by prepare_movi.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--target-frame", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--splat-radius", type=int, default=1)
    return parser.parse_args()


def load_manifest(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Path(item) for item in payload["clips"]]


def aggregate(metrics_rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(metrics_rows)
    if count == 0:
        return {"num_clips": 0}
    return {
        "num_clips": count,
        "persistent_beats_last_on_full_mae_count": int(sum(bool(row["persistent_beats_last_on_full_mae"]) for row in metrics_rows)),
        "persistent_beats_last_on_coverage_count": int(sum(bool(row["persistent_beats_last_on_coverage"]) for row in metrics_rows)),
        "mean_persistent_rgb_mae_full": sum(float(row["persistent_rgb_mae_full"]) for row in metrics_rows) / count,
        "mean_last_frame_rgb_mae_full": sum(float(row["last_frame_rgb_mae_full"]) for row in metrics_rows) / count,
        "mean_persistent_coverage": sum(float(row["persistent_coverage"]) for row in metrics_rows) / count,
        "mean_last_frame_coverage": sum(float(row["last_frame_coverage"]) for row in metrics_rows) / count,
    }


def main() -> None:
    args = parse_args()
    clip_paths = load_manifest(args.manifest)[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_clip_rows: list[dict[str, object]] = []

    for clip_path in clip_paths:
        clip = ClipSample.load_npz(clip_path)
        clip_output_dir = args.output_dir / "clips" / clip_path.stem
        metrics = run_oracle_memory_demo(
            clip=clip,
            output_dir=clip_output_dir,
            context_frames=args.context_frames,
            target_frame=args.target_frame,
            stride=max(1, args.stride),
            grid_resolution=tuple(args.grid_resolution),
            auto_bounds=True,
            splat_radius=args.splat_radius,
        )
        metrics["clip_path"] = str(clip_path)
        per_clip_rows.append(metrics)
        print(
            f"{clip_path.name}: full_mae {metrics['persistent_rgb_mae_full']:.4f} vs {metrics['last_frame_rgb_mae_full']:.4f}, "
            f"coverage {metrics['persistent_coverage']:.4f} vs {metrics['last_frame_coverage']:.4f}"
        )

    aggregate_metrics = aggregate(per_clip_rows)
    (args.output_dir / "per_clip_metrics.json").write_text(json.dumps(per_clip_rows, indent=2), encoding="utf-8")
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate_metrics, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_metrics, indent=2))


if __name__ == "__main__":
    main()
