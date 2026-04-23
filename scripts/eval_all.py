#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from world_model.eval.evaluator import pick_device, run_full_evaluation


DEFAULT_RUNS = OrderedDict(
    [
        ("no_memory", ("no_memory", Path("outputs/train_nomemory_real_movia_subset50_v4"))),
        ("memory_baseline", ("memory", Path("outputs/train_memory_real_movia_subset50_v1"))),
        ("memory_strengthened", ("memory", Path("outputs/train_memory_real_movia_subset50_v3"))),
        ("uncertainty_writes", ("memory", Path("outputs/train_memory_uncertainty_real_movia_subset50_v1"))),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 5 evaluation harness over current experiment checkpoints.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/movi_a_128_subset50/manifest.json"))
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-val-windows-per-clip", type=int, default=6)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--checkpoint-kind", choices=["last", "best"], default="last")
    parser.add_argument("--memory-grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--memory-splat-radius", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_phase5_current"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    json_path, markdown_path = run_full_evaluation(
        manifest=args.manifest,
        default_runs=DEFAULT_RUNS,
        output_dir=args.output_dir,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_val_windows_per_clip=args.max_val_windows_per_clip,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
        motion_threshold=args.motion_threshold,
        checkpoint_kind=args.checkpoint_kind,
        device=device,
    )
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
