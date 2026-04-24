#!/usr/bin/env python3
"""Prepare an evaluation-only uncertainty ablation run directory.

This clones the checkpoint and metrics metadata from an existing
`memory_uncertainty_convgru` run while overriding only the write threshold.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        required=True,
        help="Source run directory containing memory_model_best.pt and metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination ablation run directory.",
    )
    parser.add_argument(
        "--write-confidence-threshold",
        type=float,
        required=True,
        help="New confidence threshold to store in metrics.json for evaluation-time gating.",
    )
    parser.add_argument(
        "--confidence-gamma",
        type=float,
        default=1.0,
        help="Optional evaluation-time sharpening exponent applied to confidence before thresholding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.base_run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.json: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    metrics["write_confidence_threshold"] = args.write_confidence_threshold
    metrics["confidence_gamma"] = args.confidence_gamma
    metrics["ablation_parent"] = str(args.base_run_dir)
    metrics["ablation_type"] = "write_confidence_threshold"

    for filename in ("memory_model_best.pt", "memory_model_last.pt"):
        source = args.base_run_dir / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"prepared {args.output_dir} with threshold={args.write_confidence_threshold:.3f} "
        f"gamma={args.confidence_gamma:.3f}"
    )


if __name__ == "__main__":
    main()
