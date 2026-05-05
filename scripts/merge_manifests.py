#!/usr/bin/env python3
"""Merge multiple manifest.json files into one for multi-dataset training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="Paths to manifest.json files (order is preserved).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest path (parent dirs are created).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips: list[str] = []
    seen: set[str] = set()
    for path in args.manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("clips", []):
            s = str(Path(item).expanduser())
            if s not in seen:
                seen.add(s)
                clips.append(s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"clips": clips}, indent=2), encoding="utf-8")
    print(f"wrote {len(clips)} unique clips to {args.output}")


if __name__ == "__main__":
    main()
