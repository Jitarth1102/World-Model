#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from world_model.types import ClipSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an exported .npz clip from MOVi.")
    parser.add_argument("--clip", type=Path, required=True, help="Path to a prepared .npz clip.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for visualization artifacts.")
    parser.add_argument("--max-frames", type=int, default=8, help="How many frames to show in strip/overview outputs.")
    return parser.parse_args()


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0.0
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)
    depth_values = depth[valid]
    low = np.percentile(depth_values, 2.0)
    high = np.percentile(depth_values, 98.0)
    scaled = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    colored = plt.get_cmap("viridis")(scaled)[..., :3]
    colored[~valid] = 0.0
    return (colored * 255.0).astype(np.uint8)


def colorize_segmentation(segmentation: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
        ],
        dtype=np.uint8,
    )
    indexed = segmentation.astype(np.int64) % len(palette)
    return palette[indexed]


def save_gif(frames: list[np.ndarray], path: Path, duration_ms: int = 160) -> None:
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def save_strip(frames: list[np.ndarray], path: Path) -> None:
    height, width = frames[0].shape[:2]
    canvas = np.zeros((height, width * len(frames), 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        canvas[:, idx * width : (idx + 1) * width] = frame
    Image.fromarray(canvas).save(path)


def save_overview(
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    seg_frames: list[np.ndarray] | None,
    output_path: Path,
) -> None:
    rows = 3 if seg_frames is not None else 2
    cols = len(rgb_frames)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]

    for idx in range(cols):
        axes[0, idx].imshow(rgb_frames[idx])
        axes[0, idx].set_title(f"RGB {idx}")
        axes[1, idx].imshow(depth_frames[idx])
        axes[1, idx].set_title(f"Depth {idx}")
        if seg_frames is not None:
            axes[2, idx].imshow(seg_frames[idx])
            axes[2, idx].set_title(f"Seg {idx}")

    for axis in axes.ravel():
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    clip = ClipSample.load_npz(args.clip)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = min(args.max_frames, clip.num_frames)
    rgb_frames = [clip.video[idx] for idx in range(num_frames)]
    depth_frames = [normalize_depth(clip.depth[idx]) for idx in range(num_frames)]
    seg_frames = None
    if clip.segmentations is not None:
        seg_frames = [colorize_segmentation(clip.segmentations[idx]) for idx in range(num_frames)]

    save_strip(rgb_frames, args.output_dir / "rgb_strip.png")
    save_strip(depth_frames, args.output_dir / "depth_strip.png")
    save_gif(rgb_frames, args.output_dir / "rgb.gif")
    save_gif(depth_frames, args.output_dir / "depth.gif")

    if seg_frames is not None:
        save_strip(seg_frames, args.output_dir / "segmentation_strip.png")
        save_gif(seg_frames, args.output_dir / "segmentation.gif")

    save_overview(rgb_frames, depth_frames, seg_frames, args.output_dir / "overview.png")

    summary = {
        "clip_path": str(args.clip),
        "num_frames": clip.num_frames,
        "image_size": list(clip.image_size),
        "depth_min": float(clip.depth.min()),
        "depth_max": float(clip.depth.max()),
        "depth_mean": float(clip.depth.mean()),
        "intrinsics": clip.intrinsics.as_dict(),
        "metadata": clip.metadata,
        "has_segmentations": clip.segmentations is not None,
        "has_visibility": clip.visibility is not None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved visualization artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
