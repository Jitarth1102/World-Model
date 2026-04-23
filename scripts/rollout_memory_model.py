#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset
from world_model.eval.metrics_image import masked_l1, motion_mask_from_last_context, psnr
from world_model.models.world_model import MemoryConditionedWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained persistent-memory model on a single exported clip window.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--memory-grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--memory-splat-radius", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_strip(frames: list[torch.Tensor], path: Path) -> None:
    np_frames = [(frame.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8") for frame in frames]
    height, width = np_frames[0].shape[1], np_frames[0].shape[2]
    canvas = Image.new("RGB", (width * len(np_frames), height))
    for idx, frame in enumerate(np_frames):
        canvas.paste(Image.fromarray(frame.transpose(1, 2, 0)), (idx * width, 0))
    canvas.save(path)


def save_mask_strip(frames: list[torch.Tensor], path: Path) -> None:
    np_frames = [(frame.squeeze(0).cpu().numpy() * 255.0).astype("uint8") for frame in frames]
    height, width = np_frames[0].shape
    canvas = Image.new("L", (width * len(np_frames), height))
    for idx, frame in enumerate(np_frames):
        canvas.paste(Image.fromarray(frame), (idx * width, 0))
    canvas.save(path)


def save_comparison(
    context_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    predicted_rgb: torch.Tensor,
    baseline_rgb: torch.Tensor,
    output_path: Path,
) -> None:
    rows = []
    for tensor in [context_rgb, target_rgb, predicted_rgb, baseline_rgb]:
        np_frames = [(frame.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8") for frame in tensor]
        height, width = np_frames[0].shape[1], np_frames[0].shape[2]
        row = Image.new("RGB", (width * len(np_frames), height))
        for idx, frame in enumerate(np_frames):
            row.paste(Image.fromarray(frame.transpose(1, 2, 0)), (idx * width, 0))
        rows.append(row)

    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)))
    y_offset = 0
    for row in rows:
        canvas.paste(row, (0, y_offset))
        y_offset += row.height
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    dataset = MemoryConditionedClipWindowDataset(
        clip_paths=[args.clip],
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
    )
    sample = dataset[args.start_frame]

    model = MemoryConditionedWorldModel(hidden_channels=args.hidden_channels).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    context_rgb = sample["context_rgb"].unsqueeze(0).to(device)
    target_rgb = sample["target_rgb"].unsqueeze(0).to(device)
    context_poses = sample["context_poses"].unsqueeze(0).to(device)
    target_poses = sample["target_poses"].unsqueeze(0).to(device)
    target_depth = sample["target_depth"].unsqueeze(0).to(device)
    memory_condition = sample["memory_condition"].unsqueeze(0).to(device)
    baseline = context_rgb[:, -1:].repeat(1, args.predict_frames, 1, 1, 1)

    with torch.no_grad():
        pred_rgb, pred_depth = model(context_rgb, context_poses, target_poses, memory_condition)
    dynamic_mask = motion_mask_from_last_context(context_rgb, target_rgb, threshold=args.motion_threshold)
    depth_mask = (target_depth > 0.0).to(dtype=pred_depth.dtype)

    metrics = {
        "model_l1": float(masked_l1(pred_rgb, target_rgb)),
        "model_psnr": float(psnr(pred_rgb, target_rgb)),
        "model_dynamic_l1": float(masked_l1(pred_rgb, target_rgb, dynamic_mask)),
        "model_depth_l1": float(masked_l1(pred_depth, target_depth, depth_mask)),
        "baseline_l1": float(masked_l1(baseline, target_rgb)),
        "baseline_psnr": float(psnr(baseline, target_rgb)),
        "baseline_dynamic_l1": float(masked_l1(baseline, target_rgb, dynamic_mask)),
        "motion_fraction": float(dynamic_mask.mean()),
        "memory_coverage": float(sample["memory_render_coverage"]),
        "memory_occupancy_fraction": float(sample["memory_occupancy_fraction"]),
        "memory_render_l1_covered": float(sample["memory_render_l1_covered"]),
        "clip_path": str(args.clip),
        "start_frame": args.start_frame,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_strip(list(context_rgb[0].cpu()), args.output_dir / "context_strip.png")
    save_strip(list(target_rgb[0].cpu()), args.output_dir / "target_strip.png")
    save_strip(list(pred_rgb[0].cpu()), args.output_dir / "prediction_strip.png")
    save_strip(list(baseline[0].cpu()), args.output_dir / "baseline_strip.png")
    save_strip(list(sample["memory_render_rgb"]), args.output_dir / "memory_render_strip.png")
    save_mask_strip(list(dynamic_mask[0].cpu()), args.output_dir / "motion_mask_strip.png")
    save_mask_strip(list(sample["memory_render_mask"]), args.output_dir / "memory_mask_strip.png")
    save_comparison(context_rgb[0].cpu(), target_rgb[0].cpu(), pred_rgb[0].cpu(), baseline[0].cpu(), args.output_dir / "comparison.png")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved rollout artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
