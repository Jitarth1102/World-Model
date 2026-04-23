#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset, load_manifest, split_clip_paths
from world_model.eval.metrics_image import masked_l1, motion_mask_from_last_context, psnr
from world_model.inference.uncertainty_rollout import rollout_diffusion_uncertainty
from world_model.models.diffusion import ConditionalVideoDiffusion
from world_model.uncertainty.calibration import high_error_auroc, uncertainty_error_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a diffusion memory checkpoint with uncertainty-aware memory writes.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/train_diffusion_memory_real_movia_subset50_longer_v1/diffusion_model_best.pt"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/movi_a_128_subset50/manifest.json"))
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-val-windows-per-clip", type=int, default=6)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--memory-grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--memory-splat-radius", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--uncertainty-samples", type=int, default=4)
    parser.add_argument("--write-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train_diffusion_memory_uncertainty_real_movia_subset50_v1"))
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


def save_scalar_strip(frames: list[torch.Tensor], path: Path) -> None:
    normalized = []
    for frame in frames:
        array = frame.squeeze(0).cpu().numpy().astype("float32")
        min_value = float(array.min())
        max_value = float(array.max())
        if max_value > min_value:
            array = (array - min_value) / (max_value - min_value)
        else:
            array = np.zeros_like(array)
        normalized.append((np.clip(array, 0.0, 1.0) * 255.0).astype("uint8"))
    height, width = normalized[0].shape
    canvas = Image.new("L", (width * len(normalized), height))
    for idx, frame in enumerate(normalized):
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


def save_example_artifacts(example: dict[str, torch.Tensor | str | int | float], output_dir: Path, prefix: str) -> None:
    save_comparison(example["context_rgb"], example["target_rgb"], example["prediction"], example["baseline"], output_dir / f"{prefix}_comparison.png")
    save_strip(list(example["target_rgb"]), output_dir / f"{prefix}_target_strip.png")
    save_strip(list(example["prediction"]), output_dir / f"{prefix}_prediction_strip.png")
    save_strip(list(example["baseline"]), output_dir / f"{prefix}_baseline_strip.png")
    save_strip(list(example["memory_render_rgb"]), output_dir / f"{prefix}_memory_render_strip.png")
    save_mask_strip(list(example["motion_mask"]), output_dir / f"{prefix}_motion_mask_strip.png")
    save_mask_strip(list(example["memory_render_mask"]), output_dir / f"{prefix}_memory_mask_strip.png")
    save_scalar_strip(list(example["uncertainty"]), output_dir / f"{prefix}_uncertainty_strip.png")
    save_scalar_strip(list(example["confidence"]), output_dir / f"{prefix}_confidence_strip.png")
    save_mask_strip(list(example["write_mask"]), output_dir / f"{prefix}_write_mask_strip.png")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Expected a diffusion checkpoint with model_state/config")
    config = checkpoint["config"]
    model = ConditionalVideoDiffusion(
        context_frames=int(config["context_frames"]),
        predict_frames=int(config["predict_frames"]),
        variant=str(config["variant"]),
        model_channels=int(config["model_channels"]),
        diffusion_steps=int(config["diffusion_steps"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    clip_paths = load_manifest(args.manifest)
    _, val_paths = split_clip_paths(clip_paths, val_ratio=args.val_ratio, seed=args.seed)
    dataset = MemoryConditionedClipWindowDataset(
        clip_paths=val_paths,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        max_windows_per_clip=args.max_val_windows_per_clip,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
    )

    totals = {
        "model_l1": 0.0,
        "model_psnr": 0.0,
        "baseline_l1": 0.0,
        "baseline_psnr": 0.0,
        "model_dynamic_l1": 0.0,
        "baseline_dynamic_l1": 0.0,
        "model_memory_covered_l1": 0.0,
        "baseline_memory_covered_l1": 0.0,
        "motion_fraction": 0.0,
        "memory_coverage": 0.0,
        "memory_render_l1_covered": 0.0,
        "uncertainty_error_corr": 0.0,
        "high_error_auroc": 0.0,
        "write_coverage": 0.0,
        "confidence_mean": 0.0,
    }
    count = 0
    best_example: dict[str, torch.Tensor | str | int | float] | None = None
    best_example_l1 = float("inf")
    first_example: dict[str, torch.Tensor | str | int | float] | None = None

    for window in dataset.windows:
        rollout = rollout_diffusion_uncertainty(
            model=model,
            clip_path=window.clip_path,
            start_frame=window.start_frame,
            context_frames=args.context_frames,
            predict_frames=args.predict_frames,
            image_size=args.image_size,
            device=device,
            memory_grid_resolution=tuple(args.memory_grid_resolution),
            memory_stride=args.memory_stride,
            memory_splat_radius=args.memory_splat_radius,
            confidence_threshold=args.write_confidence_threshold,
            sample_steps=args.sample_steps,
            uncertainty_samples=args.uncertainty_samples,
        )
        window_tensors = rollout["window"]
        context_rgb = window_tensors.context_rgb
        target_rgb = window_tensors.target_rgb
        prediction = rollout["prediction"].unsqueeze(0)
        baseline = context_rgb[:, -1:].repeat(1, args.predict_frames, 1, 1, 1)
        dynamic_mask = motion_mask_from_last_context(context_rgb, target_rgb, threshold=args.motion_threshold)
        memory_mask = rollout["memory_render_mask"].unsqueeze(0)
        uncertainty = rollout["uncertainty"].unsqueeze(0)
        error_map = (prediction - target_rgb).abs().mean(dim=2, keepdim=True)

        current_metrics = {
            "model_l1": float(masked_l1(prediction, target_rgb)),
            "model_psnr": float(psnr(prediction, target_rgb)),
            "baseline_l1": float(masked_l1(baseline, target_rgb)),
            "baseline_psnr": float(psnr(baseline, target_rgb)),
            "model_dynamic_l1": float(masked_l1(prediction, target_rgb, dynamic_mask)),
            "baseline_dynamic_l1": float(masked_l1(baseline, target_rgb, dynamic_mask)),
            "model_memory_covered_l1": float(masked_l1(prediction, target_rgb, memory_mask)),
            "baseline_memory_covered_l1": float(masked_l1(baseline, target_rgb, memory_mask)),
            "motion_fraction": float(dynamic_mask.mean()),
            "memory_coverage": float(memory_mask.float().mean()),
            "memory_render_l1_covered": float(masked_l1(rollout["memory_render_rgb"].unsqueeze(0), target_rgb, memory_mask)),
            "uncertainty_error_corr": float(uncertainty_error_correlation(uncertainty, error_map, mask=memory_mask)),
            "high_error_auroc": float(high_error_auroc(uncertainty, error_map, mask=memory_mask)),
            "write_coverage": float(rollout["write_mask"].float().mean()),
            "confidence_mean": float(rollout["confidence"].float().mean()),
        }
        for key, value in current_metrics.items():
            totals[key] += value
        count += 1

        example = {
            "context_rgb": context_rgb[0].cpu(),
            "target_rgb": target_rgb[0].cpu(),
            "prediction": prediction[0].cpu(),
            "baseline": baseline[0].cpu(),
            "motion_mask": dynamic_mask[0].cpu(),
            "memory_render_rgb": rollout["memory_render_rgb"].cpu(),
            "memory_render_mask": rollout["memory_render_mask"].cpu(),
            "uncertainty": rollout["uncertainty"].cpu(),
            "confidence": rollout["confidence"].cpu(),
            "write_mask": rollout["write_mask"].cpu(),
            "clip_path": str(window.clip_path),
            "start_frame": int(window.start_frame),
        }
        if first_example is None:
            first_example = example
        if current_metrics["model_l1"] < best_example_l1:
            best_example_l1 = current_metrics["model_l1"]
            best_example = example

    if count == 0:
        raise RuntimeError("Validation dataset is empty.")
    final_val = {key: value / count for key, value in totals.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, args.output_dir / "diffusion_model_best.pt")
    shutil.copy2(args.checkpoint, args.output_dir / "diffusion_model_last.pt")
    if best_example is not None:
        save_example_artifacts(best_example, args.output_dir, "best_val")
    if first_example is not None:
        save_example_artifacts(first_example, args.output_dir, "final_val")

    summary = {
        "device": str(device),
        "model_type": "diffusion",
        "variant": "diffusion_memory_uncertainty",
        "epochs_completed": 0,
        "global_steps": 0,
        "batch_size": 0,
        "learning_rate": 0.0,
        "context_frames": args.context_frames,
        "predict_frames": args.predict_frames,
        "image_size": args.image_size,
        "model_channels": int(config["model_channels"]),
        "diffusion_steps": int(config["diffusion_steps"]),
        "sample_steps_eval": args.sample_steps,
        "uncertainty_samples": args.uncertainty_samples,
        "write_confidence_threshold": args.write_confidence_threshold,
        "memory_grid_resolution": list(args.memory_grid_resolution),
        "memory_stride": args.memory_stride,
        "memory_splat_radius": args.memory_splat_radius,
        "num_val_clips": len(val_paths),
        "num_val_windows": len(dataset),
        "best_val_l1": final_val["model_l1"],
        "final_val": final_val,
        "history": [],
        "example_clip_path": None if first_example is None else first_example["clip_path"],
        "example_start_frame": None if first_example is None else first_example["start_frame"],
        "source_checkpoint": str(args.checkpoint),
        "model_beats_baseline_on_val_l1": final_val["model_l1"] < final_val["baseline_l1"],
        "model_beats_baseline_on_val_dynamic_l1": final_val["model_dynamic_l1"] < final_val["baseline_dynamic_l1"],
        "model_beats_baseline_on_val_memory_covered_l1": final_val["model_memory_covered_l1"] < final_val["baseline_memory_covered_l1"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
