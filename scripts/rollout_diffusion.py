#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset
from world_model.eval.metrics_image import masked_l1, motion_mask_from_last_context, psnr
from world_model.inference.uncertainty_rollout import rollout_diffusion_uncertainty
from world_model.models.diffusion import ConditionalVideoDiffusion
from world_model.uncertainty.calibration import high_error_auroc, uncertainty_error_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained diffusion model on a single exported clip window.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--predict-frames", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--model-channels", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--enable-uncertainty", action="store_true")
    parser.add_argument("--uncertainty-samples", type=int, default=4)
    parser.add_argument("--write-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--confidence-gamma", type=float, default=1.0)
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


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Expected a diffusion checkpoint with 'model_state' and 'config'")
    config = checkpoint["config"]
    variant = config["variant"]
    context_frames = args.context_frames or int(config["context_frames"])
    predict_frames = args.predict_frames or int(config["predict_frames"])
    image_size = args.image_size or int(config["image_size"])
    model_channels = args.model_channels or int(config["model_channels"])
    diffusion_steps = args.diffusion_steps or int(config["diffusion_steps"])
    sample_steps = args.sample_steps or int(config.get("sample_steps_eval", 16))

    model = ConditionalVideoDiffusion(
        context_frames=context_frames,
        predict_frames=predict_frames,
        variant=variant,
        model_channels=model_channels,
        diffusion_steps=diffusion_steps,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = MemoryConditionedClipWindowDataset(
        clip_paths=[args.clip],
        context_frames=context_frames,
        predict_frames=predict_frames,
        image_size=image_size,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
    )
    sample = dataset[args.start_frame]

    if args.enable_uncertainty:
        rollout = rollout_diffusion_uncertainty(
            model=model,
            clip_path=args.clip,
            start_frame=args.start_frame,
            context_frames=context_frames,
            predict_frames=predict_frames,
            image_size=image_size,
            device=device,
            memory_grid_resolution=tuple(args.memory_grid_resolution),
            memory_stride=args.memory_stride,
            memory_splat_radius=args.memory_splat_radius,
            confidence_threshold=args.write_confidence_threshold,
            sample_steps=sample_steps,
            uncertainty_samples=args.uncertainty_samples,
            confidence_gamma=args.confidence_gamma,
        )
        window = rollout["window"]
        context_rgb = window.context_rgb.to(device)
        target_rgb = window.target_rgb.to(device)
        prediction = rollout["prediction"].unsqueeze(0).to(device)
        uncertainty = rollout["uncertainty"].unsqueeze(0).to(device)
        confidence = rollout["confidence"].unsqueeze(0).to(device)
        write_mask = rollout["write_mask"].unsqueeze(0).to(device)
        memory_render_rgb = rollout["memory_render_rgb"]
        memory_render_mask = rollout["memory_render_mask"]
        intermediates = []
    else:
        context_rgb = sample["context_rgb"].unsqueeze(0).to(device)
        target_rgb = sample["target_rgb"].unsqueeze(0).to(device)
        context_poses = sample["context_poses"].unsqueeze(0).to(device)
        target_poses = sample["target_poses"].unsqueeze(0).to(device)
        memory_condition = sample["memory_condition"].unsqueeze(0).to(device)
        with torch.no_grad():
            prediction, intermediates = model.sample(
                context_rgb=context_rgb,
                context_poses=context_poses,
                target_poses=target_poses,
                memory_condition=memory_condition if variant == "memory" else None,
                sample_steps=sample_steps,
                eta=0.0,
                return_intermediates=True,
            )
        uncertainty = None
        confidence = None
        write_mask = None
        memory_render_rgb = sample["memory_render_rgb"]
        memory_render_mask = sample["memory_render_mask"]

    baseline = context_rgb[:, -1:].repeat(1, predict_frames, 1, 1, 1)
    dynamic_mask = motion_mask_from_last_context(context_rgb, target_rgb, threshold=args.motion_threshold)
    memory_mask = memory_render_mask.unsqueeze(0).to(device)

    metrics = {
        "model_l1": float(masked_l1(prediction, target_rgb)),
        "model_psnr": float(psnr(prediction, target_rgb)),
        "model_dynamic_l1": float(masked_l1(prediction, target_rgb, dynamic_mask)),
        "model_memory_covered_l1": float(masked_l1(prediction, target_rgb, memory_mask)),
        "baseline_l1": float(masked_l1(baseline, target_rgb)),
        "baseline_psnr": float(psnr(baseline, target_rgb)),
        "baseline_dynamic_l1": float(masked_l1(baseline, target_rgb, dynamic_mask)),
        "baseline_memory_covered_l1": float(masked_l1(baseline, target_rgb, memory_mask)),
        "motion_fraction": float(dynamic_mask.mean()),
        "memory_coverage": float(memory_render_mask.float().mean()),
        "memory_occupancy_fraction": float(sample["memory_occupancy_fraction"]),
        "memory_render_l1_covered": float(masked_l1(memory_render_rgb.unsqueeze(0).to(device), target_rgb, memory_mask)),
        "clip_path": str(args.clip),
        "start_frame": args.start_frame,
        "variant": "diffusion_memory_uncertainty" if args.enable_uncertainty else variant,
        "model_type": "diffusion",
        "sample_steps": sample_steps,
    }
    if uncertainty is not None:
        error_map = (prediction - target_rgb).abs().mean(dim=2, keepdim=True)
        metrics.update(
            {
                "uncertainty_error_corr": float(uncertainty_error_correlation(uncertainty, error_map, mask=memory_mask)),
                "high_error_auroc": float(high_error_auroc(uncertainty, error_map, mask=memory_mask)),
                "write_coverage": float(write_mask.mean()),
                "confidence_mean": float(confidence.mean()),
                "uncertainty_samples": args.uncertainty_samples,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_strip(list(context_rgb[0].cpu()), args.output_dir / "context_strip.png")
    save_strip(list(target_rgb[0].cpu()), args.output_dir / "target_strip.png")
    save_strip(list(prediction[0].cpu()), args.output_dir / "prediction_strip.png")
    save_strip(list(baseline[0].cpu()), args.output_dir / "baseline_strip.png")
    save_strip(list(memory_render_rgb), args.output_dir / "memory_render_strip.png")
    save_mask_strip(list(dynamic_mask[0].cpu()), args.output_dir / "motion_mask_strip.png")
    save_mask_strip(list(memory_render_mask), args.output_dir / "memory_mask_strip.png")
    if uncertainty is not None:
        save_scalar_strip(list(uncertainty[0].cpu()), args.output_dir / "uncertainty_strip.png")
        save_scalar_strip(list(confidence[0].cpu()), args.output_dir / "confidence_strip.png")
        save_mask_strip(list(write_mask[0].cpu()), args.output_dir / "write_mask_strip.png")
    save_comparison(context_rgb[0].cpu(), target_rgb[0].cpu(), prediction[0].cpu(), baseline[0].cpu(), args.output_dir / "comparison.png")
    if intermediates:
        denoise_frames = [step[0, 0].cpu() for step in intermediates]
        save_strip(denoise_frames, args.output_dir / "denoising_first_frame_strip.png")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved rollout artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
