#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset, load_manifest, split_clip_paths
from world_model.eval.metrics_image import masked_l1, motion_mask_from_last_context, psnr
from world_model.inference.uncertainty_rollout import rollout_convgru_uncertainty
from world_model.models.world_model import MemoryConditionedWorldModel
from world_model.uncertainty.calibration import high_error_auroc, uncertainty_error_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 4 persistent-memory world model.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/movi_a_128_subset50/manifest.json"))
    parser.add_argument(
        "--warm-start-nomemory-checkpoint",
        type=Path,
        default=Path("outputs/train_nomemory_real_movia_subset50_v4/nomemory_model_best.pt"),
    )
    parser.add_argument(
        "--warm-start-memory-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint from an existing memory model. Compatible weights are loaded and new heads stay random.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Optional hard cap on optimizer steps.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--max-train-windows-per-clip", type=int, default=12)
    parser.add_argument("--max-val-windows-per-clip", type=int, default=6)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--dynamic-loss-weight", type=float, default=2.0)
    parser.add_argument("--depth-loss-weight", type=float, default=0.1)
    parser.add_argument("--memory-covered-loss-weight", type=float, default=1.0)
    parser.add_argument("--memory-render-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--memory-splat-radius", type=int, default=1)
    parser.add_argument("--enable-uncertainty", action="store_true")
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.25)
    parser.add_argument("--write-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--confidence-gamma", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train_memory_real"))
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
        if array.size == 0:
            array = np.zeros((1, 1), dtype="float32")
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


def baseline_copy_last(context_rgb: torch.Tensor, predict_frames: int) -> torch.Tensor:
    last_frame = context_rgb[:, -1:]
    return last_frame.repeat(1, predict_frames, 1, 1, 1)


def warm_start_from_checkpoint(model: MemoryConditionedWorldModel, checkpoint_path: Path | None) -> dict[str, object]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {
            "applied": False,
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "loaded_keys": [],
            "missing_checkpoint": checkpoint_path is not None and not checkpoint_path.exists(),
        }
    source_state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(source_state, dict) and "model_state" in source_state:
        source_state = source_state["model_state"]
    model_state = model.state_dict()
    loaded_keys: list[str] = []
    for key, value in source_state.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            model_state[key] = value.clone()
            loaded_keys.append(key)
    model.load_state_dict(model_state)
    return {
        "applied": True,
        "checkpoint": str(checkpoint_path),
        "loaded_keys": sorted(set(loaded_keys)),
        "missing_checkpoint": False,
    }


def warm_start_from_nomemory_checkpoint(
    model: MemoryConditionedWorldModel,
    checkpoint_path: Path | None,
) -> dict[str, object]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {
            "applied": False,
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "loaded_keys": [],
            "missing_checkpoint": checkpoint_path is not None and not checkpoint_path.exists(),
        }

    source_state = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    loaded_keys: list[str] = []

    def copy_if_compatible(source_key: str, target_key: str) -> None:
        if source_key not in source_state or target_key not in model_state:
            return
        if source_state[source_key].shape != model_state[target_key].shape:
            return
        model_state[target_key] = source_state[source_key].clone()
        loaded_keys.append(target_key)

    direct_prefix_map = {
        "encoder.": "image_encoder.",
        "pose_proj.": "pose_proj.",
        "temporal_cell.": "temporal_cell.",
        "decoder.": "rgb_decoder.",
    }
    for source_key in source_state:
        for source_prefix, target_prefix in direct_prefix_map.items():
            if source_key.startswith(source_prefix):
                copy_if_compatible(source_key, target_prefix + source_key[len(source_prefix) :])
                break

    # Initialize the memory encoder from the RGB encoder where shapes allow.
    memory_first = "memory_encoder.net.0.weight"
    source_first = "encoder.net.0.weight"
    if source_first in source_state and memory_first in model_state:
        source_weight = source_state[source_first]
        target_weight = model_state[memory_first].clone()
        if source_weight.shape[0] == target_weight.shape[0] and source_weight.shape[2:] == target_weight.shape[2:]:
            target_weight.zero_()
            target_weight[:, : source_weight.shape[1]] = source_weight
            model_state[memory_first] = target_weight
            loaded_keys.append(memory_first)

    for suffix in [
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
    ]:
        copy_if_compatible(f"encoder.{suffix}", f"memory_encoder.{suffix}")

    model.load_state_dict(model_state)
    return {
        "applied": True,
        "checkpoint": str(checkpoint_path),
        "loaded_keys": sorted(set(loaded_keys)),
        "missing_checkpoint": False,
    }


def compute_loss(
    pred_rgb: torch.Tensor,
    pred_depth: torch.Tensor,
    pred_log_variance: torch.Tensor | None,
    target_rgb: torch.Tensor,
    target_depth: torch.Tensor,
    context_rgb: torch.Tensor,
    memory_render_rgb: torch.Tensor,
    memory_render_mask: torch.Tensor,
    motion_threshold: float,
    dynamic_loss_weight: float,
    depth_loss_weight: float,
    memory_covered_loss_weight: float,
    memory_render_loss_weight: float,
    uncertainty_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    full_l1 = masked_l1(pred_rgb, target_rgb)
    dynamic_mask = motion_mask_from_last_context(context_rgb, target_rgb, threshold=motion_threshold)
    dynamic_l1 = masked_l1(pred_rgb, target_rgb, dynamic_mask)
    depth_mask = (target_depth > 0.0).to(dtype=pred_depth.dtype)
    depth_l1 = masked_l1(pred_depth, target_depth, depth_mask)
    memory_covered_l1 = masked_l1(pred_rgb, target_rgb, memory_render_mask)
    memory_render_l1 = masked_l1(pred_rgb, memory_render_rgb, memory_render_mask)
    uncertainty_nll = torch.zeros((), dtype=pred_rgb.dtype, device=pred_rgb.device)
    if pred_log_variance is not None:
        rgb_error_sq = (pred_rgb - target_rgb).pow(2).mean(dim=2, keepdim=True)
        uncertainty_nll = (0.5 * torch.exp(-pred_log_variance) * rgb_error_sq + 0.5 * pred_log_variance).mean()
    objective = (
        full_l1
        + dynamic_loss_weight * dynamic_l1
        + depth_loss_weight * depth_l1
        + memory_covered_loss_weight * memory_covered_l1
        + memory_render_loss_weight * memory_render_l1
        + uncertainty_loss_weight * uncertainty_nll
    )
    return objective, full_l1, dynamic_l1, depth_l1, memory_covered_l1, memory_render_l1, uncertainty_nll


def move_batch_to_device(batch: dict[str, torch.Tensor | str | int | float], device: torch.device) -> dict[str, torch.Tensor | str | int | float]:
    moved: dict[str, torch.Tensor | str | int | float] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=torch.float32)
            else:
                moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_example_artifacts(example: dict[str, torch.Tensor | str | int | float], output_dir: Path, prefix: str) -> None:
    save_comparison(
        example["context_rgb"],
        example["target_rgb"],
        example["prediction"],
        example["baseline"],
        output_dir / f"{prefix}_comparison.png",
    )
    save_strip(list(example["target_rgb"]), output_dir / f"{prefix}_target_strip.png")
    save_strip(list(example["prediction"]), output_dir / f"{prefix}_prediction_strip.png")
    save_strip(list(example["baseline"]), output_dir / f"{prefix}_baseline_strip.png")
    save_strip(list(example["memory_render_rgb"]), output_dir / f"{prefix}_memory_render_strip.png")
    save_mask_strip(list(example["motion_mask"]), output_dir / f"{prefix}_motion_mask_strip.png")
    save_mask_strip(list(example["memory_render_mask"]), output_dir / f"{prefix}_memory_mask_strip.png")
    if "uncertainty" in example:
        save_scalar_strip(list(example["uncertainty"]), output_dir / f"{prefix}_uncertainty_strip.png")
    if "confidence" in example:
        save_scalar_strip(list(example["confidence"]), output_dir / f"{prefix}_confidence_strip.png")
    if "write_mask" in example:
        save_mask_strip(list(example["write_mask"]), output_dir / f"{prefix}_write_mask_strip.png")


@torch.no_grad()
def evaluate_model(
    model: MemoryConditionedWorldModel,
    loader: DataLoader,
    device: torch.device,
    motion_threshold: float,
    enable_uncertainty: bool,
    context_frames: int,
    predict_frames: int,
    image_size: int,
    memory_grid_resolution: tuple[int, int, int],
    memory_stride: int,
    memory_splat_radius: int,
    write_confidence_threshold: float,
    confidence_gamma: float,
) -> tuple[dict[str, float], dict[str, torch.Tensor | str | int | float]]:
    model.eval()
    total_model_l1 = 0.0
    total_model_psnr = 0.0
    total_base_l1 = 0.0
    total_base_psnr = 0.0
    total_model_dynamic_l1 = 0.0
    total_base_dynamic_l1 = 0.0
    total_model_depth_l1 = 0.0
    total_model_memory_covered_l1 = 0.0
    total_base_memory_covered_l1 = 0.0
    total_motion_fraction = 0.0
    total_memory_coverage = 0.0
    total_memory_occupancy = 0.0
    total_memory_render_l1 = 0.0
    total_uncertainty_corr = 0.0
    total_high_error_auroc = 0.0
    total_write_coverage = 0.0
    total_confidence_mean = 0.0
    num_batches = 0
    first_example: dict[str, torch.Tensor | str | int | float] | None = None

    for batch in loader:
        if enable_uncertainty:
            batch_size = len(batch["clip_path"])
            for sample_idx in range(batch_size):
                clip_path = batch["clip_path"][sample_idx]
                start_frame = int(batch["start_frame"][sample_idx]) if isinstance(batch["start_frame"], torch.Tensor) else int(batch["start_frame"][sample_idx])
                rollout = rollout_convgru_uncertainty(
                    model=model,
                    clip_path=clip_path,
                    start_frame=start_frame,
                    context_frames=context_frames,
                    predict_frames=predict_frames,
                    image_size=image_size,
                    device=device,
                    memory_grid_resolution=memory_grid_resolution,
                    memory_stride=memory_stride,
                    memory_splat_radius=memory_splat_radius,
                    confidence_threshold=write_confidence_threshold,
                    confidence_gamma=confidence_gamma,
                )
                window = rollout["window"]
                context_rgb = window.context_rgb
                target_rgb = window.target_rgb
                target_depth = window.target_depth
                pred_rgb = rollout["prediction"].unsqueeze(0)
                pred_depth = rollout["pred_depth"].unsqueeze(0)
                baseline = baseline_copy_last(context_rgb, target_rgb.shape[1])
                dynamic_mask = motion_mask_from_last_context(context_rgb, target_rgb, threshold=motion_threshold)
                depth_mask = (target_depth > 0.0).to(dtype=pred_depth.dtype)
                memory_mask = rollout["memory_render_mask"].unsqueeze(0)
                uncertainty = rollout["uncertainty"].unsqueeze(0)
                error_map = (pred_rgb - target_rgb).abs().mean(dim=2, keepdim=True)

                total_model_l1 += float(masked_l1(pred_rgb, target_rgb))
                total_model_psnr += float(psnr(pred_rgb, target_rgb))
                total_base_l1 += float(masked_l1(baseline, target_rgb))
                total_base_psnr += float(psnr(baseline, target_rgb))
                total_model_dynamic_l1 += float(masked_l1(pred_rgb, target_rgb, dynamic_mask))
                total_base_dynamic_l1 += float(masked_l1(baseline, target_rgb, dynamic_mask))
                total_model_depth_l1 += float(masked_l1(pred_depth, target_depth, depth_mask))
                total_model_memory_covered_l1 += float(masked_l1(pred_rgb, target_rgb, memory_mask))
                total_base_memory_covered_l1 += float(masked_l1(baseline, target_rgb, memory_mask))
                total_motion_fraction += float(dynamic_mask.mean())
                total_memory_coverage += float(rollout["memory_render_mask"].float().mean())
                total_memory_occupancy += float(rollout["write_coverage"])
                total_memory_render_l1 += float(masked_l1(rollout["memory_render_rgb"].unsqueeze(0), target_rgb, memory_mask))
                total_uncertainty_corr += uncertainty_error_correlation(uncertainty, error_map, mask=memory_mask)
                total_high_error_auroc += high_error_auroc(uncertainty, error_map, mask=memory_mask)
                total_write_coverage += float(rollout["write_coverage"])
                total_confidence_mean += float(rollout["confidence_mean"])
                num_batches += 1

                if first_example is None:
                    first_example = {
                        "context_rgb": context_rgb[0].detach().cpu(),
                        "target_rgb": target_rgb[0].detach().cpu(),
                        "prediction": pred_rgb[0].detach().cpu(),
                        "baseline": baseline[0].detach().cpu(),
                        "motion_mask": dynamic_mask[0].detach().cpu(),
                        "memory_render_rgb": rollout["memory_render_rgb"].detach().cpu(),
                        "memory_render_mask": rollout["memory_render_mask"].detach().cpu(),
                        "uncertainty": rollout["uncertainty"].detach().cpu(),
                        "confidence": rollout["confidence"].detach().cpu(),
                        "write_mask": rollout["write_mask"].detach().cpu(),
                        "clip_path": clip_path,
                        "start_frame": start_frame,
                        "memory_render_coverage": float(rollout["memory_render_mask"].float().mean()),
                        "memory_occupancy_fraction": float(rollout["write_coverage"]),
                        "memory_render_l1_covered": float(masked_l1(rollout["memory_render_rgb"].unsqueeze(0), target_rgb, memory_mask)),
                    }
        else:
            batch = move_batch_to_device(batch, device)
            context_rgb = batch["context_rgb"]
            target_rgb = batch["target_rgb"]
            context_poses = batch["context_poses"]
            target_poses = batch["target_poses"]
            target_depth = batch["target_depth"]
            memory_condition = batch["memory_condition"]

            model_outputs = model(context_rgb, context_poses, target_poses, memory_condition)
            pred_rgb, pred_depth = model_outputs[:2]
            baseline = baseline_copy_last(context_rgb, target_rgb.shape[1])
            dynamic_mask = motion_mask_from_last_context(context_rgb, target_rgb, threshold=motion_threshold)
            depth_mask = (target_depth > 0.0).to(dtype=pred_depth.dtype)
            memory_mask = batch["memory_render_mask"]

            total_model_l1 += float(masked_l1(pred_rgb, target_rgb))
            total_model_psnr += float(psnr(pred_rgb, target_rgb))
            total_base_l1 += float(masked_l1(baseline, target_rgb))
            total_base_psnr += float(psnr(baseline, target_rgb))
            total_model_dynamic_l1 += float(masked_l1(pred_rgb, target_rgb, dynamic_mask))
            total_base_dynamic_l1 += float(masked_l1(baseline, target_rgb, dynamic_mask))
            total_model_depth_l1 += float(masked_l1(pred_depth, target_depth, depth_mask))
            total_model_memory_covered_l1 += float(masked_l1(pred_rgb, target_rgb, memory_mask))
            total_base_memory_covered_l1 += float(masked_l1(baseline, target_rgb, memory_mask))
            total_motion_fraction += float(dynamic_mask.mean())
            total_memory_coverage += float(batch["memory_render_coverage"].float().mean())
            total_memory_occupancy += float(batch["memory_occupancy_fraction"].float().mean())
            total_memory_render_l1 += float(batch["memory_render_l1_covered"].float().mean())
            num_batches += 1

            if first_example is None:
                first_example = {
                    "context_rgb": context_rgb[0].detach().cpu(),
                    "target_rgb": target_rgb[0].detach().cpu(),
                    "prediction": pred_rgb[0].detach().cpu(),
                    "baseline": baseline[0].detach().cpu(),
                    "motion_mask": dynamic_mask[0].detach().cpu(),
                    "memory_render_rgb": batch["memory_render_rgb"][0].detach().cpu(),
                    "memory_render_mask": batch["memory_render_mask"][0].detach().cpu(),
                    "clip_path": batch["clip_path"][0] if isinstance(batch["clip_path"], list) else batch["clip_path"],
                    "start_frame": int(batch["start_frame"][0]) if isinstance(batch["start_frame"], torch.Tensor) else int(batch["start_frame"]),
                    "memory_render_coverage": float(batch["memory_render_coverage"][0]) if isinstance(batch["memory_render_coverage"], torch.Tensor) else float(batch["memory_render_coverage"]),
                    "memory_occupancy_fraction": float(batch["memory_occupancy_fraction"][0]) if isinstance(batch["memory_occupancy_fraction"], torch.Tensor) else float(batch["memory_occupancy_fraction"]),
                    "memory_render_l1_covered": float(batch["memory_render_l1_covered"][0]) if isinstance(batch["memory_render_l1_covered"], torch.Tensor) else float(batch["memory_render_l1_covered"]),
                }

    if num_batches == 0:
        raise RuntimeError("Evaluation loader is empty.")
    metrics = {
        "model_l1": total_model_l1 / num_batches,
        "model_psnr": total_model_psnr / num_batches,
        "baseline_l1": total_base_l1 / num_batches,
        "baseline_psnr": total_base_psnr / num_batches,
        "model_dynamic_l1": total_model_dynamic_l1 / num_batches,
        "baseline_dynamic_l1": total_base_dynamic_l1 / num_batches,
        "model_depth_l1": total_model_depth_l1 / num_batches,
        "model_memory_covered_l1": total_model_memory_covered_l1 / num_batches,
        "baseline_memory_covered_l1": total_base_memory_covered_l1 / num_batches,
        "motion_fraction": total_motion_fraction / num_batches,
        "memory_coverage": total_memory_coverage / num_batches,
        "memory_occupancy_fraction": total_memory_occupancy / num_batches,
        "memory_render_l1_covered": total_memory_render_l1 / num_batches,
        "uncertainty_error_corr": total_uncertainty_corr / num_batches if num_batches else 0.0,
        "high_error_auroc": total_high_error_auroc / num_batches if num_batches else 0.5,
        "write_coverage": total_write_coverage / num_batches if num_batches else 0.0,
        "confidence_mean": total_confidence_mean / num_batches if num_batches else 0.0,
    }
    return metrics, first_example or {}


def train_on_exported_clips(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    clip_paths = load_manifest(args.manifest)
    train_paths, val_paths = split_clip_paths(clip_paths, val_ratio=args.val_ratio, seed=args.seed)
    train_dataset = MemoryConditionedClipWindowDataset(
        clip_paths=train_paths,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        max_windows_per_clip=args.max_train_windows_per_clip,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
    )
    val_dataset = MemoryConditionedClipWindowDataset(
        clip_paths=val_paths,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        max_windows_per_clip=args.max_val_windows_per_clip,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MemoryConditionedWorldModel(hidden_channels=args.hidden_channels, enable_uncertainty=args.enable_uncertainty).to(device)
    if args.warm_start_memory_checkpoint is not None:
        warm_start_report = warm_start_from_checkpoint(model, args.warm_start_memory_checkpoint)
    else:
        warm_start_report = warm_start_from_nomemory_checkpoint(model, args.warm_start_nomemory_checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float | int]] = []
    global_step = 0
    best_val_l1 = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_objective = 0.0
        running_l1 = 0.0
        running_psnr = 0.0
        running_dynamic = 0.0
        running_depth = 0.0
        running_memory_covered = 0.0
        running_memory_render = 0.0
        running_uncertainty_nll = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            model_outputs = model(
                batch["context_rgb"],
                batch["context_poses"],
                batch["target_poses"],
                batch["memory_condition"],
            )
            pred_rgb, pred_depth = model_outputs[:2]
            pred_log_variance = model_outputs[2] if len(model_outputs) > 2 else None
            objective, full_l1, dynamic_l1, depth_l1, memory_covered_l1, memory_render_l1, uncertainty_nll = compute_loss(
                pred_rgb=pred_rgb,
                pred_depth=pred_depth,
                pred_log_variance=pred_log_variance,
                target_rgb=batch["target_rgb"],
                target_depth=batch["target_depth"],
                context_rgb=batch["context_rgb"],
                memory_render_rgb=batch["memory_render_rgb"],
                memory_render_mask=batch["memory_render_mask"],
                motion_threshold=args.motion_threshold,
                dynamic_loss_weight=args.dynamic_loss_weight,
                depth_loss_weight=args.depth_loss_weight,
                memory_covered_loss_weight=args.memory_covered_loss_weight,
                memory_render_loss_weight=args.memory_render_loss_weight,
                uncertainty_loss_weight=args.uncertainty_loss_weight if args.enable_uncertainty else 0.0,
            )
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

            running_objective += float(objective.detach())
            running_l1 += float(full_l1.detach())
            running_psnr += float(psnr(pred_rgb.detach(), batch["target_rgb"]))
            running_dynamic += float(dynamic_l1.detach())
            running_depth += float(depth_l1.detach())
            running_memory_covered += float(memory_covered_l1.detach())
            running_memory_render += float(memory_render_l1.detach())
            running_uncertainty_nll += float(uncertainty_nll.detach())
            num_batches += 1
            global_step += 1
            if args.steps is not None and global_step >= args.steps:
                break

        train_metrics = {
            "train_objective": running_objective / max(num_batches, 1),
            "train_l1": running_l1 / max(num_batches, 1),
            "train_psnr": running_psnr / max(num_batches, 1),
            "train_dynamic_l1": running_dynamic / max(num_batches, 1),
            "train_depth_l1": running_depth / max(num_batches, 1),
            "train_memory_covered_l1": running_memory_covered / max(num_batches, 1),
            "train_memory_render_l1": running_memory_render / max(num_batches, 1),
            "train_uncertainty_nll": running_uncertainty_nll / max(num_batches, 1),
        }
        val_metrics, example = evaluate_model(
            model,
            val_loader,
            device,
            motion_threshold=args.motion_threshold,
            enable_uncertainty=args.enable_uncertainty,
            context_frames=args.context_frames,
            predict_frames=args.predict_frames,
            image_size=args.image_size,
            memory_grid_resolution=tuple(args.memory_grid_resolution),
            memory_stride=args.memory_stride,
            memory_splat_radius=args.memory_splat_radius,
            write_confidence_threshold=args.write_confidence_threshold,
            confidence_gamma=args.confidence_gamma,
        )
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:02d} train_obj={train_metrics['train_objective']:.4f} train_l1={train_metrics['train_l1']:.4f} "
            f"train_dyn={train_metrics['train_dynamic_l1']:.4f} train_cov={train_metrics['train_memory_covered_l1']:.4f} "
            f"train_depth={train_metrics['train_depth_l1']:.4f} "
            f"val_l1={val_metrics['model_l1']:.4f} val_dyn={val_metrics['model_dynamic_l1']:.4f} "
            f"val_cov={val_metrics['model_memory_covered_l1']:.4f} base_cov={val_metrics['baseline_memory_covered_l1']:.4f} "
            f"baseline_l1={val_metrics['baseline_l1']:.4f} baseline_dyn={val_metrics['baseline_dynamic_l1']:.4f} "
            f"memory_cov={val_metrics['memory_coverage']:.4f} write_cov={val_metrics['write_coverage']:.4f} "
            f"unc_corr={val_metrics['uncertainty_error_corr']:.4f}"
        )

        if val_metrics["model_l1"] < best_val_l1:
            best_val_l1 = val_metrics["model_l1"]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_dir / "memory_model_best.pt")
            save_example_artifacts(example, args.output_dir, "best_val")

        if args.steps is not None and global_step >= args.steps:
            break

    final_val_metrics, example = evaluate_model(
        model,
        val_loader,
        device,
        motion_threshold=args.motion_threshold,
        enable_uncertainty=args.enable_uncertainty,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
        write_confidence_threshold=args.write_confidence_threshold,
        confidence_gamma=args.confidence_gamma,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "memory_model_last.pt")
    save_example_artifacts(example, args.output_dir, "final_val")

    summary: dict[str, object] = {
        "device": str(device),
        "source": "npz",
        "epochs_completed": len(history),
        "global_steps": global_step,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "context_frames": args.context_frames,
        "predict_frames": args.predict_frames,
        "image_size": args.image_size,
        "hidden_channels": args.hidden_channels,
        "max_train_windows_per_clip": args.max_train_windows_per_clip,
        "max_val_windows_per_clip": args.max_val_windows_per_clip,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "motion_threshold": args.motion_threshold,
        "dynamic_loss_weight": args.dynamic_loss_weight,
        "depth_loss_weight": args.depth_loss_weight,
        "memory_covered_loss_weight": args.memory_covered_loss_weight,
        "memory_render_loss_weight": args.memory_render_loss_weight,
        "enable_uncertainty": args.enable_uncertainty,
        "uncertainty_loss_weight": args.uncertainty_loss_weight,
        "write_confidence_threshold": args.write_confidence_threshold,
        "confidence_gamma": args.confidence_gamma,
        "memory_grid_resolution": list(args.memory_grid_resolution),
        "memory_stride": args.memory_stride,
        "memory_splat_radius": args.memory_splat_radius,
        "warm_start": warm_start_report,
        "num_train_clips": len(train_paths),
        "num_val_clips": len(val_paths),
        "num_train_windows": len(train_dataset),
        "num_val_windows": len(val_dataset),
        "best_val_l1": best_val_l1,
        "final_val": final_val_metrics,
        "history": history,
        "example_clip_path": example["clip_path"],
        "example_start_frame": example["start_frame"],
        "variant": "memory_uncertainty_convgru" if args.enable_uncertainty else "memory",
        "model_beats_baseline_on_val_l1": final_val_metrics["model_l1"] < final_val_metrics["baseline_l1"],
        "model_beats_baseline_on_val_dynamic_l1": final_val_metrics["model_dynamic_l1"] < final_val_metrics["baseline_dynamic_l1"],
        "model_beats_baseline_on_val_memory_covered_l1": final_val_metrics["model_memory_covered_l1"] < final_val_metrics["baseline_memory_covered_l1"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"using device={device}")
    metrics = train_on_exported_clips(args, device)
    print(json.dumps(metrics, indent=2))
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
