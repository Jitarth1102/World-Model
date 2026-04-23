#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset, load_manifest, split_clip_paths
from world_model.eval.metrics_image import masked_l1, motion_mask_from_last_context, psnr
from world_model.models.diffusion import ConditionalVideoDiffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight diffusion predictor on exported MOVi windows.")
    parser.add_argument("--variant", choices=["no_memory", "memory"], default="no_memory")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/movi_a_128_subset50/manifest.json"))
    parser.add_argument("--steps", type=int, default=None, help="Optional hard cap on optimizer steps.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-channels", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--sample-steps-eval", type=int, default=16)
    parser.add_argument("--eval-max-batches", type=int, default=2)
    parser.add_argument("--max-train-windows-per-clip", type=int, default=4)
    parser.add_argument("--max-val-windows-per-clip", type=int, default=2)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--memory-grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--memory-splat-radius", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train_diffusion_nomemory_real_movia_subset50_v1"))
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


def baseline_copy_last(context_rgb: torch.Tensor, predict_frames: int) -> torch.Tensor:
    return context_rgb[:, -1:].repeat(1, predict_frames, 1, 1, 1)


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


@torch.no_grad()
def evaluate_model(
    model: ConditionalVideoDiffusion,
    loader: DataLoader,
    device: torch.device,
    *,
    variant: str,
    motion_threshold: float,
    sample_steps: int,
    eval_max_batches: int | None,
) -> tuple[dict[str, float], dict[str, torch.Tensor | str | int | float]]:
    model.eval()
    total_model_l1 = 0.0
    total_model_psnr = 0.0
    total_base_l1 = 0.0
    total_base_psnr = 0.0
    total_model_dynamic_l1 = 0.0
    total_base_dynamic_l1 = 0.0
    total_model_memory_covered_l1 = 0.0
    total_base_memory_covered_l1 = 0.0
    total_motion_fraction = 0.0
    total_memory_coverage = 0.0
    total_memory_occupancy = 0.0
    total_memory_render_l1 = 0.0
    num_batches = 0
    first_example: dict[str, torch.Tensor | str | int | float] | None = None

    for batch_idx, batch in enumerate(loader):
        if eval_max_batches is not None and batch_idx >= eval_max_batches:
            break
        batch = move_batch_to_device(batch, device)
        prediction, _ = model.sample(
            context_rgb=batch["context_rgb"],
            context_poses=batch["context_poses"],
            target_poses=batch["target_poses"],
            memory_condition=batch["memory_condition"] if variant == "memory" else None,
            sample_steps=sample_steps,
            eta=0.0,
            return_intermediates=False,
        )
        baseline = baseline_copy_last(batch["context_rgb"], batch["target_rgb"].shape[1])
        dynamic_mask = motion_mask_from_last_context(batch["context_rgb"], batch["target_rgb"], threshold=motion_threshold)
        memory_mask = batch["memory_render_mask"]

        total_model_l1 += float(masked_l1(prediction, batch["target_rgb"]))
        total_model_psnr += float(psnr(prediction, batch["target_rgb"]))
        total_base_l1 += float(masked_l1(baseline, batch["target_rgb"]))
        total_base_psnr += float(psnr(baseline, batch["target_rgb"]))
        total_model_dynamic_l1 += float(masked_l1(prediction, batch["target_rgb"], dynamic_mask))
        total_base_dynamic_l1 += float(masked_l1(baseline, batch["target_rgb"], dynamic_mask))
        total_model_memory_covered_l1 += float(masked_l1(prediction, batch["target_rgb"], memory_mask))
        total_base_memory_covered_l1 += float(masked_l1(baseline, batch["target_rgb"], memory_mask))
        total_motion_fraction += float(dynamic_mask.mean())
        total_memory_coverage += float(batch["memory_render_coverage"].float().mean())
        total_memory_occupancy += float(batch["memory_occupancy_fraction"].float().mean())
        total_memory_render_l1 += float(batch["memory_render_l1_covered"].float().mean())
        num_batches += 1

        if first_example is None:
            first_example = {
                "context_rgb": batch["context_rgb"][0].detach().cpu(),
                "target_rgb": batch["target_rgb"][0].detach().cpu(),
                "prediction": prediction[0].detach().cpu(),
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
        "model_memory_covered_l1": total_model_memory_covered_l1 / num_batches,
        "baseline_memory_covered_l1": total_base_memory_covered_l1 / num_batches,
        "motion_fraction": total_motion_fraction / num_batches,
        "memory_coverage": total_memory_coverage / num_batches,
        "memory_occupancy_fraction": total_memory_occupancy / num_batches,
        "memory_render_l1_covered": total_memory_render_l1 / num_batches,
    }
    return metrics, first_example or {}


def build_dataset(args: argparse.Namespace, clip_paths: list[Path], *, max_windows_per_clip: int | None) -> MemoryConditionedClipWindowDataset:
    return MemoryConditionedClipWindowDataset(
        clip_paths=clip_paths,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        max_windows_per_clip=max_windows_per_clip,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
    )


def save_checkpoint(model: ConditionalVideoDiffusion, args: argparse.Namespace, path: Path) -> None:
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "variant": args.variant,
            "context_frames": args.context_frames,
            "predict_frames": args.predict_frames,
            "image_size": args.image_size,
            "model_channels": args.model_channels,
            "diffusion_steps": args.diffusion_steps,
            "sample_steps_eval": args.sample_steps_eval,
        },
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"using device={device}")

    clip_paths = load_manifest(args.manifest)
    train_paths, val_paths = split_clip_paths(clip_paths, val_ratio=args.val_ratio, seed=args.seed)
    train_dataset = build_dataset(args, train_paths, max_windows_per_clip=args.max_train_windows_per_clip)
    val_dataset = build_dataset(args, val_paths, max_windows_per_clip=args.max_val_windows_per_clip)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConditionalVideoDiffusion(
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        variant=args.variant,
        model_channels=args.model_channels,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float | int]] = []
    global_step = 0
    best_val_l1 = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_noise_abs = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            loss, aux = model.training_loss(
                context_rgb=batch["context_rgb"],
                target_rgb=batch["target_rgb"],
                context_poses=batch["context_poses"],
                target_poses=batch["target_poses"],
                memory_condition=batch["memory_condition"] if args.variant == "memory" else None,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach())
            running_noise_abs += aux["predicted_noise_abs_mean"]
            num_batches += 1
            global_step += 1
            if args.steps is not None and global_step >= args.steps:
                break

        train_metrics = {
            "train_objective": running_loss / max(num_batches, 1),
            "train_predicted_noise_abs_mean": running_noise_abs / max(num_batches, 1),
        }
        val_metrics, example = evaluate_model(
            model,
            val_loader,
            device,
            variant=args.variant,
            motion_threshold=args.motion_threshold,
            sample_steps=args.sample_steps_eval,
            eval_max_batches=args.eval_max_batches,
        )
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:02d} train_obj={train_metrics['train_objective']:.4f} "
            f"val_l1={val_metrics['model_l1']:.4f} val_dyn={val_metrics['model_dynamic_l1']:.4f} "
            f"val_cov={val_metrics['model_memory_covered_l1']:.4f} baseline_l1={val_metrics['baseline_l1']:.4f}"
        )

        if val_metrics["model_l1"] < best_val_l1:
            best_val_l1 = val_metrics["model_l1"]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, args, args.output_dir / "diffusion_model_best.pt")
            save_comparison(example["context_rgb"], example["target_rgb"], example["prediction"], example["baseline"], args.output_dir / "best_val_comparison.png")
            save_strip(list(example["target_rgb"]), args.output_dir / "best_val_target_strip.png")
            save_strip(list(example["prediction"]), args.output_dir / "best_val_prediction_strip.png")
            save_strip(list(example["baseline"]), args.output_dir / "best_val_baseline_strip.png")
            save_strip(list(example["memory_render_rgb"]), args.output_dir / "best_val_memory_render_strip.png")
            save_mask_strip(list(example["motion_mask"]), args.output_dir / "best_val_motion_mask_strip.png")
            save_mask_strip(list(example["memory_render_mask"]), args.output_dir / "best_val_memory_mask_strip.png")

        if args.steps is not None and global_step >= args.steps:
            break

    final_val_metrics, example = evaluate_model(
        model,
        val_loader,
        device,
        variant=args.variant,
        motion_threshold=args.motion_threshold,
        sample_steps=args.sample_steps_eval,
        eval_max_batches=args.eval_max_batches,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args, args.output_dir / "diffusion_model_last.pt")
    save_comparison(example["context_rgb"], example["target_rgb"], example["prediction"], example["baseline"], args.output_dir / "final_val_comparison.png")
    save_strip(list(example["memory_render_rgb"]), args.output_dir / "final_val_memory_render_strip.png")
    save_mask_strip(list(example["memory_render_mask"]), args.output_dir / "final_val_memory_mask_strip.png")
    save_mask_strip(list(example["motion_mask"]), args.output_dir / "final_val_motion_mask_strip.png")

    summary: dict[str, object] = {
        "device": str(device),
        "model_type": "diffusion",
        "variant": args.variant,
        "epochs_completed": len(history),
        "global_steps": global_step,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "context_frames": args.context_frames,
        "predict_frames": args.predict_frames,
        "image_size": args.image_size,
        "model_channels": args.model_channels,
        "diffusion_steps": args.diffusion_steps,
        "sample_steps_eval": args.sample_steps_eval,
        "eval_max_batches": args.eval_max_batches,
        "max_train_windows_per_clip": args.max_train_windows_per_clip,
        "max_val_windows_per_clip": args.max_val_windows_per_clip,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "motion_threshold": args.motion_threshold,
        "memory_grid_resolution": list(args.memory_grid_resolution),
        "memory_stride": args.memory_stride,
        "memory_splat_radius": args.memory_splat_radius,
        "num_train_clips": len(train_paths),
        "num_val_clips": len(val_paths),
        "num_train_windows": len(train_dataset),
        "num_val_windows": len(val_dataset),
        "best_val_l1": best_val_l1,
        "final_val": final_val_metrics,
        "history": history,
        "example_clip_path": example["clip_path"],
        "example_start_frame": example["start_frame"],
        "model_beats_baseline_on_val_l1": final_val_metrics["model_l1"] < final_val_metrics["baseline_l1"],
        "model_beats_baseline_on_val_dynamic_l1": final_val_metrics["model_dynamic_l1"] < final_val_metrics["baseline_dynamic_l1"],
        "model_beats_baseline_on_val_memory_covered_l1": final_val_metrics["model_memory_covered_l1"] < final_val_metrics["baseline_memory_covered_l1"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
