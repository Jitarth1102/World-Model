#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from world_model.data.clip_dataset import ExportedClipWindowDataset, load_manifest, split_clip_paths
from world_model.data.synthetic import make_synthetic_clip
from world_model.eval.metrics_image import masked_l1, motion_mask_from_last_context, psnr
from world_model.models.convgru_predictor import NoMemoryPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 3 no-memory predictor.")
    parser.add_argument("--source", choices=["synthetic", "npz"], default="npz")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/movi_a_128_subset50/manifest.json"))
    parser.add_argument("--steps", type=int, default=None, help="Optional hard cap on optimizer steps.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--max-train-windows-per-clip", type=int, default=None)
    parser.add_argument("--max-val-windows-per-clip", type=int, default=4)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--dynamic-loss-weight", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train_nomemory_real"))
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


def compute_loss(
    prediction: torch.Tensor,
    target_rgb: torch.Tensor,
    context_rgb: torch.Tensor,
    motion_threshold: float,
    dynamic_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_l1 = masked_l1(prediction, target_rgb)
    dynamic_mask = motion_mask_from_last_context(context_rgb, target_rgb, threshold=motion_threshold)
    dynamic_l1 = masked_l1(prediction, target_rgb, dynamic_mask)
    objective = full_l1 + dynamic_loss_weight * dynamic_l1
    return objective, full_l1, dynamic_l1


def train_on_synthetic(args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    clip = make_synthetic_clip(num_frames=args.context_frames + args.predict_frames, image_size=args.image_size)
    context_rgb = torch.from_numpy(clip.video[: args.context_frames]).float().permute(0, 3, 1, 2) / 255.0
    target_rgb = torch.from_numpy(clip.video[args.context_frames : args.context_frames + args.predict_frames]).float().permute(0, 3, 1, 2) / 255.0
    context_poses = torch.from_numpy(clip.poses[: args.context_frames]).float()
    target_poses = torch.from_numpy(clip.poses[args.context_frames : args.context_frames + args.predict_frames]).float()

    model = NoMemoryPredictor(hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    context_rgb_b = context_rgb.unsqueeze(0).to(device)
    target_rgb_b = target_rgb.unsqueeze(0).to(device)
    context_poses_b = context_poses.unsqueeze(0).to(device)
    target_poses_b = target_poses.unsqueeze(0).to(device)
    baseline_rgb = baseline_copy_last(context_rgb_b, args.predict_frames)

    max_steps = args.steps or 100
    for step in range(max_steps):
        prediction = model(context_rgb_b, context_poses_b, target_poses_b)
        objective, full_l1, dynamic_l1 = compute_loss(
            prediction=prediction,
            target_rgb=target_rgb_b,
            context_rgb=context_rgb_b,
            motion_threshold=args.motion_threshold,
            dynamic_loss_weight=args.dynamic_loss_weight,
        )
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        if step % 20 == 0 or step == max_steps - 1:
            print(
                f"step={step:04d} objective={float(objective.detach()):.5f} "
                f"full_l1={float(full_l1.detach()):.5f} "
                f"dynamic_l1={float(dynamic_l1.detach()):.5f} psnr={float(psnr(prediction, target_rgb_b)):.2f}"
            )

    with torch.no_grad():
        prediction = model(context_rgb_b, context_poses_b, target_poses_b)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison(context_rgb, target_rgb, prediction[0].cpu(), baseline_rgb[0].cpu(), args.output_dir / "synthetic_comparison.png")
    torch.save(model.state_dict(), args.output_dir / "nomemory_model.pt")
    dynamic_mask = motion_mask_from_last_context(context_rgb_b, target_rgb_b, threshold=args.motion_threshold)
    metrics = {
        "train_l1": float(masked_l1(prediction, target_rgb_b)),
        "train_dynamic_l1": float(masked_l1(prediction, target_rgb_b, dynamic_mask)),
        "train_psnr": float(psnr(prediction, target_rgb_b)),
        "baseline_l1": float(masked_l1(baseline_rgb, target_rgb_b)),
        "baseline_dynamic_l1": float(masked_l1(baseline_rgb, target_rgb_b, dynamic_mask)),
        "baseline_psnr": float(psnr(baseline_rgb, target_rgb_b)),
        "motion_fraction": float(dynamic_mask.mean()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def move_batch_to_device(batch: dict[str, torch.Tensor | str | int], device: torch.device) -> dict[str, torch.Tensor | str | int]:
    moved: dict[str, torch.Tensor | str | int] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def evaluate_model(
    model: NoMemoryPredictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, torch.Tensor | str | int]]:
    model.eval()
    total_model_l1 = 0.0
    total_model_psnr = 0.0
    total_base_l1 = 0.0
    total_base_psnr = 0.0
    total_model_dynamic_l1 = 0.0
    total_base_dynamic_l1 = 0.0
    total_motion_fraction = 0.0
    num_batches = 0
    first_example: dict[str, torch.Tensor | str | int] | None = None

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        context_rgb = batch["context_rgb"]
        target_rgb = batch["target_rgb"]
        context_poses = batch["context_poses"]
        target_poses = batch["target_poses"]

        prediction = model(context_rgb, context_poses, target_poses)
        baseline = baseline_copy_last(context_rgb, target_rgb.shape[1])
        dynamic_mask = motion_mask_from_last_context(context_rgb, target_rgb)

        total_model_l1 += float(masked_l1(prediction, target_rgb))
        total_model_psnr += float(psnr(prediction, target_rgb))
        total_base_l1 += float(masked_l1(baseline, target_rgb))
        total_base_psnr += float(psnr(baseline, target_rgb))
        total_model_dynamic_l1 += float(masked_l1(prediction, target_rgb, dynamic_mask))
        total_base_dynamic_l1 += float(masked_l1(baseline, target_rgb, dynamic_mask))
        total_motion_fraction += float(dynamic_mask.mean())
        num_batches += 1

        if first_example is None:
            first_example = {
                "context_rgb": context_rgb[0].detach().cpu(),
                "target_rgb": target_rgb[0].detach().cpu(),
                "prediction": prediction[0].detach().cpu(),
                "baseline": baseline[0].detach().cpu(),
                "motion_mask": dynamic_mask[0].detach().cpu(),
                "clip_path": batch["clip_path"][0] if isinstance(batch["clip_path"], list) else batch["clip_path"],
                "start_frame": int(batch["start_frame"][0]) if isinstance(batch["start_frame"], torch.Tensor) else int(batch["start_frame"]),
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
        "motion_fraction": total_motion_fraction / num_batches,
    }
    return metrics, first_example or {}


def train_on_exported_clips(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    clip_paths = load_manifest(args.manifest)
    train_paths, val_paths = split_clip_paths(clip_paths, val_ratio=args.val_ratio, seed=args.seed)
    train_dataset = ExportedClipWindowDataset(
        clip_paths=train_paths,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        max_windows_per_clip=args.max_train_windows_per_clip,
    )
    val_dataset = ExportedClipWindowDataset(
        clip_paths=val_paths,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        max_windows_per_clip=args.max_val_windows_per_clip,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NoMemoryPredictor(hidden_channels=args.hidden_channels).to(device)
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
        num_batches = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            prediction = model(batch["context_rgb"], batch["context_poses"], batch["target_poses"])
            objective, full_l1, dynamic_l1 = compute_loss(
                prediction=prediction,
                target_rgb=batch["target_rgb"],
                context_rgb=batch["context_rgb"],
                motion_threshold=args.motion_threshold,
                dynamic_loss_weight=args.dynamic_loss_weight,
            )
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

            running_objective += float(objective.detach())
            running_l1 += float(full_l1.detach())
            running_psnr += float(psnr(prediction.detach(), batch["target_rgb"]))
            running_dynamic += float(dynamic_l1.detach())
            num_batches += 1
            global_step += 1
            if args.steps is not None and global_step >= args.steps:
                break

        train_metrics = {
            "train_objective": running_objective / max(num_batches, 1),
            "train_l1": running_l1 / max(num_batches, 1),
            "train_psnr": running_psnr / max(num_batches, 1),
            "train_dynamic_l1": running_dynamic / max(num_batches, 1),
        }
        val_metrics, example = evaluate_model(model, val_loader, device)
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:02d} train_obj={train_metrics['train_objective']:.4f} train_l1={train_metrics['train_l1']:.4f} "
            f"train_dyn={train_metrics['train_dynamic_l1']:.4f} "
            f"val_l1={val_metrics['model_l1']:.4f} val_dyn={val_metrics['model_dynamic_l1']:.4f} "
            f"baseline_l1={val_metrics['baseline_l1']:.4f} baseline_dyn={val_metrics['baseline_dynamic_l1']:.4f}"
        )

        if val_metrics["model_l1"] < best_val_l1:
            best_val_l1 = val_metrics["model_l1"]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_dir / "nomemory_model_best.pt")
            save_comparison(
                example["context_rgb"],
                example["target_rgb"],
                example["prediction"],
                example["baseline"],
                args.output_dir / "best_val_comparison.png",
            )
            save_strip(list(example["target_rgb"]), args.output_dir / "best_val_target_strip.png")
            save_strip(list(example["prediction"]), args.output_dir / "best_val_prediction_strip.png")
            save_strip(list(example["baseline"]), args.output_dir / "best_val_baseline_strip.png")

        if args.steps is not None and global_step >= args.steps:
            break

    final_val_metrics, example = evaluate_model(model, val_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "nomemory_model_last.pt")
    save_comparison(
        example["context_rgb"],
        example["target_rgb"],
        example["prediction"],
        example["baseline"],
        args.output_dir / "final_val_comparison.png",
    )

    summary: dict[str, object] = {
        "device": str(device),
        "source": args.source,
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
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"using device={device}")
    if args.source == "synthetic":
        metrics = train_on_synthetic(args, device)
    else:
        metrics = train_on_exported_clips(args, device)
    print(json.dumps(metrics, indent=2))
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
