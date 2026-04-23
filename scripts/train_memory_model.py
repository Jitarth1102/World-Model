#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from world_model.data.synthetic import make_synthetic_clip
from world_model.eval.metrics_image import masked_l1, psnr
from world_model.memory.oracle_writer import accumulate_clip_into_memory
from world_model.memory.renderer import render_memory_view
from world_model.memory.voxel_grid import VoxelGridSpec
from world_model.models.world_model import MemoryConditionedWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny synthetic overfit loop for the memory-conditioned model.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train_memory_model"))
    return parser.parse_args()


def build_memory_conditions(clip, context_frames: int, predict_frames: int) -> torch.Tensor:
    spec = VoxelGridSpec(bounds_min=(-2.0, -1.5, -2.0), bounds_max=(2.0, 1.8, 2.0), resolution=(48, 40, 48))
    memory, _ = accumulate_clip_into_memory(clip, context_frames=context_frames, memory_spec=spec)
    conditions = []
    for frame_idx in range(context_frames, context_frames + predict_frames):
        rendered = render_memory_view(memory, clip.poses[frame_idx], clip.intrinsics, splat_radius=1)
        rgba = torch.cat(
            [
                torch.from_numpy(rendered.rgb).permute(2, 0, 1),
                torch.from_numpy(rendered.mask.astype("float32")).unsqueeze(0),
            ],
            dim=0,
        ).float()
        conditions.append(rgba)
    return torch.stack(conditions, dim=0)


def save_strip(target_rgb: torch.Tensor, predicted_rgb: torch.Tensor, path: Path) -> None:
    frames = []
    target_np = (target_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
    pred_np = (predicted_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
    for idx in range(target_np.shape[0]):
        frames.append(target_np[idx].transpose(1, 2, 0))
        frames.append(pred_np[idx].transpose(1, 2, 0))
    strip = Image.new("RGB", (frames[0].shape[1] * len(frames), frames[0].shape[0]))
    for idx, frame in enumerate(frames):
        strip.paste(Image.fromarray(frame), (idx * frame.shape[1], 0))
    strip.save(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip = make_synthetic_clip(num_frames=args.context_frames + args.predict_frames, image_size=args.image_size)

    context_rgb = torch.from_numpy(clip.video[: args.context_frames]).float().permute(0, 3, 1, 2) / 255.0
    target_rgb = torch.from_numpy(clip.video[args.context_frames : args.context_frames + args.predict_frames]).float().permute(0, 3, 1, 2) / 255.0
    target_depth = torch.from_numpy(clip.depth[args.context_frames : args.context_frames + args.predict_frames]).float().unsqueeze(1)
    context_poses = torch.from_numpy(clip.poses[: args.context_frames]).float()
    target_poses = torch.from_numpy(clip.poses[args.context_frames : args.context_frames + args.predict_frames]).float()
    memory_rgbm = build_memory_conditions(clip, args.context_frames, args.predict_frames)

    model = MemoryConditionedWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    context_rgb_b = context_rgb.unsqueeze(0)
    context_poses_b = context_poses.unsqueeze(0)
    target_poses_b = target_poses.unsqueeze(0)
    target_rgb_b = target_rgb.unsqueeze(0)
    target_depth_b = target_depth.unsqueeze(0)
    memory_rgbm_b = memory_rgbm.unsqueeze(0)

    for step in range(args.steps):
        pred_rgb, pred_depth = model(context_rgb_b, context_poses_b, target_poses_b, memory_rgbm_b)
        loss = masked_l1(pred_rgb, target_rgb_b) + 0.1 * masked_l1(pred_depth, target_depth_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == args.steps - 1:
            with torch.no_grad():
                current_psnr = float(psnr(pred_rgb, target_rgb_b))
            print(f"step={step:04d} loss={float(loss.detach()):.5f} psnr={current_psnr:.2f}")

    with torch.no_grad():
        pred_rgb, _ = model(context_rgb_b, context_poses_b, target_poses_b, memory_rgbm_b)
    save_strip(target_rgb, pred_rgb[0], args.output_dir / "prediction_strip.png")
    torch.save(model.state_dict(), args.output_dir / "memory_model.pt")
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
