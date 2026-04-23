#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from world_model.data.synthetic import make_synthetic_clip
from world_model.eval.metrics_image import masked_l1, psnr
from world_model.models.convgru_predictor import NoMemoryPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny synthetic overfit loop for the no-memory predictor.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train_nomemory"))
    return parser.parse_args()


def save_strip(target: torch.Tensor, prediction: torch.Tensor, path: Path) -> None:
    frames = []
    target_np = (target.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
    pred_np = (prediction.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
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
    context_poses = torch.from_numpy(clip.poses[: args.context_frames]).float()
    target_poses = torch.from_numpy(clip.poses[args.context_frames : args.context_frames + args.predict_frames]).float()

    model = NoMemoryPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    context_rgb_b = context_rgb.unsqueeze(0)
    target_rgb_b = target_rgb.unsqueeze(0)
    context_poses_b = context_poses.unsqueeze(0)
    target_poses_b = target_poses.unsqueeze(0)

    for step in range(args.steps):
        prediction = model(context_rgb_b, context_poses_b, target_poses_b)
        loss = masked_l1(prediction, target_rgb_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == args.steps - 1:
            with torch.no_grad():
                current_psnr = float(psnr(prediction, target_rgb_b))
            print(f"step={step:04d} loss={float(loss.detach()):.5f} psnr={current_psnr:.2f}")

    with torch.no_grad():
        prediction = model(context_rgb_b, context_poses_b, target_poses_b)[0]
    save_strip(target_rgb, prediction, args.output_dir / "prediction_strip.png")
    torch.save(model.state_dict(), args.output_dir / "nomemory_model.pt")
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
