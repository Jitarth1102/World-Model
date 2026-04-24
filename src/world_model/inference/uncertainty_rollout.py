from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from world_model.memory.update_rules import (
    build_initial_memory_from_clip,
    depth_scale_from_clip,
    render_memory_condition,
    scale_intrinsics,
    write_prediction_to_memory,
)
from world_model.types import CameraIntrinsics, ClipSample
from world_model.uncertainty.confidence import uncertainty_to_confidence, variance_to_confidence


@dataclass(frozen=True)
class WindowTensors:
    clip: ClipSample
    clip_path: str
    start_frame: int
    context_rgb: torch.Tensor
    target_rgb: torch.Tensor
    target_depth: torch.Tensor
    context_poses: torch.Tensor
    target_poses: torch.Tensor
    depth_scale: float
    intrinsics: CameraIntrinsics


def _resize_sequence(tensor: torch.Tensor, image_size: int, mode: str = "bilinear") -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == (image_size, image_size):
        return tensor
    kwargs = {"size": (image_size, image_size), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)


def load_window_tensors(
    *,
    clip_path: str | Path,
    start_frame: int,
    context_frames: int,
    predict_frames: int,
    image_size: int,
) -> WindowTensors:
    clip = ClipSample.load_npz(clip_path)
    context_end = start_frame + context_frames
    target_end = context_end + predict_frames
    depth_scale = depth_scale_from_clip(clip)

    context_rgb = torch.from_numpy(clip.video[start_frame:context_end]).float().permute(0, 3, 1, 2) / 255.0
    target_rgb = torch.from_numpy(clip.video[context_end:target_end]).float().permute(0, 3, 1, 2) / 255.0
    target_depth = torch.from_numpy(clip.depth[context_end:target_end] / depth_scale).float().unsqueeze(1)
    context_rgb = _resize_sequence(context_rgb, image_size=image_size, mode="bilinear")
    target_rgb = _resize_sequence(target_rgb, image_size=image_size, mode="bilinear")
    target_depth = _resize_sequence(target_depth, image_size=image_size, mode="bilinear")

    return WindowTensors(
        clip=clip,
        clip_path=str(clip_path),
        start_frame=start_frame,
        context_rgb=context_rgb.unsqueeze(0),
        target_rgb=target_rgb.unsqueeze(0),
        target_depth=target_depth.unsqueeze(0),
        context_poses=torch.from_numpy(clip.poses[start_frame:context_end]).float().unsqueeze(0),
        target_poses=torch.from_numpy(clip.poses[context_end:target_end]).float().unsqueeze(0),
        depth_scale=depth_scale,
        intrinsics=scale_intrinsics(clip.intrinsics, target_width=image_size, target_height=image_size),
    )


def rollout_convgru_uncertainty(
    *,
    model,
    clip_path: str | Path,
    start_frame: int,
    context_frames: int,
    predict_frames: int,
    image_size: int,
    device: torch.device,
    memory_grid_resolution: tuple[int, int, int],
    memory_stride: int,
    memory_splat_radius: int,
    confidence_threshold: float | None,
    confidence_gamma: float = 1.0,
) -> dict[str, object]:
    window = load_window_tensors(
        clip_path=clip_path,
        start_frame=start_frame,
        context_frames=context_frames,
        predict_frames=predict_frames,
        image_size=image_size,
    )
    memory = build_initial_memory_from_clip(
        window.clip,
        start_frame=start_frame,
        context_frames=context_frames,
        grid_resolution=memory_grid_resolution,
        stride=memory_stride,
    )

    context_rgb = window.context_rgb.to(device)
    context_poses = window.context_poses.to(device)
    hidden, anchor_pose, prev_frame = model.initialize_rollout(context_rgb, context_poses)

    pred_rgb_frames = []
    pred_depth_frames = []
    uncertainty_frames = []
    confidence_frames = []
    write_mask_frames = []
    memory_render_rgb_frames = []
    memory_render_mask_frames = []
    write_fractions = []
    confidence_means = []

    for step_idx in range(predict_frames):
        pose_np = window.target_poses[0, step_idx].numpy()
        rendered = render_memory_condition(
            memory,
            pose_np,
            window.intrinsics,
            depth_scale=window.depth_scale,
            splat_radius=memory_splat_radius,
        )
        memory_render_rgb_frames.append(torch.from_numpy(rendered["rgb"]).permute(2, 0, 1).float())
        memory_render_mask_frames.append(torch.from_numpy(rendered["mask"]).float().unsqueeze(0))
        memory_condition_step = torch.from_numpy(rendered["condition"]).float().unsqueeze(0).to(device)
        hidden, prev_frame, pred_depth, log_variance = model.predict_step(
            hidden=hidden,
            prev_frame=prev_frame,
            anchor_pose=anchor_pose,
            target_pose=window.target_poses[:, step_idx].to(device),
            memory_condition_step=memory_condition_step,
        )
        if log_variance is None:
            log_variance = torch.zeros_like(pred_depth)
        confidence = uncertainty_to_confidence(log_variance, gamma=confidence_gamma)
        rgb_np = prev_frame[0].detach().cpu().permute(1, 2, 0).numpy()
        depth_np = pred_depth[0, 0].detach().cpu().clamp(min=0.0).numpy() * window.depth_scale
        confidence_np = confidence[0, 0].detach().cpu().numpy()
        write_mask_np, write_stats = write_prediction_to_memory(
            memory,
            rgb_frame=rgb_np,
            depth_frame=depth_np,
            pose=pose_np,
            intrinsics=window.intrinsics,
            confidence_map=confidence_np,
            confidence_threshold=confidence_threshold,
            stride=memory_stride,
        )
        pred_rgb_frames.append(prev_frame[0].detach().cpu())
        pred_depth_frames.append(pred_depth[0].detach().cpu())
        uncertainty_frames.append(torch.exp(log_variance[0]).detach().cpu())
        confidence_frames.append(confidence[0].detach().cpu())
        write_mask_frames.append(torch.from_numpy(write_mask_np).float().unsqueeze(0))
        write_fractions.append(write_stats.write_fraction)
        confidence_means.append(write_stats.mean_confidence)

    return {
        "window": window,
        "prediction": torch.stack(pred_rgb_frames, dim=0),
        "pred_depth": torch.stack(pred_depth_frames, dim=0),
        "uncertainty": torch.stack(uncertainty_frames, dim=0),
        "confidence": torch.stack(confidence_frames, dim=0),
        "write_mask": torch.stack(write_mask_frames, dim=0),
        "memory_render_rgb": torch.stack(memory_render_rgb_frames, dim=0),
        "memory_render_mask": torch.stack(memory_render_mask_frames, dim=0),
        "write_coverage": float(np.mean(write_fractions)) if write_fractions else 0.0,
        "confidence_mean": float(np.mean(confidence_means)) if confidence_means else 0.0,
    }


def rollout_diffusion_uncertainty(
    *,
    model,
    clip_path: str | Path,
    start_frame: int,
    context_frames: int,
    predict_frames: int,
    image_size: int,
    device: torch.device,
    memory_grid_resolution: tuple[int, int, int],
    memory_stride: int,
    memory_splat_radius: int,
    confidence_threshold: float | None,
    sample_steps: int,
    uncertainty_samples: int,
    confidence_gamma: float = 1.0,
) -> dict[str, object]:
    window = load_window_tensors(
        clip_path=clip_path,
        start_frame=start_frame,
        context_frames=context_frames,
        predict_frames=predict_frames,
        image_size=image_size,
    )
    memory = build_initial_memory_from_clip(
        window.clip,
        start_frame=start_frame,
        context_frames=context_frames,
        grid_resolution=memory_grid_resolution,
        stride=memory_stride,
    )

    memory_render_rgb_frames = []
    memory_render_mask_frames = []
    memory_condition_steps = []
    for step_idx in range(predict_frames):
        pose_np = window.target_poses[0, step_idx].numpy()
        rendered = render_memory_condition(
            memory,
            pose_np,
            window.intrinsics,
            depth_scale=window.depth_scale,
            splat_radius=memory_splat_radius,
        )
        memory_render_rgb_frames.append(torch.from_numpy(rendered["rgb"]).permute(2, 0, 1).float())
        memory_render_mask_frames.append(torch.from_numpy(rendered["mask"]).float().unsqueeze(0))
        memory_condition_steps.append(torch.from_numpy(rendered["condition"]).float())
    memory_condition = torch.stack(memory_condition_steps, dim=0).unsqueeze(0).to(device)
    context_rgb = window.context_rgb.to(device)
    context_poses = window.context_poses.to(device)
    target_poses = window.target_poses.to(device)

    sample_predictions = []
    for _ in range(uncertainty_samples):
        prediction, _ = model.sample(
            context_rgb=context_rgb,
            context_poses=context_poses,
            target_poses=target_poses,
            memory_condition=memory_condition,
            sample_steps=sample_steps,
            eta=0.0,
            return_intermediates=False,
        )
        sample_predictions.append(prediction)

    stacked_predictions = torch.stack(sample_predictions, dim=0)
    mean_prediction = stacked_predictions.mean(dim=0)
    variance_map = stacked_predictions.var(dim=0, unbiased=False).mean(dim=2, keepdim=True)
    confidence = variance_to_confidence(variance_map, gamma=confidence_gamma)

    write_mask_frames = []
    write_fractions = []
    confidence_means = []
    for step_idx in range(predict_frames):
        pose_np = window.target_poses[0, step_idx].numpy()
        rgb_np = mean_prediction[0, step_idx].detach().cpu().permute(1, 2, 0).numpy()
        depth_np = window.target_depth[0, step_idx, 0].detach().cpu().numpy() * window.depth_scale
        confidence_np = confidence[0, step_idx, 0].detach().cpu().numpy()
        write_mask_np, write_stats = write_prediction_to_memory(
            memory,
            rgb_frame=rgb_np,
            depth_frame=depth_np,
            pose=pose_np,
            intrinsics=window.intrinsics,
            confidence_map=confidence_np,
            confidence_threshold=confidence_threshold,
            stride=memory_stride,
        )
        write_mask_frames.append(torch.from_numpy(write_mask_np).float().unsqueeze(0))
        write_fractions.append(write_stats.write_fraction)
        confidence_means.append(write_stats.mean_confidence)

    return {
        "window": window,
        "prediction": mean_prediction[0].detach().cpu(),
        "pred_depth": window.target_depth[0].detach().cpu(),
        "uncertainty": variance_map[0].detach().cpu(),
        "confidence": confidence[0].detach().cpu(),
        "write_mask": torch.stack(write_mask_frames, dim=0),
        "memory_render_rgb": torch.stack(memory_render_rgb_frames, dim=0),
        "memory_render_mask": torch.stack(memory_render_mask_frames, dim=0),
        "write_coverage": float(np.mean(write_fractions)) if write_fractions else 0.0,
        "confidence_mean": float(np.mean(confidence_means)) if confidence_means else 0.0,
    }
