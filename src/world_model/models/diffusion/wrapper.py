from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from world_model.models.convgru_predictor import relative_pose_features
from world_model.models.diffusion.sampler import ddim_sample_loop
from world_model.models.diffusion.schedule import DiffusionSchedule
from world_model.models.diffusion.unet import SmallConditionalUNet


@dataclass(frozen=True)
class DiffusionConditioning:
    conditioning: torch.Tensor
    pose_condition: torch.Tensor


class ConditionalVideoDiffusion(nn.Module):
    """Small DDPM/DDIM-style predictor for future RGB frames.

    The model predicts all future frames jointly by flattening the time axis into
    channels. Conditioning is intentionally simple:
    - context RGB frames flattened into channels
    - target pose features flattened into a single vector
    - optional persistent-memory render flattened into channels
    """

    def __init__(
        self,
        *,
        context_frames: int,
        predict_frames: int,
        variant: str = "no_memory",
        model_channels: int = 64,
        diffusion_steps: int = 64,
        image_channels: int = 3,
        memory_channels_per_frame: int = 5,
    ):
        super().__init__()
        if variant not in {"no_memory", "memory"}:
            raise ValueError(f"Unsupported diffusion variant: {variant}")
        self.context_frames = context_frames
        self.predict_frames = predict_frames
        self.variant = variant
        self.image_channels = image_channels
        self.memory_channels_per_frame = memory_channels_per_frame
        self.target_channels = predict_frames * image_channels
        self.pose_dim = predict_frames * 12
        self.context_condition_channels = context_frames * image_channels
        self.memory_condition_channels = predict_frames * memory_channels_per_frame if variant == "memory" else 0
        self.conditioning_channels = self.context_condition_channels + self.memory_condition_channels

        self.schedule = DiffusionSchedule(num_steps=diffusion_steps)
        self.unet = SmallConditionalUNet(
            input_channels=self.target_channels,
            conditioning_channels=self.conditioning_channels,
            pose_dim=self.pose_dim,
            model_channels=model_channels,
        )

    def rgb_sequence_to_channels(self, rgb: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, height, width = rgb.shape
        return (rgb * 2.0 - 1.0).reshape(batch_size, steps * channels, height, width)

    def channels_to_rgb_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = tensor.shape
        if channels != self.target_channels:
            raise ValueError(f"Expected {self.target_channels} target channels, got {channels}")
        rgb = tensor.reshape(batch_size, self.predict_frames, self.image_channels, height, width)
        return ((rgb + 1.0) * 0.5).clamp(0.0, 1.0)

    def _normalize_memory_condition(self, memory_condition: torch.Tensor) -> torch.Tensor:
        memory = memory_condition.clone()
        memory[:, :, :3] = memory[:, :, :3] * 2.0 - 1.0
        batch_size, steps, channels, height, width = memory.shape
        return memory.reshape(batch_size, steps * channels, height, width)

    def build_conditioning(
        self,
        *,
        context_rgb: torch.Tensor,
        context_poses: torch.Tensor,
        target_poses: torch.Tensor,
        memory_condition: torch.Tensor | None = None,
    ) -> DiffusionConditioning:
        batch_size = context_rgb.shape[0]
        context_condition = self.rgb_sequence_to_channels(context_rgb)

        pose_features = []
        anchor_pose = context_poses[:, -1]
        for step_idx in range(target_poses.shape[1]):
            pose_features.append(relative_pose_features(anchor_pose, target_poses[:, step_idx]))
        pose_condition = torch.cat(pose_features, dim=1)
        if pose_condition.shape != (batch_size, self.pose_dim):
            raise ValueError(f"Expected pose condition shape {(batch_size, self.pose_dim)}, got {tuple(pose_condition.shape)}")

        if self.variant == "memory":
            if memory_condition is None:
                raise ValueError("memory_condition is required for the memory-conditioned diffusion variant")
            memory_flat = self._normalize_memory_condition(memory_condition)
            conditioning = torch.cat([context_condition, memory_flat], dim=1)
        else:
            conditioning = context_condition

        return DiffusionConditioning(conditioning=conditioning, pose_condition=pose_condition)

    def training_loss(
        self,
        *,
        context_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        context_poses: torch.Tensor,
        target_poses: torch.Tensor,
        memory_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        conditioning = self.build_conditioning(
            context_rgb=context_rgb,
            context_poses=context_poses,
            target_poses=target_poses,
            memory_condition=memory_condition,
        )
        clean = self.rgb_sequence_to_channels(target_rgb)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, self.schedule.num_steps, (clean.shape[0],), device=clean.device, dtype=torch.long)
        noisy = self.schedule.q_sample(clean, timesteps, noise)
        predicted_noise = self.unet(noisy, timesteps, conditioning.conditioning, conditioning.pose_condition)
        loss = F.mse_loss(predicted_noise, noise)
        metrics = {
            "diffusion_loss": float(loss.detach()),
            "predicted_noise_abs_mean": float(predicted_noise.detach().abs().mean()),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        *,
        context_rgb: torch.Tensor,
        context_poses: torch.Tensor,
        target_poses: torch.Tensor,
        memory_condition: torch.Tensor | None = None,
        sample_steps: int = 25,
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        conditioning = self.build_conditioning(
            context_rgb=context_rgb,
            context_poses=context_poses,
            target_poses=target_poses,
            memory_condition=memory_condition,
        )
        sample, intermediates = ddim_sample_loop(
            model=self.unet,
            schedule=self.schedule,
            shape=(
                context_rgb.shape[0],
                self.target_channels,
                context_rgb.shape[-2],
                context_rgb.shape[-1],
            ),
            conditioning=conditioning.conditioning,
            pose_condition=conditioning.pose_condition,
            sample_steps=sample_steps,
            eta=eta,
            clip_denoised=True,
            return_intermediates=return_intermediates,
        )
        prediction = self.channels_to_rgb_sequence(sample)
        rgb_intermediates = [self.channels_to_rgb_sequence(item.to(sample.device)).cpu() for item in intermediates] if return_intermediates else []
        return prediction, rgb_intermediates
