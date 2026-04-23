from __future__ import annotations

import torch
from torch import nn

from world_model.models.convgru_predictor import ConvGRUCell, relative_pose_features
from world_model.models.decoder import SimpleConvDecoder
from world_model.models.encoder import SimpleConvEncoder
from world_model.uncertainty.heads import HeteroscedasticUncertaintyHead


class MemoryConditionedWorldModel(nn.Module):
    """Lightweight RGB-D-capable backbone for later persistent-memory experiments.

    For now, the model consumes rendered memory views directly as an input condition.
    This keeps the learned component small while preserving the write/read/condition loop.
    """

    def __init__(
        self,
        image_channels: int = 3,
        memory_channels: int = 5,
        hidden_channels: int = 64,
        pose_dim: int = 12,
        residual_scale: float = 0.25,
        enable_uncertainty: bool = False,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.enable_uncertainty = enable_uncertainty
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.image_encoder = SimpleConvEncoder(image_channels, hidden_channels)
        self.memory_encoder = SimpleConvEncoder(memory_channels, hidden_channels)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.fuse = nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1)
        self.temporal_cell = ConvGRUCell(hidden_channels, hidden_channels)
        self.rgb_decoder = SimpleConvDecoder(hidden_channels, image_channels)
        self.depth_decoder = SimpleConvDecoder(hidden_channels, 1)
        self.uncertainty_head = HeteroscedasticUncertaintyHead(hidden_channels, 1) if enable_uncertainty else None

    def encode_context(self, context_rgb: torch.Tensor) -> torch.Tensor:
        hidden = None
        for frame_idx in range(context_rgb.shape[1]):
            encoded = self.image_encoder(context_rgb[:, frame_idx])
            hidden = self.temporal_cell(encoded, hidden)
        return hidden

    def initialize_rollout(
        self,
        context_rgb: torch.Tensor,
        context_poses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode_context(context_rgb)
        anchor_pose = context_poses[:, -1]
        prev_frame = context_rgb[:, -1]
        return hidden, anchor_pose, prev_frame

    def predict_step(
        self,
        hidden: torch.Tensor,
        prev_frame: torch.Tensor,
        anchor_pose: torch.Tensor,
        target_pose: torch.Tensor,
        memory_condition_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pose_features = relative_pose_features(anchor_pose, target_pose)
        pose_embedding = self.pose_proj(pose_features)[:, :, None, None]
        pose_map = pose_embedding.expand(-1, -1, hidden.shape[2], hidden.shape[3])
        prev_latent = self.image_encoder(prev_frame)
        memory_latent = self.memory_encoder(memory_condition_step)
        fused = self.fuse(torch.cat([prev_latent, memory_latent, pose_map], dim=1))
        hidden = self.temporal_cell(fused, hidden)
        residual = torch.tanh(self.rgb_decoder(hidden)) * self.residual_scale
        next_frame = torch.clamp(prev_frame + residual, 0.0, 1.0)
        depth = torch.relu(self.depth_decoder(hidden))
        log_variance = None
        if self.uncertainty_head is not None:
            log_variance = torch.clamp(self.uncertainty_head(hidden), min=self.logvar_min, max=self.logvar_max)
        return hidden, next_frame, depth, log_variance

    def forward(
        self,
        context_rgb: torch.Tensor,
        context_poses: torch.Tensor,
        target_poses: torch.Tensor,
        memory_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, anchor_pose, prev_frame = self.initialize_rollout(context_rgb, context_poses)
        rgb_predictions = []
        depth_predictions = []
        log_variances = []
        for step_idx in range(target_poses.shape[1]):
            hidden, prev_frame, depth, log_variance = self.predict_step(
                hidden=hidden,
                prev_frame=prev_frame,
                anchor_pose=anchor_pose,
                target_pose=target_poses[:, step_idx],
                memory_condition_step=memory_condition[:, step_idx],
            )
            rgb_predictions.append(prev_frame)
            depth_predictions.append(depth)
            if log_variance is not None:
                log_variances.append(log_variance)
        rgb_stack = torch.stack(rgb_predictions, dim=1)
        depth_stack = torch.stack(depth_predictions, dim=1)
        if log_variances:
            return rgb_stack, depth_stack, torch.stack(log_variances, dim=1)
        return rgb_stack, depth_stack
