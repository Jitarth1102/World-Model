from __future__ import annotations

import torch
from torch import nn

from world_model.models.convgru_predictor import ConvGRUCell, relative_pose_features
from world_model.models.decoder import SimpleConvDecoder
from world_model.models.encoder import SimpleConvEncoder


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
    ):
        super().__init__()
        self.residual_scale = residual_scale
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

    def encode_context(self, context_rgb: torch.Tensor) -> torch.Tensor:
        hidden = None
        for frame_idx in range(context_rgb.shape[1]):
            encoded = self.image_encoder(context_rgb[:, frame_idx])
            hidden = self.temporal_cell(encoded, hidden)
        return hidden

    def forward(
        self,
        context_rgb: torch.Tensor,
        context_poses: torch.Tensor,
        target_poses: torch.Tensor,
        memory_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_context(context_rgb)
        anchor_pose = context_poses[:, -1]
        prev_frame = context_rgb[:, -1]
        rgb_predictions = []
        depth_predictions = []
        for step_idx in range(target_poses.shape[1]):
            pose_features = relative_pose_features(anchor_pose, target_poses[:, step_idx])
            pose_embedding = self.pose_proj(pose_features)[:, :, None, None]
            pose_map = pose_embedding.expand(-1, -1, hidden.shape[2], hidden.shape[3])
            prev_latent = self.image_encoder(prev_frame)
            memory_latent = self.memory_encoder(memory_condition[:, step_idx])
            fused = self.fuse(torch.cat([prev_latent, memory_latent, pose_map], dim=1))
            hidden = self.temporal_cell(fused, hidden)
            residual = torch.tanh(self.rgb_decoder(hidden)) * self.residual_scale
            prev_frame = torch.clamp(prev_frame + residual, 0.0, 1.0)
            rgb_predictions.append(prev_frame)
            depth_predictions.append(torch.relu(self.depth_decoder(hidden)))
        return torch.stack(rgb_predictions, dim=1), torch.stack(depth_predictions, dim=1)
