from __future__ import annotations

import torch
from torch import nn

from world_model.models.decoder import SimpleConvDecoder
from world_model.models.encoder import SimpleConvEncoder


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                (x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            )
        combined = torch.cat([x, hidden], dim=1)
        reset_gate, update_gate = torch.chunk(torch.sigmoid(self.gates(combined)), 2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset_gate * hidden], dim=1)))
        return (1.0 - update_gate) * hidden + update_gate * candidate


def relative_pose_features(anchor_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    anchor_inv = torch.linalg.inv(anchor_pose)
    relative = anchor_inv @ target_pose
    return relative[:, :3, :4].reshape(relative.shape[0], -1)


class NoMemoryPredictor(nn.Module):
    """Small RGB future-view predictor used as a carrier for later memory experiments."""

    def __init__(self, image_channels: int = 3, hidden_channels: int = 64, pose_dim: int = 12):
        super().__init__()
        self.encoder = SimpleConvEncoder(image_channels, hidden_channels)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.temporal_cell = ConvGRUCell(hidden_channels, hidden_channels)
        self.decoder = SimpleConvDecoder(hidden_channels, image_channels)

    def encode_context(self, context_rgb: torch.Tensor) -> torch.Tensor:
        hidden = None
        for frame_idx in range(context_rgb.shape[1]):
            encoded = self.encoder(context_rgb[:, frame_idx])
            hidden = self.temporal_cell(encoded, hidden)
        return hidden

    def forward(self, context_rgb: torch.Tensor, context_poses: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_context(context_rgb)
        anchor_pose = context_poses[:, -1]
        predictions = []
        for step_idx in range(target_poses.shape[1]):
            pose_features = relative_pose_features(anchor_pose, target_poses[:, step_idx])
            pose_embedding = self.pose_proj(pose_features)[:, :, None, None]
            step_input = pose_embedding.expand(-1, -1, hidden.shape[2], hidden.shape[3])
            hidden = self.temporal_cell(step_input, hidden)
            predictions.append(torch.sigmoid(self.decoder(hidden)))
        return torch.stack(predictions, dim=1)
