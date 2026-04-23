from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from world_model.models.diffusion.timestep import TimestepEmbedding


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(embedding_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(embedding))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallConditionalUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conditioning_channels: int,
        pose_dim: int,
        model_channels: int = 64,
        embedding_dim: int | None = None,
    ):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = model_channels * 4
        self.input_channels = input_channels
        self.conditioning_channels = conditioning_channels
        self.pose_dim = pose_dim
        self.model_channels = model_channels
        self.embedding_dim = embedding_dim

        self.time_embedding = TimestepEmbedding(model_channels, embedding_dim)
        self.pose_embedding = nn.Sequential(
            nn.Linear(pose_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.input_proj = nn.Conv2d(input_channels + conditioning_channels, model_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(model_channels, model_channels, embedding_dim)
        self.downsample1 = nn.Conv2d(model_channels, model_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResidualBlock(model_channels * 2, model_channels * 2, embedding_dim)
        self.downsample2 = nn.Conv2d(model_channels * 2, model_channels * 4, kernel_size=4, stride=2, padding=1)
        self.mid = ResidualBlock(model_channels * 4, model_channels * 4, embedding_dim)
        self.upsample1 = nn.ConvTranspose2d(model_channels * 4, model_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(model_channels * 4, model_channels * 2, embedding_dim)
        self.upsample2 = nn.ConvTranspose2d(model_channels * 2, model_channels, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(model_channels * 2, model_channels, embedding_dim)
        self.out_norm = _group_norm(model_channels)
        self.out = nn.Conv2d(model_channels, input_channels, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
        pose_condition: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.time_embedding(timesteps) + self.pose_embedding(pose_condition)

        h0 = self.input_proj(torch.cat([noisy_input, conditioning], dim=1))
        h1 = self.down1(h0, embedding)
        d1 = self.downsample1(h1)
        h2 = self.down2(d1, embedding)
        d2 = self.downsample2(h2)
        mid = self.mid(d2, embedding)
        u1 = self.upsample1(mid)
        u1 = self.up1(torch.cat([u1, h2], dim=1), embedding)
        u2 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u2, h1], dim=1), embedding)
        return self.out(F.silu(self.out_norm(u2)))
