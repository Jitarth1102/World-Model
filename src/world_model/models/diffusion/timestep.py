from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = sinusoidal_timestep_embedding(timesteps, self.embedding_dim)
        return self.mlp(embedding)
