from __future__ import annotations

import torch
from torch import nn


class SimpleConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        mid = hidden_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(mid, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
