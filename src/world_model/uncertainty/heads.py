from __future__ import annotations

import torch
from torch import nn

from world_model.models.decoder import SimpleConvDecoder


class HeteroscedasticUncertaintyHead(nn.Module):
    """Predicts a scalar log-variance map for RGB uncertainty."""

    def __init__(self, hidden_channels: int, out_channels: int = 1, initial_log_variance: float = -2.0):
        super().__init__()
        self.decoder = SimpleConvDecoder(hidden_channels, out_channels)
        final_layer = self.decoder.net[-1]
        if isinstance(final_layer, nn.ConvTranspose2d) and final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, initial_log_variance)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden)
