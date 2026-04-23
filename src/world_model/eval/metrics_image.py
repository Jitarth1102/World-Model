from __future__ import annotations

import torch


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return (prediction - target).abs().mean()
    mask = mask.to(dtype=prediction.dtype)
    denom = mask.sum().clamp_min(1.0)
    return ((prediction - target).abs() * mask).sum() / denom


def psnr(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = ((prediction - target) ** 2).mean().clamp_min(eps)
    return -10.0 * torch.log10(mse)
