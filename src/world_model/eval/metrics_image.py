from __future__ import annotations

import torch


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return (prediction - target).abs().mean()
    mask = mask.to(dtype=prediction.dtype)
    mask = torch.broadcast_to(mask, prediction.shape)
    denom = mask.sum().clamp_min(1.0)
    return ((prediction - target).abs() * mask).sum() / denom


def psnr(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = ((prediction - target) ** 2).mean().clamp_min(eps)
    return -10.0 * torch.log10(mse)


def motion_mask_from_last_context(
    context_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Return a binary mask for pixels that change relative to the last context frame."""
    reference = context_rgb[:, -1:].expand(-1, target_rgb.shape[1], -1, -1, -1)
    delta = (target_rgb - reference).abs().mean(dim=2, keepdim=True)
    return (delta > threshold).to(dtype=target_rgb.dtype)
