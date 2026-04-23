from __future__ import annotations

import torch


def uncertainty_to_confidence(
    log_variance: torch.Tensor,
    clamp_min: float = -6.0,
    clamp_max: float = 6.0,
) -> torch.Tensor:
    clamped = torch.clamp(log_variance, min=clamp_min, max=clamp_max)
    precision = torch.exp(-clamped)
    return precision / (1.0 + precision)


def variance_to_confidence(variance: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    precision = torch.rsqrt(torch.clamp(variance, min=eps))
    return precision / (1.0 + precision)


def confidence_to_write_mask(confidence: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
    if threshold is None:
        return torch.ones_like(confidence, dtype=confidence.dtype)
    return (confidence >= threshold).to(dtype=confidence.dtype)
