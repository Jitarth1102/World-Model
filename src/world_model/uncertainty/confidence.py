from __future__ import annotations

import torch


def calibrate_confidence(confidence: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if gamma == 1.0:
        return confidence
    calibrated = torch.clamp(confidence, min=1e-6, max=1.0)
    return torch.pow(calibrated, gamma)


def uncertainty_to_confidence(
    log_variance: torch.Tensor,
    clamp_min: float = -6.0,
    clamp_max: float = 6.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    clamped = torch.clamp(log_variance, min=clamp_min, max=clamp_max)
    precision = torch.exp(-clamped)
    confidence = precision / (1.0 + precision)
    return calibrate_confidence(confidence, gamma=gamma)


def variance_to_confidence(variance: torch.Tensor, eps: float = 1e-6, gamma: float = 1.0) -> torch.Tensor:
    precision = torch.rsqrt(torch.clamp(variance, min=eps))
    confidence = precision / (1.0 + precision)
    return calibrate_confidence(confidence, gamma=gamma)


def confidence_to_write_mask(confidence: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
    if threshold is None:
        return torch.ones_like(confidence, dtype=confidence.dtype)
    return (confidence >= threshold).to(dtype=confidence.dtype)
