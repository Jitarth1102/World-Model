from __future__ import annotations

import numpy as np
import torch


def _flatten_metric_tensors(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty_np = uncertainty.detach().cpu().float().reshape(-1).numpy()
    error_np = error.detach().cpu().float().reshape(-1).numpy()
    if mask is not None:
        valid = mask.detach().cpu().bool().reshape(-1).numpy()
        uncertainty_np = uncertainty_np[valid]
        error_np = error_np[valid]
    finite = np.isfinite(uncertainty_np) & np.isfinite(error_np)
    return uncertainty_np[finite], error_np[finite]


def uncertainty_error_correlation(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    uncertainty_np, error_np = _flatten_metric_tensors(uncertainty, error, mask)
    if len(uncertainty_np) < 2:
        return 0.0
    if np.std(uncertainty_np) < 1e-8 or np.std(error_np) < 1e-8:
        return 0.0
    return float(np.corrcoef(uncertainty_np, error_np)[0, 1])


def high_error_auroc(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    mask: torch.Tensor | None = None,
    positive_quantile: float = 0.9,
) -> float:
    scores, error_np = _flatten_metric_tensors(uncertainty, error, mask)
    if len(scores) < 2:
        return 0.5
    threshold = float(np.quantile(error_np, positive_quantile))
    labels = error_np >= threshold
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return 0.5

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sum_ranks_pos = float(ranks[labels].sum())
    auc = (sum_ranks_pos - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(np.clip(auc, 0.0, 1.0))
