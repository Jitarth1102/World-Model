from __future__ import annotations

import torch

from world_model.eval.metrics_image import masked_l1


def mask_coverage(mask: torch.Tensor) -> torch.Tensor:
    return mask.to(dtype=torch.float32).mean()


def memory_covered_l1(prediction: torch.Tensor, target: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
    return masked_l1(prediction, target, memory_mask)


def oracle_alignment_l1(prediction: torch.Tensor, memory_render_rgb: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
    return masked_l1(prediction, memory_render_rgb, memory_mask)


def baseline_advantage(model_metric: torch.Tensor | float, baseline_metric: torch.Tensor | float) -> torch.Tensor | float:
    return baseline_metric - model_metric
