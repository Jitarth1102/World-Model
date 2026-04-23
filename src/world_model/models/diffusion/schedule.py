from __future__ import annotations

import torch
from torch import nn


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.num_steps = num_steps
        betas = linear_beta_schedule(num_steps, beta_start=beta_start, beta_end=beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

    def extract(self, values: torch.Tensor, timesteps: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        gathered = values.gather(0, timesteps)
        return gathered.reshape((target_shape[0],) + (1,) * (len(target_shape) - 1))

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.extract(self.sqrt_alphas_cumprod, timesteps, clean.shape) * clean + self.extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, clean.shape
        ) * noise

    def predict_start_from_noise(self, noisy: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy.shape) * noisy - self.extract(
            self.sqrt_recipm1_alphas_cumprod, timesteps, noisy.shape
        ) * noise
