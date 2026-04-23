from __future__ import annotations

import torch

from world_model.models.diffusion.schedule import DiffusionSchedule


def _sample_timesteps(num_diffusion_steps: int, sample_steps: int, device: torch.device) -> torch.Tensor:
    if sample_steps >= num_diffusion_steps:
        return torch.arange(num_diffusion_steps - 1, -1, -1, device=device, dtype=torch.long)
    return torch.linspace(num_diffusion_steps - 1, 0, sample_steps, device=device).long()


@torch.no_grad()
def ddim_sample_loop(
    *,
    model,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    conditioning: torch.Tensor,
    pose_condition: torch.Tensor,
    sample_steps: int,
    eta: float = 0.0,
    clip_denoised: bool = True,
    return_intermediates: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    device = conditioning.device
    x = torch.randn(shape, device=device)
    timesteps = _sample_timesteps(schedule.num_steps, sample_steps, device=device)
    intermediates: list[torch.Tensor] = []

    for idx, timestep in enumerate(timesteps):
        t_batch = torch.full((shape[0],), int(timestep.item()), device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch, conditioning, pose_condition)
        predicted_start = schedule.predict_start_from_noise(x, t_batch, predicted_noise)
        if clip_denoised:
            predicted_start = predicted_start.clamp(-1.0, 1.0)
        if return_intermediates:
            intermediates.append(predicted_start.detach().cpu())

        if idx == len(timesteps) - 1:
            x = predicted_start
            break

        next_timestep = timesteps[idx + 1]
        alpha_bar_t = schedule.alphas_cumprod[int(timestep.item())]
        alpha_bar_next = schedule.alphas_cumprod[int(next_timestep.item())]
        sigma = eta * torch.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_next))
        noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)
        direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma**2, min=0.0)) * predicted_noise
        x = torch.sqrt(alpha_bar_next) * predicted_start + direction + sigma * noise

    return x, intermediates
