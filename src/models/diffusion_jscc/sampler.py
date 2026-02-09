"""DDIM sampler for fast inference with configurable step count."""

import torch

from src.models.diffusion_jscc.diffusion import GaussianDiffusion


@torch.no_grad()
def ddim_sample(
    diffusion: GaussianDiffusion,
    x_init: torch.Tensor,
    snr_db: float,
    num_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """DDIM sampling starting from noisy x_init.

    Starts from x_init with added noise (not pure noise), then iteratively
    denoises. Even 1 step gives a reasonable result since x_init is already
    a decent reconstruction.

    Args:
        diffusion: GaussianDiffusion model (with UNet and schedule).
        x_init: VAE-JSCC reconstruction (B, C, H, W) in [0, 1].
        snr_db: Channel SNR in dB (scalar).
        num_steps: Number of DDIM sampling steps.
        eta: Stochasticity (0=deterministic DDIM, 1=DDPM).

    Returns:
        Refined image (B, C, H, W) in [0, 1].
    """
    device = x_init.device
    B = x_init.shape[0]
    T = diffusion.T

    # Create evenly spaced timestep subsequence
    step_indices = torch.linspace(T - 1, 0, num_steps + 1, device=device).long()

    # Start from x_init with noise scaled by a timestep partway through the schedule
    # Use the first timestep in our subsequence to determine noise level
    t_start = step_indices[0]
    noise = torch.randn_like(x_init)
    x_t = diffusion.q_sample(x_init, t_start.expand(B), noise)

    for i in range(num_steps):
        t_cur = step_indices[i]
        t_next = step_indices[i + 1]

        t_batch = t_cur.expand(B)

        # Predict noise
        noise_pred = diffusion.unet(x_t, x_init, t_batch, snr_db)

        # Current and next alpha_cumprod
        alpha_cur = diffusion.alphas_cumprod[t_cur]
        alpha_next = diffusion.alphas_cumprod[t_next]

        # Predicted x_0
        x0_pred = (x_t - torch.sqrt(1 - alpha_cur) * noise_pred) / torch.sqrt(alpha_cur)
        x0_pred = x0_pred.clamp(0, 1)

        # DDIM step
        if eta > 0 and i < num_steps - 1:
            sigma = (
                eta
                * torch.sqrt((1 - alpha_next) / (1 - alpha_cur))
                * torch.sqrt(1 - alpha_cur / alpha_next)
            )
            dir_xt = torch.sqrt(1 - alpha_next - sigma**2) * noise_pred
            x_t = torch.sqrt(alpha_next) * x0_pred + dir_xt + sigma * torch.randn_like(x_t)
        else:
            dir_xt = torch.sqrt(1 - alpha_next) * noise_pred
            x_t = torch.sqrt(alpha_next) * x0_pred + dir_xt

    return x_t.clamp(0, 1)
