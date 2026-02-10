"""DDIM sampler for fast inference with configurable step count."""

import torch

from src.models.diffusion_jscc.diffusion import GaussianDiffusion


@torch.no_grad()
def ddim_sample(
    diffusion: GaussianDiffusion,
    x_init: torch.Tensor,
    snr_db: float,
    num_steps: int = 5,
    eta: float = 0.0,
    t_start: int = 200,
) -> torch.Tensor:
    """DDIM sampling starting from lightly-noised x_init.

    The VAE-JSCC reconstruction x_init is already a decent image (~24dB PSNR).
    We add a small amount of noise corresponding to timestep t_start, then
    denoise from t_start→0 in num_steps DDIM steps. This provides light
    refinement rather than full generation from noise.

    Args:
        diffusion: GaussianDiffusion model (with UNet and schedule).
        x_init: VAE-JSCC reconstruction (B, C, H, W) in [0, 1].
        snr_db: Channel SNR in dB (scalar).
        num_steps: Number of DDIM sampling steps.
        eta: Stochasticity (0=deterministic DDIM, 1=DDPM).
        t_start: Starting timestep (lower = less noise added to x_init).

    Returns:
        Refined image (B, C, H, W) in [0, 1].
    """
    device = x_init.device
    B = x_init.shape[0]

    # Clamp t_start to valid range
    t_start = min(t_start, diffusion.T - 1)

    # Create evenly spaced timestep subsequence from t_start down to 0
    step_indices = torch.linspace(t_start, 0, num_steps + 1, device=device).long()

    # Add noise to x_init at level t_start (light corruption, not pure noise)
    t_batch = step_indices[0].expand(B)
    noise = torch.randn_like(x_init)
    x_t = diffusion.q_sample(x_init, t_batch, noise)

    for i in range(num_steps):
        t_cur = step_indices[i]
        t_next = step_indices[i + 1]
        t_b = t_cur.expand(B)

        # Predict noise
        noise_pred = diffusion.unet(x_t, x_init, t_b, snr_db)

        # Current and next alpha_cumprod
        alpha_cur = diffusion.alphas_cumprod[t_cur]
        alpha_next = diffusion.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

        # Predicted x_0
        x0_pred = (x_t - torch.sqrt(1 - alpha_cur) * noise_pred) / torch.sqrt(alpha_cur)
        x0_pred = x0_pred.clamp(0, 1)

        if t_next == 0:
            x_t = x0_pred
        elif eta > 0:
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
