"""Diffusion process: cosine schedule, forward diffusion, and loss computation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.diffusion_jscc.unet import ConditionalUNet


def cosine_beta_schedule(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021).

    Args:
        T: Number of diffusion timesteps.
        s: Small offset to prevent beta from being too small at t=0.

    Returns:
        (T,) tensor of beta values.
    """
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion process with epsilon-prediction loss.

    Wraps a conditional UNet and provides:
    - Forward diffusion (adding noise to clean images)
    - Training loss (predict noise at random timestep)
    - Pre-computed schedule tensors
    """

    def __init__(self, unet: ConditionalUNet, T: int = 1000) -> None:
        """Initialize diffusion process.

        Args:
            unet: Conditional UNet for noise prediction.
            T: Number of diffusion timesteps.
        """
        super().__init__()
        self.unet = unet
        self.T = T

        # Compute schedule
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register as buffers (moved to device with model, not parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean image at timestep t.

        Args:
            x_0: Clean images (B, C, H, W) in [0, 1].
            t: Timesteps (B,) integers in [0, T).
            noise: Optional pre-sampled noise. If None, sampled from N(0, 1).

        Returns:
            Noisy images x_t (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def training_loss(
        self,
        x_clean: torch.Tensor,
        x_init: torch.Tensor,
        snr_db: float,
    ) -> torch.Tensor:
        """Compute diffusion training loss (epsilon-prediction MSE).

        Args:
            x_clean: Ground truth images (B, C, H, W) in [0, 1].
            x_init: VAE-JSCC reconstruction condition (B, C, H, W) in [0, 1].
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Scalar MSE loss between predicted and actual noise.
        """
        B = x_clean.shape[0]
        device = x_clean.device

        # Random timesteps
        t = torch.randint(0, self.T, (B,), device=device)

        # Random noise
        noise = torch.randn_like(x_clean)

        # Forward diffusion on CLEAN image
        x_noisy = self.q_sample(x_clean, t, noise)

        # Predict noise conditioned on x_init and SNR
        noise_pred = self.unet(x_noisy, x_init, t, snr_db)

        return F.mse_loss(noise_pred, noise)
