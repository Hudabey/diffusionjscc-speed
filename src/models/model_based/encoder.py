"""S-K Encoder: image -> latent channel symbols with SNR conditioning."""

import torch
import torch.nn as nn

from src.channel.utils import normalize_power


class ResBlock(nn.Module):
    """Residual block: Conv-BN-ReLU-Conv-BN + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation for SNR conditioning.

    Uses (1 + gamma) * features + beta so the transform starts as identity
    (gamma~0, beta~0 at init) and can never zero out features entirely.
    """

    def __init__(self, snr_embed_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(snr_embed_dim, channels)
        self.beta = nn.Linear(snr_embed_dim, channels)
        # Zero-init so FiLM starts as identity
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(
        self, features: torch.Tensor, snr_embed: torch.Tensor
    ) -> torch.Tensor:
        gamma = self.gamma(snr_embed).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(snr_embed).unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * features + beta


class SKEncoder(nn.Module):
    """S-K encoder with 3 stride-2 stages (8x downsampling) and FiLM SNR conditioning.

    Lighter than the VAE encoder (3 stages instead of 4), producing latent
    channel symbols that are power-normalized for AWGN transmission.
    """

    SNR_MIN = -5.0
    SNR_MAX = 25.0

    def __init__(
        self,
        latent_channels: int = 12,
        snr_embed_dim: int = 128,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        ch = base_channels

        # SNR embedding MLP
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, snr_embed_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 1: /2
        self.down1 = nn.Sequential(
            nn.Conv2d(3, ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(ch)
        self.film1 = FiLMConditioner(snr_embed_dim, ch)

        # Stage 2: /4
        self.down2 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResBlock(ch * 2)
        self.film2 = FiLMConditioner(snr_embed_dim, ch * 2)

        # Stage 3: /8
        self.down3 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),
        )
        self.res3 = ResBlock(ch * 4)
        self.film3 = FiLMConditioner(snr_embed_dim, ch * 4)

        # 1x1 conv to latent channels
        self.to_latent = nn.Conv2d(ch * 4, latent_channels, 1)

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Encode image to power-normalized latent channel symbols.

        Args:
            x: Input images (B, 3, H, W) in [0, 1]. H, W must be divisible by 8.
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Power-normalized latent (B, latent_channels, H/8, W/8).
        """
        B = x.shape[0]
        snr_norm = (snr_db - self.SNR_MIN) / (self.SNR_MAX - self.SNR_MIN)
        snr_t = torch.tensor([[snr_norm]], device=x.device, dtype=x.dtype).expand(B, 1)
        snr_embed = self.snr_mlp(snr_t)

        h = self.down1(x)
        h = self.film1(self.res1(h), snr_embed)

        h = self.down2(h)
        h = self.film2(self.res2(h), snr_embed)

        h = self.down3(h)
        h = self.film3(self.res3(h), snr_embed)

        z = self.to_latent(h)

        # Power normalize (flatten, normalize, reshape)
        spatial_shape = z.shape
        z_flat = z.flatten(1)
        z_flat = normalize_power(z_flat, target_power=1.0, mode="per_sample")
        z = z_flat.view(spatial_shape)

        return z
