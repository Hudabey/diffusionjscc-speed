"""DeepJSCC encoder: image -> latent channel symbols with SNR-adaptive attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: Conv-BN-PReLU-Conv-BN + skip + PReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SNRAttention(nn.Module):
    """Channel-wise attention modulated by SNR.

    At low SNR: suppress detail channels, keep only robust features.
    At high SNR: activate all channels for maximum detail.
    """

    SNR_MIN = -5.0
    SNR_MAX = 25.0

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.snr_net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(channels + 128, channels),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        B = z.shape[0]
        snr_norm = (snr_db - self.SNR_MIN) / (self.SNR_MAX - self.SNR_MIN)
        snr_t = torch.tensor([[snr_norm]], device=z.device, dtype=z.dtype).expand(B, 1)
        snr_emb = self.snr_net(snr_t)
        pool = F.adaptive_avg_pool2d(z, 1).view(B, -1)
        gate = self.gate(torch.cat([pool, snr_emb], dim=1))
        return z * gate.unsqueeze(-1).unsqueeze(-1)


class JSCCEncoder(nn.Module):
    """Fully-convolutional JSCC encoder with SNR-adaptive channel attention.

    4 stride-2 downsampling stages (16x spatial reduction) with 5x5 kernels
    and 2 ResBlocks per stage. No flatten, no linear, no VAE.
    Inspired by Bourtsoulatze et al. 2019 (DeepJSCC) and ADJSCC.
    """

    def __init__(
        self,
        latent_channels: int = 192,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        ch = base_channels

        # Stage 1: /2
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, ch, 5, stride=2, padding=2),
            nn.PReLU(ch),
            ResBlock(ch),
            ResBlock(ch),
        )
        # Stage 2: /4
        self.stage2 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 5, stride=2, padding=2),
            nn.PReLU(ch * 2),
            ResBlock(ch * 2),
            ResBlock(ch * 2),
        )
        # Stage 3: /8
        self.stage3 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 5, stride=2, padding=2),
            nn.PReLU(ch * 4),
            ResBlock(ch * 4),
            ResBlock(ch * 4),
        )
        # Stage 4: /16
        self.stage4 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 4, 5, stride=2, padding=2),
            nn.PReLU(ch * 4),
            ResBlock(ch * 4),
            ResBlock(ch * 4),
        )

        # Project to latent channels
        self.to_latent = nn.Conv2d(ch * 4, latent_channels, 1)

        # SNR-adaptive channel attention
        self.snr_attention = SNRAttention(latent_channels)

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Encode image to latent channel symbols.

        Args:
            x: Input images (B, 3, H, W) in [0, 1]. H, W must be divisible by 16.
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Latent feature map (B, latent_channels, H/16, W/16).
        """
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        z = self.to_latent(h)
        z = self.snr_attention(z, snr_db)
        return z


# Backward-compat alias used by old imports
Encoder = JSCCEncoder
