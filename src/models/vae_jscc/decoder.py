"""DeepJSCC decoder: latent channel symbols -> reconstructed image."""

import torch
import torch.nn as nn

from src.models.vae_jscc.encoder import ResBlock, SNRAttention


class JSCCDecoder(nn.Module):
    """Fully-convolutional JSCC decoder with SNR-adaptive attention.

    Mirror of encoder: 4 stride-2 transposed convolution stages (16x upsample)
    with 5x5 kernels and 2 ResBlocks per stage. SNRAttention applied after
    every stage for strong SNR conditioning at all scales.
    """

    def __init__(
        self,
        latent_channels: int = 192,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        ch = base_channels

        # Expand from latent channels
        self.from_latent = nn.Conv2d(latent_channels, ch * 4, 1)

        # Stage 1: x2
        self.stage1 = nn.Sequential(
            ResBlock(ch * 4),
            ResBlock(ch * 4),
            nn.ConvTranspose2d(ch * 4, ch * 4, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(ch * 4),
        )
        self.attn1 = SNRAttention(ch * 4)

        # Stage 2: x4
        self.stage2 = nn.Sequential(
            ResBlock(ch * 4),
            ResBlock(ch * 4),
            nn.ConvTranspose2d(ch * 4, ch * 2, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(ch * 2),
        )
        self.attn2 = SNRAttention(ch * 2)

        # Stage 3: x8
        self.stage3 = nn.Sequential(
            ResBlock(ch * 2),
            ResBlock(ch * 2),
            nn.ConvTranspose2d(ch * 2, ch, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(ch),
        )
        self.attn3 = SNRAttention(ch)

        # Stage 4: x16 -> output
        self.stage4 = nn.Sequential(
            ResBlock(ch),
            ResBlock(ch),
            nn.ConvTranspose2d(ch, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_noisy: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Decode noisy latent symbols to reconstructed image.

        Args:
            z_noisy: Noisy latent (B, latent_channels, H_lat, W_lat).
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Reconstructed image (B, 3, H_lat*16, W_lat*16) in [0, 1].
        """
        h = self.from_latent(z_noisy)
        h = self.attn1(self.stage1(h), snr_db)
        h = self.attn2(self.stage2(h), snr_db)
        h = self.attn3(self.stage3(h), snr_db)
        h = self.stage4(h)
        return h


# Backward-compat alias
Decoder = JSCCDecoder
