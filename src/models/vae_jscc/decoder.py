"""VAE-JSCC decoder: spatial latent map → reconstructed image."""

import torch
import torch.nn as nn

from src.models.vae_jscc.encoder import FiLMConditioner, ResBlock


class Decoder(nn.Module):
    """Fully-convolutional VAE-JSCC decoder with FiLM SNR conditioning.

    Mirror of the encoder: 1×1 conv to expand channels, then 4 stride-2
    transposed convolution stages with residual blocks. Outputs image in [0, 1].
    """

    def __init__(
        self,
        latent_channels: int,
        snr_embed_dim: int = 256,
        base_channels: int = 64,
    ) -> None:
        """Initialize decoder.

        Args:
            latent_channels: Number of channels in the latent spatial map.
            snr_embed_dim: Dimension of the SNR embedding.
            base_channels: Base number of channels (matches encoder).
        """
        super().__init__()
        self.latent_channels = latent_channels
        ch = base_channels

        # SNR embedding MLP (separate from encoder's)
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, snr_embed_dim),
            nn.ReLU(inplace=True),
        )

        # 1×1 conv from latent channels to feature channels
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_channels, ch * 4, 1),
            nn.ReLU(inplace=True),
        )

        # Stage 1: ×2
        self.res1 = ResBlock(ch * 4)
        self.film1 = FiLMConditioner(snr_embed_dim, ch * 4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),
        )

        # Stage 2: ×4
        self.res2 = ResBlock(ch * 4)
        self.film2 = FiLMConditioner(snr_embed_dim, ch * 4)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )

        # Stage 3: ×8
        self.res3 = ResBlock(ch * 2)
        self.film3 = FiLMConditioner(snr_embed_dim, ch * 2)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # Stage 4: ×16 → output
        self.res4 = ResBlock(ch)
        self.film4 = FiLMConditioner(snr_embed_dim, ch)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ch, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Decode spatial latent map to reconstructed image.

        Args:
            z: Latent spatial map (B, latent_channels, H_lat, W_lat).
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Reconstructed image (B, 3, H_lat*16, W_lat*16) in [0, 1].
        """
        B = z.shape[0]
        snr_t = torch.tensor([[snr_db]], device=z.device, dtype=z.dtype).expand(B, 1)
        snr_embed = self.snr_mlp(snr_t)

        h = self.from_latent(z)

        h = self.film1(self.res1(h), snr_embed)
        h = self.up1(h)

        h = self.film2(self.res2(h), snr_embed)
        h = self.up2(h)

        h = self.film3(self.res3(h), snr_embed)
        h = self.up3(h)

        h = self.film4(self.res4(h), snr_embed)
        h = self.up4(h)

        return h
