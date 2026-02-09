"""VAE-JSCC: full model combining encoder, channel, and decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.channel.awgn import awgn_channel
from src.channel.utils import normalize_power
from src.models.vae_jscc.decoder import Decoder
from src.models.vae_jscc.encoder import Encoder


def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad image tensor so H and W are multiples of `multiple`.

    Args:
        x: Image tensor (B, C, H, W).
        multiple: Target multiple for spatial dimensions.

    Returns:
        Padded tensor. Original content is in the top-left corner.
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


class VAEJSCC(nn.Module):
    """VAE-based Joint Source-Channel Coding.

    Fully convolutional architecture: encoder produces spatial latent maps
    (mu, log_sigma), which are flattened for power normalization and AWGN
    channel transmission, then reshaped back for the decoder.

    The latent is a spatial feature map, not a flat vector, which avoids
    huge linear layers and naturally handles variable input sizes.
    """

    def __init__(
        self,
        latent_channels: int = 192,
        snr_embed_dim: int = 256,
        base_channels: int = 64,
    ) -> None:
        """Initialize VAE-JSCC.

        Args:
            latent_channels: Number of latent channels. Controls bandwidth ratio:
                rho = latent_channels / (3 * 16^2) for the standard 4-stage encoder.
                E.g., 192 channels with 256x256 input → latent_dim = 192*16*16 = 49152,
                rho = 49152 / (3*256*256) = 1/4.
            snr_embed_dim: Dimension of SNR embedding for FiLM conditioning.
            base_channels: Base channel count for conv layers.
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = Encoder(latent_channels, snr_embed_dim, base_channels)
        self.decoder = Decoder(latent_channels, snr_embed_dim, base_channels)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon.

        Args:
            mu: Mean of the latent distribution (any shape).
            log_sigma: Log standard deviation (same shape as mu).

        Returns:
            Sampled latent tensor (same shape as mu).
        """
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        x: torch.Tensor,
        snr_db: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → channel → decode.

        Args:
            x: Input images (B, 3, H, W) in [0, 1]. H, W must be divisible by 16.
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Tuple of (x_hat, mu_flat, log_sigma_flat).
            x_hat: Reconstructed images (B, 3, H, W) in [0, 1].
            mu_flat, log_sigma_flat: Flattened latent parameters (B, k).
        """
        # Encode → spatial latent maps
        mu_map, logsig_map = self.encoder(x, snr_db)
        spatial_shape = mu_map.shape  # (B, C_lat, H_lat, W_lat)

        # Flatten for reparameterization, power norm, and channel
        mu = mu_map.flatten(1)
        log_sigma = logsig_map.flatten(1)

        if self.training:
            z = self.reparameterize(mu, log_sigma)
        else:
            z = mu  # Use mean at eval time for deterministic output

        z = normalize_power(z, target_power=1.0, mode="per_sample")
        z_noisy, _ = awgn_channel(z, snr_db)

        # Reshape back to spatial map for decoder
        z_spatial = z_noisy.view(spatial_shape)
        x_hat = self.decoder(z_spatial, snr_db)

        return x_hat, mu, log_sigma

    @staticmethod
    def compute_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        beta: float = 0.001,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss: reconstruction + beta * KL divergence.

        Args:
            x: Original images (B, 3, H, W).
            x_hat: Reconstructed images (B, 3, H, W).
            mu: Latent mean (B, k).
            log_sigma: Latent log std (B, k).
            beta: Weight for KL divergence. Keep small (0.0001–0.01)
                since this is a reconstruction model, not generative.

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss).
        """
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(
            1 + log_sigma - mu.pow(2) - log_sigma.exp()
        )
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
