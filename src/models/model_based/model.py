"""Model-Based JSCC: full pipeline combining S-K encoder, channel, and unrolled decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.channel.awgn import awgn_channel
from src.models.model_based.decoder import UnrolledDecoder
from src.models.model_based.encoder import SKEncoder


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


class ModelBasedJSCC(nn.Module):
    """Model-based JSCC with S-K encoder and unrolled iterative decoder.

    Combines communication-theoretic structure (power normalization, iterative
    decoding with learned step sizes) with learned CNN components. Targets
    3-10x fewer parameters than VAE-JSCC while maintaining competitive PSNR.
    """

    def __init__(
        self,
        latent_channels: int = 12,
        num_iterations: int = 6,
        denoiser_channels: int = 64,
        snr_embed_dim: int = 128,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.num_iterations = num_iterations

        self.encoder = SKEncoder(
            latent_channels=latent_channels,
            snr_embed_dim=snr_embed_dim,
            base_channels=base_channels,
        )
        self.decoder = UnrolledDecoder(
            latent_channels=latent_channels,
            num_iterations=num_iterations,
            denoiser_channels=denoiser_channels,
            snr_embed_dim=snr_embed_dim,
            base_channels=base_channels,
        )

    def forward(
        self, x: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """Full forward pass: encode -> channel -> iterative decode.

        Args:
            x: Input images (B, 3, H, W) in [0, 1]. H, W must be divisible by 8.
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Reconstructed images (B, 3, H, W) in [0, 1].
        """
        # Encode (power normalization is built into encoder)
        z = self.encoder(x, snr_db)

        # AWGN channel
        z_noisy, _ = awgn_channel(z.flatten(1), snr_db)
        z_noisy = z_noisy.view(z.shape)

        # Iterative decode
        x_hat = self.decoder(z_noisy, snr_db)

        return x_hat

    def run_k_iterations(
        self, x: torch.Tensor, snr_db: float, k: int
    ) -> torch.Tensor:
        """Forward pass with a specific number of decoder iterations.

        Args:
            x: Input images (B, 3, H, W) in [0, 1].
            snr_db: Channel SNR in dB.
            k: Number of decoder iterations to run.

        Returns:
            Reconstructed images (B, 3, H, W) in [0, 1].
        """
        z = self.encoder(x, snr_db)
        z_noisy, _ = awgn_channel(z.flatten(1), snr_db)
        z_noisy = z_noisy.view(z.shape)
        return self.decoder(z_noisy, snr_db, num_iterations=k)

    @staticmethod
    def compute_loss(
        x: torch.Tensor, x_hat: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE reconstruction loss.

        Args:
            x: Original images (B, 3, H, W).
            x_hat: Reconstructed images (B, 3, H, W).

        Returns:
            MSE loss scalar.
        """
        return F.mse_loss(x_hat, x, reduction="mean")

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_step_sizes(self) -> list[float]:
        """Return learned step sizes from the decoder."""
        return self.decoder.get_step_sizes()
