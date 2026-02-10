"""DeepJSCC: fully-convolutional autoencoder JSCC (no VAE components).

Named VAEJSCC for backward compatibility with downstream code (diffusion model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.channel.awgn import awgn_channel
from src.channel.utils import normalize_power
from src.models.vae_jscc.decoder import JSCCDecoder
from src.models.vae_jscc.encoder import JSCCEncoder


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
    """DeepJSCC-style fully-convolutional JSCC.

    Despite the class name (kept for backward compatibility), this is NOT a VAE.
    It is a clean autoencoder: encoder -> power normalize -> AWGN -> decoder.
    No mu/log_sigma, no reparameterization, no KL divergence.
    """

    def __init__(
        self,
        latent_channels: int = 192,
        snr_embed_dim: int = 256,
        base_channels: int = 64,
    ) -> None:
        """Initialize DeepJSCC.

        Args:
            latent_channels: Number of latent channels. Controls bandwidth ratio:
                rho = latent_channels / (3 * 16^2).
            snr_embed_dim: Accepted for backward compatibility (unused).
            base_channels: Base channel count for conv layers.
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = JSCCEncoder(latent_channels, base_channels)
        self.decoder = JSCCDecoder(latent_channels, base_channels)

    def forward(
        self,
        x: torch.Tensor,
        snr_db: float,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Full forward pass: encode -> power normalize -> channel -> decode.

        Args:
            x: Input images (B, 3, H, W) in [0, 1]. H, W must be divisible by 16.
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Tuple of (x_hat, z, None).
            x_hat: Reconstructed images (B, 3, H, W) in [0, 1].
            z: Latent channel symbols (B, k) flattened, after power normalization.
            None: Placeholder for backward compatibility.
        """
        # Encode
        z_spatial = self.encoder(x, snr_db)
        spatial_shape = z_spatial.shape

        # Power normalize
        z = z_spatial.flatten(1)
        z = normalize_power(z, target_power=1.0, mode="per_sample")

        # AWGN channel
        z_noisy, _ = awgn_channel(z, snr_db)

        # Reshape back and decode
        z_noisy_spatial = z_noisy.view(spatial_shape)
        x_hat = self.decoder(z_noisy_spatial, snr_db)

        return x_hat, z, None

    @staticmethod
    def compute_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor | None = None,
        log_sigma: torch.Tensor | None = None,
        beta: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute MSE + MS-SSIM combined loss. No KL.

        Args:
            x: Original images (B, 3, H, W).
            x_hat: Reconstructed images (B, 3, H, W).
            z: Ignored (backward compat).
            log_sigma: Ignored (backward compat).
            beta: Ignored (backward compat).

        Returns:
            Tuple of (total_loss, mse_loss, msssim_loss).
        """
        mse = F.mse_loss(x_hat, x, reduction="mean")
        try:
            from pytorch_msssim import ms_ssim
            msssim_val = ms_ssim(
                x_hat.clamp(0, 1), x, data_range=1.0, size_average=True,
                win_size=7, weights=[0.0448, 0.2856, 0.3001, 0.2363],
            )
            msssim_loss = 1.0 - msssim_val
        except Exception:
            msssim_loss = torch.tensor(0.0, device=x.device)
        loss = 0.7 * mse + 0.3 * msssim_loss
        return loss, mse, msssim_loss

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
