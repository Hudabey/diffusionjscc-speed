"""Unrolled iterative decoder for model-based JSCC."""

import torch
import torch.nn as nn

from src.models.model_based.encoder import FiLMConditioner, ResBlock


def _num_groups(channels: int, target: int = 8) -> int:
    """Find the largest number of groups <= target that divides channels."""
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SmallDenoiser(nn.Module):
    """Lightweight CNN denoiser for one unrolled iteration (~50K params).

    Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + residual.
    SNR conditioning via additive embedding before first conv.
    """

    def __init__(self, channels: int, snr_embed_dim: int) -> None:
        super().__init__()
        self.snr_proj = nn.Linear(snr_embed_dim, channels)
        ng = _num_groups(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(ng, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(ng, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, snr_embed: torch.Tensor) -> torch.Tensor:
        """Apply denoiser with SNR conditioning.

        Args:
            x: Input feature map (B, C, H, W).
            snr_embed: SNR embedding (B, snr_embed_dim).

        Returns:
            Denoised feature map (B, C, H, W).
        """
        # Additive SNR conditioning
        snr_bias = self.snr_proj(snr_embed).unsqueeze(-1).unsqueeze(-1)
        h = x + snr_bias
        return x + self.net(h)


class UnrolledDecoder(nn.Module):
    """Unrolled iterative decoder with learned step sizes and denoisers.

    K iterations of: z = z - alpha * gradient_step + denoiser(z, snr)
    followed by a reconstruction network to upsample latent to image.
    """

    SNR_MIN = -5.0
    SNR_MAX = 25.0

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

        # SNR embedding MLP (shared across iterations)
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, snr_embed_dim),
            nn.ReLU(inplace=True),
        )

        # Learned step sizes (one per iteration)
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_iterations)]
        )

        # Separate denoiser per iteration
        self.denoisers = nn.ModuleList(
            [SmallDenoiser(latent_channels, snr_embed_dim) for _ in range(num_iterations)]
        )

        # Reconstruction network: latent -> image (8x upsample)
        ch = base_channels
        self.recon = nn.Sequential(
            ResBlock(latent_channels),
            nn.ConvTranspose2d(latent_channels, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            ResBlock(ch * 2),
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            ResBlock(ch),
            nn.ConvTranspose2d(ch, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        # FiLM conditioning in reconstruction (applied after first ResBlock)
        self.recon_film = FiLMConditioner(snr_embed_dim, latent_channels)

    def _get_snr_embed(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Compute SNR embedding."""
        B = z.shape[0]
        snr_norm = (snr_db - self.SNR_MIN) / (self.SNR_MAX - self.SNR_MIN)
        snr_t = torch.tensor([[snr_norm]], device=z.device, dtype=z.dtype).expand(B, 1)
        return self.snr_mlp(snr_t)

    def forward(
        self, z_received: torch.Tensor, snr_db: float, num_iterations: int | None = None
    ) -> torch.Tensor:
        """Decode received latent symbols to reconstructed image.

        Args:
            z_received: Noisy latent from channel (B, latent_channels, H_lat, W_lat).
            snr_db: Channel SNR in dB (scalar).
            num_iterations: Override number of iterations (for eval). Defaults to self.num_iterations.

        Returns:
            Reconstructed image (B, 3, H_lat*8, W_lat*8) in [0, 1].
        """
        K = num_iterations if num_iterations is not None else self.num_iterations
        K = min(K, self.num_iterations)  # Can't exceed available denoisers

        snr_embed = self._get_snr_embed(z_received, snr_db)

        # Unrolled iterations in latent space
        z = z_received
        for i in range(K):
            # Gradient step: move toward received signal
            grad = z - z_received
            z = z - self.alphas[i] * grad + (self.denoisers[i](z, snr_embed) - z)

        # Apply FiLM conditioning before reconstruction
        z = self.recon_film(self.recon[0](z), snr_embed)  # ResBlock + FiLM
        x_hat = self.recon[1:](z)  # rest of reconstruction

        return x_hat

    def get_step_sizes(self) -> list[float]:
        """Return current learned step sizes for logging."""
        return [alpha.item() for alpha in self.alphas]
