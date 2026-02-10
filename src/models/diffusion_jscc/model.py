"""DiffusionJSCC: full pipeline combining frozen VAE-JSCC backbone + diffusion refinement."""

import torch
import torch.nn as nn

from src.models.vae_jscc.model import VAEJSCC, pad_to_multiple
from src.models.diffusion_jscc.diffusion import GaussianDiffusion
from src.models.diffusion_jscc.sampler import ddim_sample
from src.models.diffusion_jscc.unet import ConditionalUNet


def load_vae_backbone(
    checkpoint_path: str,
    latent_channels: int = 192,
    snr_embed_dim: int = 256,
    device: torch.device | str = "cpu",
) -> VAEJSCC:
    """Load and freeze a pre-trained VAE-JSCC model.

    Args:
        checkpoint_path: Path to VAE-JSCC checkpoint.
        latent_channels: Must match the trained model.
        snr_embed_dim: Must match the trained model.
        device: Torch device.

    Returns:
        Frozen VAEJSCC model in eval mode.
    """
    model = VAEJSCC(latent_channels=latent_channels, snr_embed_dim=snr_embed_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


class DiffusionJSCC(nn.Module):
    """Full Diffusion-JSCC pipeline.

    Combines a frozen VAE-JSCC backbone (transmitter + initial receiver)
    with a trainable diffusion model that refines the reconstruction.
    """

    def __init__(
        self,
        vae: VAEJSCC,
        diffusion: GaussianDiffusion,
    ) -> None:
        """Initialize DiffusionJSCC.

        Args:
            vae: Frozen VAE-JSCC backbone.
            diffusion: Trainable GaussianDiffusion (wraps the UNet).
        """
        super().__init__()
        self.vae = vae
        self.diffusion = diffusion

    @torch.no_grad()
    def get_vae_reconstruction(
        self, x: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """Get VAE-JSCC reconstruction (frozen, no grad).

        Args:
            x: Input images (B, 3, H, W) in [0, 1].
            snr_db: Channel SNR in dB.

        Returns:
            VAE reconstruction (B, 3, H, W) in [0, 1].
        """
        x_padded = pad_to_multiple(x, 16)
        x_init, _, _ = self.vae(x_padded, snr_db)
        x_init = x_init[:, :, :x.shape[2], :x.shape[3]]
        return x_init.clamp(0, 1)

    def training_step(
        self, x: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        Args:
            x: Clean images (B, 3, H, W) in [0, 1].
            snr_db: Channel SNR in dB.

        Returns:
            Scalar diffusion loss.
        """
        x_init = self.get_vae_reconstruction(x, snr_db)
        return self.diffusion.training_loss(x, x_init, snr_db)

    @torch.no_grad()
    def sample(
        self, x: torch.Tensor, snr_db: float, num_steps: int = 5,
        t_start: int = 200,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full pipeline: VAE-JSCC → diffusion refinement.

        Args:
            x: Input images (B, 3, H, W) in [0, 1].
            snr_db: Channel SNR in dB.
            num_steps: Number of DDIM sampling steps.
            t_start: Starting timestep for refinement (lower = lighter refinement).

        Returns:
            Tuple of (x_refined, x_init).
            x_refined: Diffusion-refined output (B, 3, H, W).
            x_init: VAE-JSCC initial reconstruction (B, 3, H, W).
        """
        x_init = self.get_vae_reconstruction(x, snr_db)
        x_refined = ddim_sample(
            self.diffusion, x_init, snr_db,
            num_steps=num_steps, t_start=t_start,
        )
        return x_refined, x_init
