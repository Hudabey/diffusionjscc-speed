"""Tests for Diffusion-JSCC model components."""

import torch
import pytest

from src.models.diffusion_jscc.unet import ConditionalUNet, sinusoidal_embedding
from src.models.diffusion_jscc.diffusion import GaussianDiffusion, cosine_beta_schedule
from src.models.diffusion_jscc.sampler import ddim_sample


# Tiny model config for fast tests
TINY_UNET_KWARGS = dict(
    in_channels=6,
    out_channels=3,
    base_channels=16,
    channel_mults=(1, 2, 4),
    num_res_blocks=1,
    attention_levels=(2,),
    time_embed_dim=64,
    snr_embed_dim=64,
)
IMG_SIZE = 32
BATCH = 2
T_TEST = 10  # tiny timestep count for tests


def _make_unet(**overrides):
    kwargs = {**TINY_UNET_KWARGS, **overrides}
    return ConditionalUNet(**kwargs)


def _make_diffusion(**overrides):
    unet = _make_unet(**overrides)
    return GaussianDiffusion(unet, T=T_TEST)


class TestUNet:
    def test_output_shape(self):
        """UNet produces output matching spatial dims with 3 channels."""
        unet = _make_unet()
        x_noisy = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        x_init = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        t = torch.randint(0, T_TEST, (BATCH,))
        out = unet(x_noisy, x_init, t, snr_db=10.0)
        assert out.shape == (BATCH, 3, IMG_SIZE, IMG_SIZE)

    def test_time_conditioning(self):
        """Different timesteps produce different outputs."""
        unet = _make_unet()
        unet.eval()
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        x_init = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

        with torch.no_grad():
            out_t0 = unet(x, x_init, torch.tensor([0]), snr_db=10.0)
            out_t9 = unet(x, x_init, torch.tensor([9]), snr_db=10.0)

        diff = (out_t0 - out_t9).abs().mean().item()
        assert diff > 1e-4, f"Time conditioning had no effect (diff={diff})"

    def test_snr_conditioning(self):
        """Different SNR values produce different outputs."""
        unet = _make_unet()
        unet.eval()
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        x_init = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        t = torch.tensor([5])

        with torch.no_grad():
            out_low = unet(x, x_init, t, snr_db=0.0)
            out_high = unet(x, x_init, t, snr_db=20.0)

        diff = (out_low - out_high).abs().mean().item()
        assert diff > 1e-4, f"SNR conditioning had no effect (diff={diff})"

    def test_parameter_count(self):
        """Full-scale UNet param count is in 15M-30M range."""
        unet = ConditionalUNet(
            in_channels=6, out_channels=3, base_channels=64,
            channel_mults=(1, 2, 4, 4), num_res_blocks=2,
            attention_levels=(2,),
        )
        n = unet.count_parameters()
        assert 10_000_000 <= n <= 50_000_000, f"Param count {n:,} outside range"


class TestDiffusion:
    def test_cosine_schedule(self):
        """Betas in (0,1), alphas_cumprod monotonically decreasing."""
        betas = cosine_beta_schedule(100)
        assert (betas > 0).all()
        assert (betas < 1).all()

        alphas_cumprod = torch.cumprod(1 - betas, dim=0)
        diffs = alphas_cumprod[1:] - alphas_cumprod[:-1]
        assert (diffs <= 0).all(), "alphas_cumprod not monotonically decreasing"

    def test_q_sample_shape(self):
        """Forward diffusion preserves shape."""
        diff = _make_diffusion()
        x = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        t = torch.randint(0, T_TEST, (BATCH,))
        x_noisy = diff.q_sample(x, t)
        assert x_noisy.shape == x.shape

    def test_q_sample_noise_increases(self):
        """Higher timestep → more noise (larger MSE from original)."""
        diff = _make_diffusion()
        x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
        noise = torch.randn_like(x)

        mse_early = (diff.q_sample(x, torch.tensor([1]), noise) - x).pow(2).mean().item()
        mse_late = (diff.q_sample(x, torch.tensor([T_TEST - 1]), noise) - x).pow(2).mean().item()
        assert mse_late > mse_early, "Later timestep should add more noise"

    def test_loss_computes(self):
        """Diffusion loss is a positive finite scalar."""
        diff = _make_diffusion()
        x = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        x_init = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        loss = diff.training_loss(x, x_init, snr_db=10.0)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_loss_backward(self):
        """Gradients flow to all UNet parameters."""
        diff = _make_diffusion()
        x = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        x_init = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        loss = diff.training_loss(x, x_init, snr_db=10.0)
        loss.backward()

        n_with_grad = sum(
            1 for p in diff.unet.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        n_total = sum(1 for p in diff.unet.parameters() if p.requires_grad)
        assert n_with_grad / n_total > 0.8, (
            f"Only {n_with_grad}/{n_total} params have nonzero gradients"
        )


class TestSampler:
    def test_ddim_sample_shape(self):
        """DDIM output has correct shape and is in [0, 1]."""
        diff = _make_diffusion()
        diff.eval()
        x_init = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
        out = ddim_sample(diff, x_init, snr_db=10.0, num_steps=5)
        assert out.shape == x_init.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_ddim_steps_matter(self):
        """Different step counts produce different outputs."""
        diff = _make_diffusion()
        diff.eval()
        torch.manual_seed(42)
        x_init = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)

        out_1 = ddim_sample(diff, x_init, snr_db=10.0, num_steps=1)
        out_5 = ddim_sample(diff, x_init, snr_db=10.0, num_steps=5)
        diff_val = (out_1 - out_5).abs().mean().item()
        assert diff_val > 1e-4, f"Different step counts gave same output (diff={diff_val})"

    def test_ddim_1_step(self):
        """Even 1 step produces a valid (non-NaN, non-garbage) output."""
        diff = _make_diffusion()
        diff.eval()
        x_init = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
        out = ddim_sample(diff, x_init, snr_db=10.0, num_steps=1)
        assert torch.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestFullPipeline:
    def test_pipeline_shapes(self):
        """Full pipeline produces valid outputs (using mock VAE)."""
        from src.models.vae_jscc.model import VAEJSCC
        from src.models.diffusion_jscc.model import DiffusionJSCC

        vae = VAEJSCC(latent_channels=32, snr_embed_dim=32, base_channels=16)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        diff = _make_diffusion()
        pipeline = DiffusionJSCC(vae, diff)

        x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
        x_refined, x_init = pipeline.sample(x, snr_db=10.0, num_steps=3)

        assert x_refined.shape == x.shape
        assert x_init.shape == x.shape
        assert x_refined.min() >= 0.0
        assert x_refined.max() <= 1.0

    def test_training_step(self):
        """Training step produces a finite scalar loss."""
        from src.models.vae_jscc.model import VAEJSCC
        from src.models.diffusion_jscc.model import DiffusionJSCC

        vae = VAEJSCC(latent_channels=32, snr_embed_dim=32, base_channels=16)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        diff = _make_diffusion()
        pipeline = DiffusionJSCC(vae, diff)

        x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
        loss = pipeline.training_step(x, snr_db=10.0)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
