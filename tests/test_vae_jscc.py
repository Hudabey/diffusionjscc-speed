"""Tests for VAE-JSCC model components and training."""

import torch
import pytest

from src.models.vae_jscc.encoder import Encoder
from src.models.vae_jscc.decoder import Decoder
from src.models.vae_jscc.model import VAEJSCC, pad_to_multiple


# Use tiny models and small inputs for fast tests
LATENT_CH = 32
SNR_EMBED_DIM = 32
BASE_CH = 16
INPUT_SIZE = 32  # 32/16 = 2×2 bottleneck → latent_dim = 32*4 = 128
BATCH_SIZE = 2


def _make_model(**kwargs) -> VAEJSCC:
    """Create a small test model."""
    return VAEJSCC(
        latent_channels=kwargs.get("latent_channels", LATENT_CH),
        snr_embed_dim=kwargs.get("snr_embed_dim", SNR_EMBED_DIM),
        base_channels=kwargs.get("base_channels", BASE_CH),
    )


def _make_input(batch_size: int = BATCH_SIZE, size: int = INPUT_SIZE) -> torch.Tensor:
    """Create a random input image batch."""
    return torch.rand(batch_size, 3, size, size)


class TestEncoder:
    def test_output_shape(self):
        """Encoder produces mu and log_sigma spatial maps with correct shapes."""
        enc = Encoder(LATENT_CH, SNR_EMBED_DIM, BASE_CH)
        x = _make_input()
        mu, log_sigma = enc(x, snr_db=10.0)
        # 32/16 = 2 → spatial size 2×2
        assert mu.shape == (BATCH_SIZE, LATENT_CH, 2, 2)
        assert log_sigma.shape == (BATCH_SIZE, LATENT_CH, 2, 2)

    def test_different_input_sizes(self):
        """Encoder handles different input spatial sizes."""
        enc = Encoder(LATENT_CH, SNR_EMBED_DIM, BASE_CH)
        for size in [32, 48, 64]:
            x = torch.rand(1, 3, size, size)
            mu, log_sigma = enc(x, snr_db=10.0)
            expected_spatial = size // 16
            assert mu.shape == (1, LATENT_CH, expected_spatial, expected_spatial)


class TestDecoder:
    def test_output_shape(self):
        """Decoder produces image with correct spatial size (16× latent spatial)."""
        dec = Decoder(LATENT_CH, SNR_EMBED_DIM, BASE_CH)
        z = torch.randn(BATCH_SIZE, LATENT_CH, 2, 2)
        out = dec(z, snr_db=10.0)
        assert out.shape == (BATCH_SIZE, 3, 32, 32)

    def test_output_range(self):
        """Decoder output is in [0, 1] due to sigmoid."""
        dec = Decoder(LATENT_CH, SNR_EMBED_DIM, BASE_CH)
        z = torch.randn(BATCH_SIZE, LATENT_CH, 2, 2)
        out = dec(z, snr_db=10.0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_larger_spatial(self):
        """Decoder upsamples correctly from larger latent spatial size."""
        dec = Decoder(LATENT_CH, SNR_EMBED_DIM, BASE_CH)
        z = torch.randn(1, LATENT_CH, 4, 4)  # would come from 64×64 input
        out = dec(z, snr_db=10.0)
        assert out.shape == (1, 3, 64, 64)


class TestVAEJSCC:
    def test_model_forward(self):
        """Full forward pass: output matches input spatial size and is in [0, 1]."""
        model = _make_model()
        x = _make_input()
        x_hat, mu, log_sigma = model(x, snr_db=10.0)
        assert x_hat.shape == x.shape
        assert x_hat.min() >= 0.0
        assert x_hat.max() <= 1.0
        # Flattened latent: 32 channels × 2×2 = 128
        assert mu.shape == (BATCH_SIZE, LATENT_CH * 2 * 2)
        assert log_sigma.shape == (BATCH_SIZE, LATENT_CH * 2 * 2)

    def test_model_differentiable(self):
        """Loss backpropagation works and produces gradients on all parameters."""
        model = _make_model()
        x = _make_input()
        x_hat, mu, log_sigma = model(x, snr_db=10.0)

        loss, recon_loss, kl_loss = model.compute_loss(
            x, x_hat, mu, log_sigma, beta=0.001
        )
        loss.backward()

        n_with_grad = 0
        n_total = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                n_total += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    n_with_grad += 1

        assert n_with_grad / n_total > 0.9, (
            f"Only {n_with_grad}/{n_total} params have nonzero gradients"
        )

    def test_loss_components(self):
        """Reconstruction and KL losses are non-negative."""
        model = _make_model()
        x = _make_input()
        x_hat, mu, log_sigma = model(x, snr_db=10.0)

        loss, recon_loss, kl_loss = model.compute_loss(
            x, x_hat, mu, log_sigma, beta=0.001
        )

        assert recon_loss.item() >= 0
        assert kl_loss.item() >= 0
        assert loss.item() >= recon_loss.item()

    def test_snr_conditioning(self):
        """Different SNR values produce different outputs."""
        model = _make_model()
        model.eval()
        x = _make_input(batch_size=1)

        with torch.no_grad():
            out_low, _, _ = model(x, snr_db=0.0)
            out_high, _, _ = model(x, snr_db=20.0)

        diff = (out_low - out_high).abs().mean().item()
        assert diff > 1e-6, f"SNR conditioning had no effect (diff={diff})"

    def test_power_constraint(self):
        """After encoding + power normalization, z has approximately unit power."""
        model = _make_model()
        x = _make_input(batch_size=4)
        mu_map, logsig_map = model.encoder(x, snr_db=10.0)
        z = model.reparameterize(mu_map.flatten(1), logsig_map.flatten(1))

        from src.channel.utils import normalize_power
        z_norm = normalize_power(z, target_power=1.0, mode="per_sample")

        power = (z_norm ** 2).mean(dim=1)
        for i in range(power.shape[0]):
            assert abs(power[i].item() - 1.0) < 0.01, (
                f"Sample {i} power = {power[i].item()}, expected ~1.0"
            )

    def test_small_overfit(self):
        """Training on 1 batch for 100 steps reduces loss by >50%."""
        model = _make_model()
        model.train()

        x = _make_input(batch_size=4, size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initial loss
        x_hat, mu, log_sigma = model(x, snr_db=10.0)
        initial_loss, _, _ = model.compute_loss(x, x_hat, mu, log_sigma, beta=0.0001)
        initial_loss_val = initial_loss.item()

        # Train 100 steps
        for _ in range(100):
            x_hat, mu, log_sigma = model(x, snr_db=10.0)
            loss, _, _ = model.compute_loss(x, x_hat, mu, log_sigma, beta=0.0001)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        reduction = (initial_loss_val - final_loss) / initial_loss_val
        assert reduction > 0.5, (
            f"Loss only reduced by {reduction:.1%} "
            f"(from {initial_loss_val:.4f} to {final_loss:.4f}), expected >50%"
        )

    def test_model_parameter_count(self):
        """Full-scale model parameter count is in a reasonable range (5M–50M)."""
        model = VAEJSCC(latent_channels=192, snr_embed_dim=256, base_channels=64)
        n_params = model.count_parameters()
        assert 5_000_000 <= n_params <= 50_000_000, (
            f"Parameter count {n_params:,} outside expected range [5M, 50M]"
        )

    def test_reparameterize(self):
        """Reparameterization produces correct statistics."""
        mu = torch.zeros(1000, 128)
        log_sigma = torch.zeros(1000, 128)  # sigma=1 → standard normal

        z = VAEJSCC.reparameterize(mu, log_sigma)

        assert abs(z.mean().item()) < 0.1
        assert abs(z.std().item() - 1.0) < 0.1

    def test_variable_input_size(self):
        """Model handles different input sizes at eval time."""
        model = _make_model()
        model.eval()
        for size in [32, 48, 64]:
            x = torch.rand(1, 3, size, size)
            with torch.no_grad():
                x_hat, mu, log_sigma = model(x, snr_db=10.0)
            assert x_hat.shape == x.shape


class TestPadToMultiple:
    def test_no_pad_needed(self):
        """Input already a multiple — no padding."""
        x = torch.rand(1, 3, 32, 32)
        out = pad_to_multiple(x, 16)
        assert out.shape == (1, 3, 32, 32)

    def test_pad_needed(self):
        """Input not a multiple — gets padded."""
        x = torch.rand(1, 3, 30, 25)
        out = pad_to_multiple(x, 16)
        assert out.shape[2] % 16 == 0
        assert out.shape[3] % 16 == 0
        assert out.shape[2] >= 30
        assert out.shape[3] >= 25

    def test_content_preserved(self):
        """Original content is preserved in top-left corner."""
        x = torch.rand(1, 3, 30, 25)
        out = pad_to_multiple(x, 16)
        assert torch.allclose(out[:, :, :30, :25], x)
