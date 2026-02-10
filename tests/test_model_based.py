"""Tests for Model-Based JSCC model components and training."""

import torch
import pytest

from src.models.model_based.encoder import SKEncoder
from src.models.model_based.decoder import UnrolledDecoder
from src.models.model_based.model import ModelBasedJSCC, pad_to_multiple


# Tiny config for fast tests
LATENT_CH = 4
NUM_ITERATIONS = 2
DENOISER_CH = 16
SNR_EMBED_DIM = 16
BASE_CH = 16
INPUT_SIZE = 32  # 32/8 = 4x4 bottleneck
BATCH_SIZE = 2


def _make_model(**kwargs) -> ModelBasedJSCC:
    """Create a small test model."""
    return ModelBasedJSCC(
        latent_channels=kwargs.get("latent_channels", LATENT_CH),
        num_iterations=kwargs.get("num_iterations", NUM_ITERATIONS),
        denoiser_channels=kwargs.get("denoiser_channels", DENOISER_CH),
        snr_embed_dim=kwargs.get("snr_embed_dim", SNR_EMBED_DIM),
        base_channels=kwargs.get("base_channels", BASE_CH),
    )


def _make_input(batch_size: int = BATCH_SIZE, size: int = INPUT_SIZE) -> torch.Tensor:
    """Create a random input image batch."""
    return torch.rand(batch_size, 3, size, size)


class TestSKEncoder:
    def test_encoder_output_shape(self):
        """Encoder produces latent with correct spatial shape (8x downsampled)."""
        enc = SKEncoder(LATENT_CH, SNR_EMBED_DIM, BASE_CH)
        x = _make_input()
        z = enc(x, snr_db=10.0)
        # 32/8 = 4 -> spatial size 4x4
        assert z.shape == (BATCH_SIZE, LATENT_CH, 4, 4)

    def test_encoder_power_constraint(self):
        """After encoding, z has approximately unit power per sample."""
        enc = SKEncoder(LATENT_CH, SNR_EMBED_DIM, BASE_CH)
        x = _make_input(batch_size=4)
        z = enc(x, snr_db=10.0)

        power = (z.flatten(1) ** 2).mean(dim=1)
        for i in range(power.shape[0]):
            assert abs(power[i].item() - 1.0) < 0.01, (
                f"Sample {i} power = {power[i].item()}, expected ~1.0"
            )


class TestUnrolledDecoder:
    def test_decoder_output_shape(self):
        """Decoder produces image with correct spatial size (8x latent spatial)."""
        dec = UnrolledDecoder(LATENT_CH, NUM_ITERATIONS, DENOISER_CH, SNR_EMBED_DIM, BASE_CH)
        z = torch.randn(BATCH_SIZE, LATENT_CH, 4, 4)
        out = dec(z, snr_db=10.0)
        assert out.shape == (BATCH_SIZE, 3, 32, 32)

    def test_decoder_output_range(self):
        """Decoder output is in [0, 1] due to sigmoid."""
        dec = UnrolledDecoder(LATENT_CH, NUM_ITERATIONS, DENOISER_CH, SNR_EMBED_DIM, BASE_CH)
        z = torch.randn(BATCH_SIZE, LATENT_CH, 4, 4)
        out = dec(z, snr_db=10.0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestModelBasedJSCC:
    def test_full_model_forward(self):
        """Full forward pass: output matches input spatial size and is in [0, 1]."""
        model = _make_model()
        x = _make_input()
        x_hat = model(x, snr_db=10.0)
        assert x_hat.shape == x.shape
        assert x_hat.min() >= 0.0
        assert x_hat.max() <= 1.0

    def test_model_differentiable(self):
        """Loss backpropagation works and produces gradients on all parameters."""
        model = _make_model()
        x = _make_input()
        x_hat = model(x, snr_db=10.0)

        loss = model.compute_loss(x, x_hat)
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

    def test_step_sizes_learnable(self):
        """Step sizes (alphas) are parameters and have gradients after backward."""
        model = _make_model()
        x = _make_input()
        x_hat = model(x, snr_db=10.0)
        loss = model.compute_loss(x, x_hat)
        loss.backward()

        step_sizes = model.get_step_sizes()
        assert len(step_sizes) == NUM_ITERATIONS
        for i, alpha in enumerate(model.decoder.alphas):
            assert alpha.requires_grad
            assert alpha.grad is not None, f"Alpha {i} has no gradient"

    def test_iteration_convergence(self):
        """Different iteration counts produce different outputs."""
        model = _make_model(num_iterations=6)
        model.eval()
        x = _make_input(batch_size=1)

        with torch.no_grad():
            out_k1 = model.run_k_iterations(x, snr_db=10.0, k=1)
            out_k6 = model.run_k_iterations(x, snr_db=10.0, k=6)

        diff = (out_k1 - out_k6).abs().mean().item()
        assert diff > 1e-6, f"K=1 and K=6 produced same output (diff={diff})"

    def test_param_count(self):
        """Full-scale model has fewer than 5M parameters."""
        model = ModelBasedJSCC(
            latent_channels=12,
            num_iterations=6,
            denoiser_channels=64,
            snr_embed_dim=128,
            base_channels=64,
        )
        n_params = model.count_parameters()
        assert n_params < 5_000_000, (
            f"Parameter count {n_params:,} exceeds 5M target"
        )

    def test_small_overfit(self):
        """Training on 1 batch for 100 steps reduces loss by >50%."""
        model = _make_model()
        model.train()

        x = _make_input(batch_size=4, size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initial loss
        x_hat = model(x, snr_db=10.0)
        initial_loss = model.compute_loss(x, x_hat).item()

        # Train 100 steps
        for _ in range(100):
            x_hat = model(x, snr_db=10.0)
            loss = model.compute_loss(x, x_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        reduction = (initial_loss - final_loss) / initial_loss
        assert reduction > 0.5, (
            f"Loss only reduced by {reduction:.1%} "
            f"(from {initial_loss:.4f} to {final_loss:.4f}), expected >50%"
        )

    def test_snr_affects_output(self):
        """Different SNR values produce different outputs."""
        model = _make_model()
        model.eval()
        x = _make_input(batch_size=1)

        with torch.no_grad():
            out_low = model(x, snr_db=0.0)
            out_high = model(x, snr_db=20.0)

        diff = (out_low - out_high).abs().mean().item()
        assert diff > 1e-6, f"SNR conditioning had no effect (diff={diff})"

    def test_variable_input_size(self):
        """Model handles different input sizes at eval time."""
        model = _make_model()
        model.eval()
        for size in [32, 48, 64]:
            x = torch.rand(1, 3, size, size)
            with torch.no_grad():
                x_hat = model(x, snr_db=10.0)
            assert x_hat.shape == x.shape


class TestPadToMultiple8:
    def test_no_pad_needed(self):
        """Input already a multiple of 8 - no padding."""
        x = torch.rand(1, 3, 32, 32)
        out = pad_to_multiple(x, 8)
        assert out.shape == (1, 3, 32, 32)

    def test_pad_needed(self):
        """Input not a multiple of 8 - gets padded."""
        x = torch.rand(1, 3, 30, 25)
        out = pad_to_multiple(x, 8)
        assert out.shape[2] % 8 == 0
        assert out.shape[3] % 8 == 0
        assert out.shape[2] >= 30
        assert out.shape[3] >= 25
