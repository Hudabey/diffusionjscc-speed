"""Tests for channel models and utilities."""

import torch
import pytest
from scipy import stats

from src.channel.awgn import awgn_channel
from src.channel.rayleigh import rayleigh_channel
from src.channel.utils import (
    db_to_linear,
    linear_to_db,
    normalize_power,
    snr_to_noise_std,
)


# ── AWGN tests ────────────────────────────────────────────────────────────────


def test_awgn_output_shape():
    z = torch.randn(8, 256)
    z_noisy, noise_std = awgn_channel(z, snr_db=10.0)
    assert z_noisy.shape == z.shape


def test_awgn_noise_variance():
    """At SNR=10dB with signal_power=1.0, empirical noise variance ≈ theoretical."""
    snr_db = 10.0
    signal_power = 1.0
    expected_var = signal_power / db_to_linear(snr_db)  # 0.1

    torch.manual_seed(0)
    z = torch.zeros(100000)  # zero signal to isolate noise
    z_noisy, noise_std = awgn_channel(z, snr_db=snr_db, signal_power=signal_power)
    empirical_var = z_noisy.var().item()

    assert abs(empirical_var - expected_var) / expected_var < 0.05


def test_awgn_differentiable():
    z = torch.randn(4, 64, requires_grad=True)
    z_noisy, _ = awgn_channel(z, snr_db=10.0)
    loss = z_noisy.sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.shape == z.shape


def test_awgn_high_snr():
    torch.manual_seed(42)
    z = torch.randn(8, 256)
    z_noisy, _ = awgn_channel(z, snr_db=100.0)
    mse = (z - z_noisy).pow(2).mean().item()
    assert mse < 1e-8


def test_awgn_low_snr():
    torch.manual_seed(42)
    z = torch.randn(8, 256)
    z_noisy, _ = awgn_channel(z, snr_db=-10.0)
    mse = (z - z_noisy).pow(2).mean().item()
    assert mse > 1.0


# ── Rayleigh tests ────────────────────────────────────────────────────────────


def test_rayleigh_output_shape():
    z = torch.randn(8, 256)
    z_noisy, h, noise_std = rayleigh_channel(z, snr_db=10.0)
    assert z_noisy.shape == z.shape
    assert h.shape == z.shape
    assert isinstance(noise_std, float)


def test_rayleigh_fading_distribution():
    """Fading coefficients should follow Rayleigh distribution.

    For Rayleigh with scale = 1/sqrt(2), the mean = sqrt(pi/2) * scale ≈ 0.8862.
    We use a KS test against the theoretical distribution.
    """
    torch.manual_seed(42)
    z = torch.ones(100000)
    _, h, _ = rayleigh_channel(z, snr_db=100.0)  # high SNR to isolate h

    h_np = h.numpy()
    # h = sqrt(h_real^2 + h_imag^2) where h_real, h_imag ~ N(0,1)
    # This is Rayleigh with scale = 1.0
    _, p_value = stats.kstest(h_np, "rayleigh", args=(0, 1.0))
    assert p_value > 0.01, f"KS test failed with p={p_value}"


def test_rayleigh_differentiable():
    z = torch.randn(4, 64, requires_grad=True)
    z_noisy, h, _ = rayleigh_channel(z, snr_db=10.0)
    loss = z_noisy.sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.shape == z.shape


# ── Utility tests ─────────────────────────────────────────────────────────────


def test_power_normalization_per_batch():
    torch.manual_seed(0)
    z = torch.randn(32, 128) * 5.0  # high power
    z_norm = normalize_power(z, target_power=1.0, mode="per_batch")
    actual_power = (z_norm ** 2).mean().item()
    assert abs(actual_power - 1.0) < 0.01


def test_power_normalization_per_sample():
    torch.manual_seed(0)
    z = torch.randn(32, 128) * 5.0
    z_norm = normalize_power(z, target_power=2.0, mode="per_sample")
    per_sample_power = (z_norm ** 2).mean(dim=1)
    for p in per_sample_power:
        assert abs(p.item() - 2.0) < 0.01


def test_snr_conversion_roundtrip():
    for db_val in [-10.0, 0.0, 10.0, 20.0, 30.0]:
        linear_val = db_to_linear(db_val)
        db_back = linear_to_db(linear_val)
        assert abs(db_back - db_val) < 1e-6, f"Roundtrip failed for {db_val} dB"
