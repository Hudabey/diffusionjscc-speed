"""Tests for digital baseline simulators."""

import torch
import pytest

from src.eval.digital_baselines import (
    jpeg_compress_decompress,
    webp_compress_decompress,
    channel_capacity_awgn,
    can_decode,
    shannon_bound_psnr,
    digital_baseline,
)
from src.eval.metrics import compute_psnr


def _make_smooth_image(h: int = 64, w: int = 64) -> torch.Tensor:
    """Create a smooth gradient image that compresses well (unlike random noise)."""
    x = torch.linspace(0, 1, w).unsqueeze(0).expand(h, w)
    y = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w)
    r = x
    g = y
    b = (x + y) / 2
    return torch.stack([r, g, b])


# ── Codec tests ───────────────────────────────────────────────────────────────


def test_jpeg_compress_decompress():
    img = _make_smooth_image()
    recon, bits = jpeg_compress_decompress(img, quality=50)
    assert recon.shape == img.shape
    psnr = compute_psnr(img.unsqueeze(0), recon.unsqueeze(0))
    assert psnr["mean"].item() > 20.0


def test_jpeg_bits_reasonable():
    img = _make_smooth_image()
    _, bits = jpeg_compress_decompress(img, quality=50)
    raw_bits = 3 * 64 * 64 * 8
    assert bits < raw_bits
    assert bits > 0


def test_webp_compress_decompress():
    img = _make_smooth_image()
    recon, bits = webp_compress_decompress(img, quality=50)
    assert recon.shape == img.shape
    psnr = compute_psnr(img.unsqueeze(0), recon.unsqueeze(0))
    assert psnr["mean"].item() > 20.0


# ── Capacity tests ────────────────────────────────────────────────────────────


def test_channel_capacity_known_values():
    # C(0dB) = 0.5 * log2(1 + 1) = 0.5 bits
    assert abs(channel_capacity_awgn(0.0) - 0.5) < 0.01
    # C(10dB) = 0.5 * log2(1 + 10) ≈ 1.73 bits
    assert abs(channel_capacity_awgn(10.0) - 1.73) < 0.01
    # C(20dB) = 0.5 * log2(1 + 100) ≈ 3.33 bits
    assert abs(channel_capacity_awgn(20.0) - 3.33) < 0.01


def test_can_decode_above_capacity():
    # Low rate + high SNR → success
    assert can_decode(source_rate_bpp=0.5, bandwidth_ratio=0.25, snr_db=20.0)


def test_cannot_decode_below_capacity():
    # High rate + low SNR → failure
    assert not can_decode(source_rate_bpp=2.0, bandwidth_ratio=0.25, snr_db=0.0)


# ── Shannon bound tests ──────────────────────────────────────────────────────


def test_shannon_bound_increases_with_snr():
    psnrs = [shannon_bound_psnr(snr, 0.25) for snr in [0, 5, 10, 15, 20]]
    for i in range(len(psnrs) - 1):
        assert psnrs[i + 1] > psnrs[i]


# ── End-to-end digital baseline test ─────────────────────────────────────────


def test_digital_baseline_cliff():
    """Verify cliff effect: high PSNR at high SNR, low PSNR at low SNR."""
    # Use smooth images that compress well (natural-image-like)
    images = torch.stack([_make_smooth_image(64, 64) for _ in range(4)])

    torch.manual_seed(42)
    result_high = digital_baseline(
        images, snr_db=25.0, bandwidth_ratio=0.25, codec="jpeg", quality=50,
    )
    torch.manual_seed(42)
    result_low = digital_baseline(
        images, snr_db=-5.0, bandwidth_ratio=0.25, codec="jpeg", quality=50,
    )

    # High SNR should succeed with good quality
    assert result_high["success_rate"] > 0.5
    # Low SNR should fail
    assert result_low["success_rate"] < 0.5
    # Gap should be large
    assert result_high["psnr"] - result_low["psnr"] > 15.0
