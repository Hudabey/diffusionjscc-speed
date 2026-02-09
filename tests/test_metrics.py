"""Tests for evaluation metrics."""

import torch
import pytest

from src.eval.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_all_metrics,
)


def test_psnr_identical():
    x = torch.rand(2, 3, 64, 64)
    result = compute_psnr(x, x)
    assert result["mean"].item() > 60.0


def test_psnr_known():
    """For known MSE, PSNR should match the formula: 10*log10(1/MSE)."""
    x = torch.zeros(1, 3, 32, 32)
    # Add known noise level
    mse_target = 0.01
    x_hat = x + (mse_target ** 0.5)  # uniform offset → MSE = mse_target
    result = compute_psnr(x, x_hat)
    expected_psnr = 10.0 * torch.log10(torch.tensor(1.0 / mse_target)).item()
    assert abs(result["mean"].item() - expected_psnr) < 0.5


def test_psnr_range():
    torch.manual_seed(42)
    x = torch.rand(4, 3, 64, 64)
    noise = torch.randn_like(x) * 0.1
    x_hat = (x + noise).clamp(0, 1)
    result = compute_psnr(x, x_hat)
    psnr = result["mean"].item()
    assert 0.0 < psnr < 60.0


def test_ssim_identical():
    x = torch.rand(2, 3, 64, 64)
    ssim = compute_ssim(x, x)
    assert ssim.item() > 0.99


def test_ssim_range():
    torch.manual_seed(42)
    x = torch.rand(2, 3, 64, 64)
    noise = torch.randn_like(x) * 0.2
    x_hat = (x + noise).clamp(0, 1)
    ssim = compute_ssim(x, x_hat)
    assert 0.0 <= ssim.item() <= 1.0


def test_lpips_identical():
    x = torch.rand(2, 3, 64, 64)
    score = compute_lpips(x, x)
    assert score.item() < 0.01


def test_lpips_different():
    x = torch.zeros(2, 3, 64, 64)
    x_hat = torch.ones(2, 3, 64, 64)
    score = compute_lpips(x, x_hat)
    assert score.item() > 0.3


def test_compute_all_metrics():
    torch.manual_seed(42)
    x = torch.rand(2, 3, 256, 256)
    noise = torch.randn_like(x) * 0.1
    x_hat = (x + noise).clamp(0, 1)
    metrics = compute_all_metrics(x, x_hat)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "ms_ssim" in metrics
    assert "lpips" in metrics
    assert isinstance(metrics["psnr"], float)
