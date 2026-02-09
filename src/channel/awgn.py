"""Additive White Gaussian Noise (AWGN) channel model."""

import torch

from src.channel.utils import snr_to_noise_std


def awgn_channel(
    z: torch.Tensor, snr_db: float, signal_power: float = 1.0
) -> tuple[torch.Tensor, float]:
    """Pass a signal through an AWGN channel.

    Args:
        z: Transmitted signal tensor (any shape). Works with real or complex tensors.
        snr_db: Channel SNR in decibels.
        signal_power: Average signal power used for noise calibration.

    Returns:
        Tuple of (noisy_signal, noise_std).
    """
    noise_std = snr_to_noise_std(snr_db, signal_power)
    noise = torch.randn_like(z) * noise_std
    z_noisy = z + noise
    return z_noisy, noise_std
