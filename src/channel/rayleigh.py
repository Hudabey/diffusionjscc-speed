"""Rayleigh flat-fading channel model (single-tap)."""

import torch

from src.channel.utils import snr_to_noise_std


def rayleigh_channel(
    z: torch.Tensor, snr_db: float, signal_power: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Pass a signal through a Rayleigh fading channel.

    For real-valued z, the fading coefficient h is Rayleigh-distributed
    (magnitude of a complex Gaussian), applied element-wise.

    Received signal: y = h * z + n, where n ~ N(0, sigma^2).

    Args:
        z: Transmitted signal tensor (any shape, real-valued).
        snr_db: Channel SNR in decibels.
        signal_power: Average signal power used for noise calibration.

    Returns:
        Tuple of (noisy_faded_signal, fading_coefficients, noise_std).
    """
    noise_std = snr_to_noise_std(snr_db, signal_power)

    # Rayleigh fading: h = |h_complex| where h_complex ~ CN(0, 1)
    # |CN(0,1)| has Rayleigh distribution with scale 1/sqrt(2)
    h_real = torch.randn_like(z)
    h_imag = torch.randn_like(z)
    h = (h_real ** 2 + h_imag ** 2).sqrt()

    # Faded and noisy signal
    noise = torch.randn_like(z) * noise_std
    z_noisy = h * z + noise

    return z_noisy, h, noise_std
