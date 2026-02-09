"""Channel utility functions: power normalization and SNR conversions."""

import torch


def db_to_linear(db: float) -> float:
    """Convert decibels to linear scale."""
    return 10.0 ** (db / 10.0)


def linear_to_db(linear: float) -> float:
    """Convert linear scale to decibels."""
    return 10.0 * torch.log10(torch.tensor(linear, dtype=torch.float64)).item()


def snr_to_noise_std(snr_db: float, signal_power: float = 1.0) -> float:
    """Convert SNR in dB to noise standard deviation.

    Args:
        snr_db: Signal-to-noise ratio in decibels.
        signal_power: Average signal power (default 1.0).

    Returns:
        Noise standard deviation sigma.
    """
    snr_linear = db_to_linear(snr_db)
    return (signal_power / snr_linear) ** 0.5


def normalize_power(
    z: torch.Tensor, target_power: float = 1.0, mode: str = "per_batch"
) -> torch.Tensor:
    """Normalize transmitted signal to meet average power constraint.

    Args:
        z: Input tensor of any shape, with batch dimension first.
        target_power: Target average power.
        mode: 'per_batch' normalizes across the entire batch,
              'per_sample' normalizes each sample independently.

    Returns:
        Power-normalized tensor with the same shape as z.
    """
    if mode == "per_batch":
        current_power = (z ** 2).mean()
    elif mode == "per_sample":
        # Mean over all dims except batch
        dims = tuple(range(1, z.ndim))
        current_power = (z ** 2).mean(dim=dims, keepdim=True)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    scale = (target_power / (current_power + 1e-12)) ** 0.5
    return z * scale
