"""Image quality metrics: PSNR, SSIM, MS-SSIM, LPIPS."""

import torch
import lpips as lpips_lib
from pytorch_msssim import ssim as _ssim_fn, ms_ssim as _ms_ssim_fn

# Module-level LPIPS network (lazy-loaded to avoid import-time downloads)
_lpips_net: lpips_lib.LPIPS | None = None


def _get_lpips_net(net: str = "alex") -> lpips_lib.LPIPS:
    """Get or create the LPIPS network (singleton)."""
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = lpips_lib.LPIPS(net=net, verbose=False)
    return _lpips_net


def compute_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        x: Original images, shape (B, C, H, W), range [0, 1].
        x_hat: Reconstructed images, same shape and range.

    Returns:
        Dict with 'per_sample' (B,) and 'mean' scalar, both in dB.
    """
    mse = (x - x_hat).pow(2).mean(dim=(1, 2, 3))  # (B,)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
    return {"per_sample": psnr, "mean": psnr.mean()}


def compute_ssim(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Compute Structural Similarity Index.

    Args:
        x: Original images, shape (B, C, H, W), range [0, 1].
        x_hat: Reconstructed images, same shape and range.

    Returns:
        Scalar SSIM value.
    """
    return _ssim_fn(x, x_hat, data_range=1.0, size_average=True)


def compute_msssim(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Compute Multi-Scale Structural Similarity Index.

    Args:
        x: Original images, shape (B, C, H, W), range [0, 1].
            Minimum spatial size: 161x161 for default 5-scale MS-SSIM.
        x_hat: Reconstructed images, same shape and range.

    Returns:
        Scalar MS-SSIM value.
    """
    return _ms_ssim_fn(x, x_hat, data_range=1.0, size_average=True)


def compute_lpips(
    x: torch.Tensor, x_hat: torch.Tensor, net: str = "alex"
) -> torch.Tensor:
    """Compute Learned Perceptual Image Patch Similarity.

    Args:
        x: Original images, shape (B, C, H, W), range [0, 1].
        x_hat: Reconstructed images, same shape and range.
        net: Backbone network ('alex' or 'vgg').

    Returns:
        Scalar mean LPIPS value (lower is better).
    """
    lpips_net = _get_lpips_net(net)
    device = x.device
    lpips_net = lpips_net.to(device)

    # LPIPS expects [-1, 1] range
    x_scaled = x * 2.0 - 1.0
    x_hat_scaled = x_hat * 2.0 - 1.0

    with torch.no_grad():
        score = lpips_net(x_scaled, x_hat_scaled)
    return score.mean()


def compute_all_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]:
    """Compute all image quality metrics.

    Args:
        x: Original images, shape (B, C, H, W), range [0, 1].
        x_hat: Reconstructed images, same shape and range.

    Returns:
        Dict with 'psnr', 'ssim', 'ms_ssim', 'lpips' as float values.
    """
    psnr_result = compute_psnr(x, x_hat)

    result = {"psnr": psnr_result["mean"].item()}

    result["ssim"] = compute_ssim(x, x_hat).item()

    # MS-SSIM requires minimum 161x161 spatial size
    _, _, h, w = x.shape
    if h >= 161 and w >= 161:
        result["ms_ssim"] = compute_msssim(x, x_hat).item()
    else:
        result["ms_ssim"] = float("nan")

    result["lpips"] = compute_lpips(x, x_hat).item()

    return result
