"""Digital baseline simulators using information-theoretic capacity bounds.

Simulates JPEG + ideal channel coding and WebP + ideal channel coding
transmission over AWGN channels. Uses Shannon capacity to determine
success/failure rather than implementing actual LDPC/Polar decoders.
"""

import io
from typing import Literal

import numpy as np
import torch
from PIL import Image

from src.eval.metrics import compute_psnr, compute_ssim


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (C, H, W) float tensor in [0,1] to a PIL Image."""
    arr = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a (C, H, W) float tensor in [0,1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def jpeg_compress_decompress(
    image: torch.Tensor, quality: int = 50
) -> tuple[torch.Tensor, int]:
    """Compress and decompress a single image with JPEG.

    Args:
        image: (C, H, W) float tensor in [0, 1].
        quality: JPEG quality 1-95.

    Returns:
        Tuple of (reconstructed tensor, bits used).
    """
    pil_img = _tensor_to_pil(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    bits_used = buf.tell() * 8
    buf.seek(0)
    recon = Image.open(buf).convert("RGB")
    return _pil_to_tensor(recon), bits_used


def webp_compress_decompress(
    image: torch.Tensor, quality: int = 50
) -> tuple[torch.Tensor, int]:
    """Compress and decompress a single image with WebP.

    Args:
        image: (C, H, W) float tensor in [0, 1].
        quality: WebP quality 1-100.

    Returns:
        Tuple of (reconstructed tensor, bits used).
    """
    pil_img = _tensor_to_pil(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="WEBP", quality=quality)
    bits_used = buf.tell() * 8
    buf.seek(0)
    recon = Image.open(buf).convert("RGB")
    return _pil_to_tensor(recon), bits_used


def channel_capacity_awgn(snr_db: float) -> float:
    """Shannon capacity in bits per real channel use for AWGN.

    C = 0.5 * log2(1 + SNR_linear)
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return 0.5 * np.log2(1.0 + snr_linear)


def can_decode(source_rate_bpp: float, bandwidth_ratio: float, snr_db: float) -> bool:
    """Check if digital transmission succeeds via Shannon capacity.

    Args:
        source_rate_bpp: Source coding rate in bits per pixel.
        bandwidth_ratio: Channel uses per source dimension (rho).
        snr_db: Channel SNR in dB.

    Returns:
        True if required channel rate is within capacity.
    """
    required_channel_rate = source_rate_bpp / bandwidth_ratio
    capacity = channel_capacity_awgn(snr_db)
    return required_channel_rate <= capacity


def shannon_bound_psnr(snr_db: float, bandwidth_ratio: float) -> float:
    """Optimal PSNR for Gaussian source over AWGN channel.

    D = sigma_x^2 * 2^(-2 * rho * C(SNR)), assuming unit variance source.
    PSNR = 10 * log10(1 / D).

    This is the absolute theoretical limit — no practical system achieves this.
    """
    capacity = channel_capacity_awgn(snr_db)
    exponent = -2.0 * bandwidth_ratio * capacity
    D = 2.0 ** exponent
    if D >= 1.0:
        return 0.0
    if D <= 1e-15:
        return 150.0  # cap at practical limit
    return 10.0 * np.log10(1.0 / D)


def _select_quality(
    sample_image: torch.Tensor,
    codec: Literal["jpeg", "webp"],
    bandwidth_ratio: float,
    target_snr_db: float = 10.0,
) -> int:
    """Auto-select codec quality to match a target transmission rate.

    Picks the quality level whose compressed size best fits within the
    channel capacity at the target SNR for the given bandwidth ratio.
    """
    _, h, w = sample_image.shape
    n_pixels = 3 * h * w  # source dimensions
    k_channel_uses = bandwidth_ratio * n_pixels
    capacity = channel_capacity_awgn(target_snr_db)
    target_bits = k_channel_uses * capacity

    compress_fn = jpeg_compress_decompress if codec == "jpeg" else webp_compress_decompress
    qualities = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    best_q = qualities[0]
    best_diff = float("inf")
    for q in qualities:
        _, bits = compress_fn(sample_image, q)
        diff = abs(bits - target_bits)
        if diff < best_diff:
            best_diff = diff
            best_q = q
        # Also prefer qualities that don't exceed target too much
        if bits <= target_bits * 1.2 and diff < best_diff * 1.5:
            best_q = q

    return best_q


def digital_baseline(
    images: torch.Tensor,
    snr_db: float,
    bandwidth_ratio: float,
    codec: Literal["jpeg", "webp"] = "jpeg",
    quality: int | None = None,
) -> dict:
    """Simulate digital transmission: compress, channel code, transmit, decode.

    Uses Shannon's channel coding theorem as an idealized bound:
    - If the source rate fits within channel capacity -> perfect decoding
    - If not -> complete failure (return gray noise image)

    This is GENEROUS to the digital baseline (real codes perform worse).

    Args:
        images: (B, C, H, W) in [0, 1].
        snr_db: Channel SNR in dB.
        bandwidth_ratio: Channel uses per source dimension (rho).
        codec: "jpeg" or "webp".
        quality: Codec quality. Auto-selected if None.

    Returns:
        Dict with reconstructed images, metrics, and transmission metadata.
    """
    B = images.shape[0]
    compress_fn = jpeg_compress_decompress if codec == "jpeg" else webp_compress_decompress

    if quality is None:
        quality = _select_quality(images[0], codec, bandwidth_ratio)

    capacity = channel_capacity_awgn(snr_db)
    reconstructed = []
    total_bits = 0
    n_success = 0

    for i in range(B):
        recon_img, bits = compress_fn(images[i], quality)
        total_bits += bits

        _, h, w = images[i].shape
        n_pixels = 3 * h * w
        source_rate_bpp = bits / n_pixels
        success = can_decode(source_rate_bpp, bandwidth_ratio, snr_db)

        if success:
            reconstructed.append(recon_img)
            n_success += 1
        else:
            # Decoding failure: return gray noise
            failed = torch.rand_like(images[i]) * 0.3 + 0.35
            reconstructed.append(failed)

    recon_batch = torch.stack(reconstructed).to(images.device)
    avg_bits = total_bits / B
    _, h, w = images[0].shape
    n_pixels = 3 * h * w
    avg_source_rate = avg_bits / n_pixels
    required_channel_rate = avg_source_rate / bandwidth_ratio

    psnr_result = compute_psnr(images, recon_batch)
    ssim_result = compute_ssim(images, recon_batch)

    return {
        "reconstructed": recon_batch,
        "psnr": psnr_result["mean"].item(),
        "psnr_per_sample": psnr_result["per_sample"].tolist(),
        "ssim": ssim_result.item(),
        "success_rate": n_success / B,
        "source_rate_bpp": avg_source_rate,
        "required_channel_rate": required_channel_rate,
        "capacity": capacity,
        "codec": codec,
        "quality": quality,
        "snr_db": snr_db,
        "bandwidth_ratio": bandwidth_ratio,
    }
