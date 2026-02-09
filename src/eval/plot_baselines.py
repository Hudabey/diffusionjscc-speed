"""Publication-quality cliff effect and baseline comparison plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eval.digital_baselines import channel_capacity_awgn, shannon_bound_psnr

# Consistent project color scheme
COLORS = {
    "jpeg": "#1f77b4",   # blue
    "webp": "#d62728",   # red
    "shannon": "#2ca02c", # green
    "bound": "#333333",  # dark gray
}
MARKERS = {"jpeg": "o", "webp": "s"}


def _setup_style() -> None:
    """Apply clean publication style."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _find_cliff_snr(
    snr_list: list[float], source_rate_bpp: float, bandwidth_ratio: float
) -> float | None:
    """Find the threshold SNR where decoding transitions from fail to success."""
    required = source_rate_bpp / bandwidth_ratio
    for snr in sorted(snr_list):
        if channel_capacity_awgn(snr) >= required:
            return snr
    return None


def plot_cliff_effect(
    results: list[dict],
    bandwidth_ratio: float,
    snr_list: list[float],
    output_dir: str,
) -> None:
    """Generate the main cliff effect plot for a single bandwidth ratio.

    Args:
        results: List of result dicts from digital_baseline().
        bandwidth_ratio: The rho value for this plot.
        snr_list: SNR values evaluated.
        output_dir: Directory to save figures.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Group results by codec
    codecs = {}
    for r in results:
        if abs(r["bandwidth_ratio"] - bandwidth_ratio) > 1e-6:
            continue
        codec = r["codec"]
        if codec not in codecs:
            codecs[codec] = {"snr": [], "psnr": [], "rate": []}
        codecs[codec]["snr"].append(r["snr_db"])
        codecs[codec]["psnr"].append(r["psnr"])
        codecs[codec]["rate"].append(r["source_rate_bpp"])

    # Plot each codec
    for codec, data in codecs.items():
        order = np.argsort(data["snr"])
        snrs = np.array(data["snr"])[order]
        psnrs = np.array(data["psnr"])[order]
        avg_rate = np.mean(data["rate"])

        ax.plot(
            snrs, psnrs,
            color=COLORS[codec], marker=MARKERS[codec], markersize=6,
            linewidth=2, label=f"{codec.upper()} + ideal code ({avg_rate:.2f} bpp)",
        )

        # Vertical dashed line at cliff threshold
        cliff_snr = _find_cliff_snr(snr_list, avg_rate, bandwidth_ratio)
        if cliff_snr is not None:
            ax.axvline(
                cliff_snr, color=COLORS[codec], linestyle=":",
                alpha=0.5, linewidth=1,
            )

    # Shannon bound
    snr_fine = np.linspace(min(snr_list), max(snr_list), 200)
    bound_psnr = [shannon_bound_psnr(s, bandwidth_ratio) for s in snr_fine]
    ax.plot(
        snr_fine, bound_psnr,
        color=COLORS["bound"], linestyle="--", linewidth=2,
        label="Shannon bound (Gaussian source)",
    )

    rho_frac = f"1/{int(1/bandwidth_ratio)}" if bandwidth_ratio < 1 else str(bandwidth_ratio)
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Digital Transmission — Cliff Effect (ρ = {rho_frac})")
    ax.legend(loc="lower right")
    ax.set_xlim(min(snr_list) - 1, max(snr_list) + 1)
    ax.set_ylim(bottom=5)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rho_str = str(bandwidth_ratio).replace(".", "")
    fig.savefig(out / f"cliff_effect_rho_{rho_str}.pdf")
    fig.savefig(out / f"cliff_effect_rho_{rho_str}.png", dpi=150)
    plt.close(fig)


def plot_all_rho(
    results: list[dict],
    bandwidth_ratios: list[float],
    snr_list: list[float],
    output_dir: str,
) -> None:
    """Plot cliff effect across all bandwidth ratios (one line per rho, JPEG only).

    Args:
        results: List of result dicts.
        bandwidth_ratios: All rho values.
        snr_list: SNR values evaluated.
        output_dir: Directory to save figures.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(bandwidth_ratios) - 1, 1)) for i in range(len(bandwidth_ratios))]

    for idx, rho in enumerate(sorted(bandwidth_ratios)):
        data = {"snr": [], "psnr": []}
        for r in results:
            if r["codec"] == "jpeg" and abs(r["bandwidth_ratio"] - rho) < 1e-6:
                data["snr"].append(r["snr_db"])
                data["psnr"].append(r["psnr"])

        if not data["snr"]:
            continue

        order = np.argsort(data["snr"])
        snrs = np.array(data["snr"])[order]
        psnrs = np.array(data["psnr"])[order]

        rho_frac = f"1/{int(1/rho)}" if rho < 1 else str(rho)
        ax.plot(
            snrs, psnrs, color=colors[idx], marker="o", markersize=5,
            linewidth=2, label=f"JPEG, ρ = {rho_frac}",
        )

    # Shannon bounds
    snr_fine = np.linspace(min(snr_list), max(snr_list), 200)
    for idx, rho in enumerate(sorted(bandwidth_ratios)):
        bound = [shannon_bound_psnr(s, rho) for s in snr_fine]
        rho_frac = f"1/{int(1/rho)}" if rho < 1 else str(rho)
        ax.plot(
            snr_fine, bound, color=colors[idx], linestyle="--",
            alpha=0.4, linewidth=1.5,
        )

    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Digital Transmission — Cliff Effect Across Bandwidth Ratios")
    ax.legend(loc="lower right", ncol=2)
    ax.set_xlim(min(snr_list) - 1, max(snr_list) + 1)
    ax.set_ylim(bottom=5)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cliff_effect_all_rho.pdf")
    fig.savefig(out / "cliff_effect_all_rho.png", dpi=150)
    plt.close(fig)


def plot_codec_comparison(
    results: list[dict],
    bandwidth_ratio: float,
    snr_list: list[float],
    output_dir: str,
) -> None:
    """Plot JPEG vs WebP comparison at a single bandwidth ratio.

    Args:
        results: List of result dicts.
        bandwidth_ratio: The rho value for this plot.
        snr_list: SNR values evaluated.
        output_dir: Directory to save figures.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    codecs = {}
    for r in results:
        if abs(r["bandwidth_ratio"] - bandwidth_ratio) > 1e-6:
            continue
        codec = r["codec"]
        if codec not in codecs:
            codecs[codec] = {"snr": [], "psnr": [], "ssim": []}
        codecs[codec]["snr"].append(r["snr_db"])
        codecs[codec]["psnr"].append(r["psnr"])
        codecs[codec]["ssim"].append(r["ssim"])

    # PSNR comparison
    for codec, data in codecs.items():
        order = np.argsort(data["snr"])
        snrs = np.array(data["snr"])[order]
        psnrs = np.array(data["psnr"])[order]
        axes[0].plot(
            snrs, psnrs, color=COLORS[codec], marker=MARKERS[codec],
            markersize=6, linewidth=2, label=f"{codec.upper()}",
        )

    axes[0].set_xlabel("Channel SNR (dB)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR Comparison")
    axes[0].legend()
    axes[0].set_ylim(bottom=5)

    # SSIM comparison
    for codec, data in codecs.items():
        order = np.argsort(data["snr"])
        snrs = np.array(data["snr"])[order]
        ssims = np.array(data["ssim"])[order]
        axes[1].plot(
            snrs, ssims, color=COLORS[codec], marker=MARKERS[codec],
            markersize=6, linewidth=2, label=f"{codec.upper()}",
        )

    axes[1].set_xlabel("Channel SNR (dB)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM Comparison")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    rho_frac = f"1/{int(1/bandwidth_ratio)}" if bandwidth_ratio < 1 else str(bandwidth_ratio)
    fig.suptitle(f"JPEG vs WebP — ρ = {rho_frac}", fontsize=14, y=1.02)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "codec_comparison.pdf")
    fig.savefig(out / "codec_comparison.png", dpi=150)
    plt.close(fig)
