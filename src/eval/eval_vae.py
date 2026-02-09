"""Evaluation script for trained VAE-JSCC model."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eval.digital_baselines import shannon_bound_psnr
from src.eval.metrics import compute_psnr, compute_ssim, compute_lpips
from src.models.vae_jscc.model import VAEJSCC, pad_to_multiple
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


# Consistent color scheme extending the baselines palette
COLORS = {
    "jpeg": "#1f77b4",
    "webp": "#d62728",
    "vae_jscc": "#ff7f0e",  # orange
    "bound": "#333333",
}


def load_model(
    checkpoint_path: str, latent_channels: int, snr_embed_dim: int, device: torch.device
) -> VAEJSCC:
    """Load a trained VAE-JSCC model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        latent_channels: Number of latent channels the model was trained with.
        snr_embed_dim: SNR embedding dimension.
        device: Torch device.

    Returns:
        Loaded VAEJSCC model in eval mode.
    """
    model = VAEJSCC(latent_channels=latent_channels, snr_embed_dim=snr_embed_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_snr(
    model: VAEJSCC,
    dataloader: torch.utils.data.DataLoader,
    snr_db: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate at a single SNR, computing PSNR, SSIM, and LPIPS.

    Args:
        model: Trained VAE-JSCC model in eval mode.
        dataloader: Eval dataloader (batch_size=1 for variable sizes).
        snr_db: Channel SNR in dB.
        device: Torch device.

    Returns:
        Dict with 'psnr', 'ssim', 'lpips' averaged over the dataset.
    """
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for x in dataloader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)

        x_padded = pad_to_multiple(x, 16)
        x_hat, _, _ = model(x_padded, snr_db)

        # Crop back to original size
        x_hat = x_hat[:, :, :x.shape[2], :x.shape[3]]
        x_hat = x_hat.clamp(0, 1)
        all_psnr.append(compute_psnr(x, x_hat)["mean"].item())
        all_ssim.append(compute_ssim(x, x_hat).item())
        all_lpips.append(compute_lpips(x, x_hat).item())

    return {
        "psnr": sum(all_psnr) / len(all_psnr),
        "ssim": sum(all_ssim) / len(all_ssim),
        "lpips": sum(all_lpips) / len(all_lpips),
    }


def plot_vae_vs_baselines(
    vae_results: list[dict],
    baseline_results_path: str,
    bandwidth_ratio: float,
    output_dir: str,
) -> None:
    """Plot VAE-JSCC vs digital baselines and Shannon bound.

    Args:
        vae_results: List of VAE eval result dicts with 'snr_db' and 'psnr'.
        baseline_results_path: Path to M1 baselines results.json.
        bandwidth_ratio: Bandwidth ratio used.
        output_dir: Directory to save figures.
    """
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

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # VAE-JSCC results
    vae_snrs = sorted([r["snr_db"] for r in vae_results])
    vae_psnrs = [
        next(r["psnr"] for r in vae_results if r["snr_db"] == s) for s in vae_snrs
    ]
    ax.plot(
        vae_snrs, vae_psnrs,
        color=COLORS["vae_jscc"], marker="D", markersize=7,
        linewidth=2.5, label="VAE-JSCC (ours)", zorder=5,
    )

    # Load and plot digital baselines
    baseline_path = Path(baseline_results_path)
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)

        for codec in ["jpeg", "webp"]:
            codec_data = [
                r for r in baselines
                if r.get("codec") == codec
                and abs(r["bandwidth_ratio"] - bandwidth_ratio) < 1e-6
            ]
            if not codec_data:
                continue
            snrs = sorted([r["snr_db"] for r in codec_data])
            psnrs = [
                next(r["psnr"] for r in codec_data if r["snr_db"] == s) for s in snrs
            ]
            marker = "o" if codec == "jpeg" else "s"
            ax.plot(
                snrs, psnrs,
                color=COLORS[codec], marker=marker, markersize=5,
                linewidth=1.5, alpha=0.8, label=f"{codec.upper()} + ideal code",
            )

    # Shannon bound
    snr_fine = np.linspace(-5, 25, 200)
    bound_psnr = [shannon_bound_psnr(s, bandwidth_ratio) for s in snr_fine]
    ax.plot(
        snr_fine, bound_psnr,
        color=COLORS["bound"], linestyle="--", linewidth=2,
        label="Shannon bound (Gaussian source)",
    )

    rho_frac = f"1/{int(1/bandwidth_ratio)}" if bandwidth_ratio < 1 else str(bandwidth_ratio)
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"VAE-JSCC vs Digital Baselines (ρ = {rho_frac})")
    ax.legend(loc="lower right")
    ax.set_xlim(-6, 26)
    ax.set_ylim(bottom=5)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "vae_vs_baselines.pdf")
    fig.savefig(out / "vae_vs_baselines.png", dpi=150)
    plt.close(fig)


def main(config_path: str) -> None:
    """Run full VAE-JSCC evaluation sweep.

    Args:
        config_path: Path to YAML config file.
    """
    cfg = load_config(config_path)
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg.model if hasattr(cfg, "model") else None
    latent_channels = model_cfg.latent_channels if model_cfg and hasattr(model_cfg, "latent_channels") else 192
    snr_embed_dim = model_cfg.snr_embed_dim if model_cfg else 256

    # Load best checkpoint
    ckpt_path = "outputs/vae_jscc/ckpt_best.pt"
    if not Path(ckpt_path).exists():
        ckpt_path = "outputs/vae_jscc/ckpt_last.pt"

    logger = get_logger("eval_vae")
    logger.info(f"Loading checkpoint from {ckpt_path}")

    model = load_model(ckpt_path, latent_channels, snr_embed_dim, device)
    logger.info(f"Model loaded: {model.count_parameters():,} params")

    # Eval dataloader
    from src.data.datasets import get_loaders
    loaders = get_loaders(cfg)
    eval_loader = loaders.test

    # SNR sweep
    snr_list = [-5, -2, 0, 2, 5, 8, 10, 12, 15, 18, 20, 25]
    bandwidth_ratio = cfg.jscc.bandwidth_ratio if hasattr(cfg, "jscc") else 0.25

    results = []
    for snr_db in snr_list:
        metrics = evaluate_snr(model, eval_loader, snr_db, device)
        result = {
            "snr_db": snr_db,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
            "bandwidth_ratio": bandwidth_ratio,
            "model": "vae_jscc",
            "latent_channels": latent_channels,
        }
        results.append(result)
        logger.info(
            f"SNR={snr_db:5.1f}dB | PSNR={metrics['psnr']:.1f}dB "
            f"SSIM={metrics['ssim']:.3f} LPIPS={metrics['lpips']:.3f}"
        )

    # Save results
    out_dir = Path("outputs/vae_jscc")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Generate comparison plot
    fig_dir = out_dir / "figures"
    baseline_results_path = "outputs/baselines/results.json"
    plot_vae_vs_baselines(results, baseline_results_path, bandwidth_ratio, str(fig_dir))
    logger.info(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE-JSCC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args, _ = parser.parse_known_args()
    main(args.config)
