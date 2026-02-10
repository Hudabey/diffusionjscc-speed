"""Evaluation script for trained Model-Based JSCC model."""

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
from src.models.model_based.model import ModelBasedJSCC, pad_to_multiple
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


COLORS = {
    "jpeg": "#1f77b4",
    "webp": "#d62728",
    "vae_jscc": "#ff7f0e",
    "model_based": "#2ca02c",  # green
    "bound": "#333333",
}


def load_model(
    checkpoint_path: str,
    latent_channels: int,
    num_iterations: int,
    denoiser_channels: int,
    snr_embed_dim: int,
    base_channels: int,
    device: torch.device,
) -> ModelBasedJSCC:
    """Load a trained Model-Based JSCC model from checkpoint."""
    model = ModelBasedJSCC(
        latent_channels=latent_channels,
        num_iterations=num_iterations,
        denoiser_channels=denoiser_channels,
        snr_embed_dim=snr_embed_dim,
        base_channels=base_channels,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_snr(
    model: ModelBasedJSCC,
    dataloader: torch.utils.data.DataLoader,
    snr_db: float,
    device: torch.device,
    num_iterations: int | None = None,
) -> dict[str, float]:
    """Evaluate at a single SNR, computing PSNR, SSIM, and LPIPS."""
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for x in dataloader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)

        x_padded = pad_to_multiple(x, 8)
        if num_iterations is not None:
            x_hat = model.run_k_iterations(x_padded, snr_db, k=num_iterations)
        else:
            x_hat = model(x_padded, snr_db)

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


def plot_model_based_vs_all(
    mb_results: list[dict],
    vae_results_path: str,
    baseline_results_path: str,
    bandwidth_ratio: float,
    output_dir: str,
) -> None:
    """Plot Model-Based JSCC vs VAE-JSCC, digital baselines, and Shannon bound."""
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

    # Model-Based JSCC results
    mb_snrs = sorted([r["snr_db"] for r in mb_results])
    mb_psnrs = [
        next(r["psnr"] for r in mb_results if r["snr_db"] == s) for s in mb_snrs
    ]
    ax.plot(
        mb_snrs, mb_psnrs,
        color=COLORS["model_based"], marker="^", markersize=7,
        linewidth=2.5, label="Model-Based JSCC (ours)", zorder=5,
    )

    # VAE-JSCC results
    vae_path = Path(vae_results_path)
    if vae_path.exists():
        with open(vae_path) as f:
            vae_data = json.load(f)
        vae_snrs = sorted([r["snr_db"] for r in vae_data])
        vae_psnrs = [
            next(r["psnr"] for r in vae_data if r["snr_db"] == s) for s in vae_snrs
        ]
        ax.plot(
            vae_snrs, vae_psnrs,
            color=COLORS["vae_jscc"], marker="D", markersize=5,
            linewidth=1.5, alpha=0.8, label="VAE-JSCC",
        )

    # Digital baselines
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
    ax.set_title(f"Model-Based JSCC vs All Methods ({chr(961)} = {rho_frac})")
    ax.legend(loc="lower right")
    ax.set_xlim(-6, 26)
    ax.set_ylim(bottom=5)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "model_based_vs_all.pdf")
    fig.savefig(out / "model_based_vs_all.png", dpi=150)
    plt.close(fig)


def plot_iteration_convergence(
    iter_results: dict[int, list[dict]],
    output_dir: str,
) -> None:
    """Plot PSNR vs SNR for different iteration counts."""
    plt.rcParams.update({
        "font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis
    ks = sorted(iter_results.keys())

    for idx, k in enumerate(ks):
        color = cmap(idx / max(len(ks) - 1, 1))
        results = iter_results[k]
        snrs = sorted([r["snr_db"] for r in results])
        psnrs = [next(r["psnr"] for r in results if r["snr_db"] == s) for s in snrs]
        ax.plot(snrs, psnrs, marker="o", markersize=5, linewidth=1.5,
                color=color, label=f"K={k}")

    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Iteration Convergence")
    ax.legend()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "iteration_convergence.pdf")
    fig.savefig(out / "iteration_convergence.png", dpi=150)
    plt.close(fig)


def plot_learned_step_sizes(model: ModelBasedJSCC, output_dir: str) -> None:
    """Plot the learned step sizes (alphas) per iteration."""
    plt.rcParams.update({
        "font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    step_sizes = model.get_step_sizes()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(step_sizes) + 1), step_sizes, color="#2ca02c", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step Size (alpha)")
    ax.set_title("Learned Step Sizes")
    ax.set_xticks(range(1, len(step_sizes) + 1))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "learned_step_sizes.pdf")
    fig.savefig(out / "learned_step_sizes.png", dpi=150)
    plt.close(fig)


def main(config_path: str) -> None:
    """Run full Model-Based JSCC evaluation sweep."""
    cfg = load_config(config_path)
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg.model if hasattr(cfg, "model") else None
    latent_channels = getattr(model_cfg, "latent_channels", 12)
    num_iterations = getattr(model_cfg, "num_iterations", 6)
    denoiser_channels = getattr(model_cfg, "denoiser_channels", 64)
    snr_embed_dim = getattr(model_cfg, "snr_embed_dim", 128)
    base_channels = getattr(model_cfg, "base_channels", 64)

    # Load best checkpoint
    ckpt_path = "outputs/model_based/ckpt_best.pt"
    if not Path(ckpt_path).exists():
        ckpt_path = "outputs/model_based/ckpt_last.pt"

    logger = get_logger("eval_model_based")
    logger.info(f"Loading checkpoint from {ckpt_path}")

    model = load_model(
        ckpt_path, latent_channels, num_iterations, denoiser_channels,
        snr_embed_dim, base_channels, device,
    )
    logger.info(f"Model loaded: {model.count_parameters():,} params")
    logger.info(f"Learned step sizes: {model.get_step_sizes()}")

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
            "model": "model_based",
            "num_iterations": num_iterations,
        }
        results.append(result)
        logger.info(
            f"SNR={snr_db:5.1f}dB | PSNR={metrics['psnr']:.1f}dB "
            f"SSIM={metrics['ssim']:.3f} LPIPS={metrics['lpips']:.3f}"
        )

    # Save results
    out_dir = Path("outputs/model_based")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Iteration sweep
    iter_counts = [1, 2, 4, 6]
    iter_counts = [k for k in iter_counts if k <= num_iterations]
    iter_results = {}
    for k in iter_counts:
        k_results = []
        for snr_db in snr_list:
            metrics = evaluate_snr(model, eval_loader, snr_db, device, num_iterations=k)
            k_results.append({"snr_db": snr_db, "psnr": metrics["psnr"]})
            logger.info(f"  K={k} SNR={snr_db:5.1f}dB | PSNR={metrics['psnr']:.1f}dB")
        iter_results[k] = k_results

    # Generate plots
    fig_dir = out_dir / "figures"
    plot_model_based_vs_all(
        results,
        "outputs/vae_jscc/eval_results.json",
        "outputs/baselines/results.json",
        bandwidth_ratio,
        str(fig_dir),
    )
    plot_iteration_convergence(iter_results, str(fig_dir))
    plot_learned_step_sizes(model, str(fig_dir))
    logger.info(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model-Based JSCC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args, _ = parser.parse_known_args()
    main(args.config)
