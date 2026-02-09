"""Evaluation script for trained Diffusion-JSCC model."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.datasets import get_loaders
from src.eval.digital_baselines import shannon_bound_psnr
from src.eval.metrics import compute_psnr, compute_ssim, compute_lpips
from src.models.diffusion_jscc.diffusion import GaussianDiffusion
from src.models.diffusion_jscc.model import DiffusionJSCC, load_vae_backbone
from src.models.diffusion_jscc.sampler import ddim_sample
from src.models.diffusion_jscc.unet import ConditionalUNet
from src.models.vae_jscc.model import pad_to_multiple
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed


COLORS = {
    "jpeg": "#1f77b4",
    "webp": "#d62728",
    "vae_jscc": "#ff7f0e",
    "diffusion": "#9467bd",
    "bound": "#333333",
}


def load_diffusion_pipeline(cfg, device) -> DiffusionJSCC:
    """Load full Diffusion-JSCC pipeline from checkpoints."""
    backbone_cfg = cfg.backbone if hasattr(cfg, "backbone") else None
    ckpt_path = backbone_cfg.checkpoint if backbone_cfg else "outputs/vae_jscc/ckpt_best.pt"
    vae = load_vae_backbone(ckpt_path, device=device)

    model_cfg = cfg.model if hasattr(cfg, "model") else None
    T = model_cfg.num_timesteps if model_cfg and hasattr(model_cfg, "num_timesteps") else 1000

    unet = ConditionalUNet(
        in_channels=model_cfg.in_channels if model_cfg and hasattr(model_cfg, "in_channels") else 6,
        out_channels=model_cfg.out_channels if model_cfg and hasattr(model_cfg, "out_channels") else 3,
        base_channels=model_cfg.base_channels if model_cfg and hasattr(model_cfg, "base_channels") else 64,
        channel_mults=tuple(model_cfg.channel_mults) if model_cfg and hasattr(model_cfg, "channel_mults") else (1, 2, 4, 4),
        num_res_blocks=model_cfg.num_res_blocks if model_cfg and hasattr(model_cfg, "num_res_blocks") else 2,
        attention_levels=tuple(model_cfg.attention_levels) if model_cfg and hasattr(model_cfg, "attention_levels") else (2,),
    )

    diffusion = GaussianDiffusion(unet, T=T).to(device)

    diff_ckpt_path = "outputs/diffusion_jscc/ckpt_best.pt"
    if not Path(diff_ckpt_path).exists():
        diff_ckpt_path = "outputs/diffusion_jscc/ckpt_last.pt"

    ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=True)
    diffusion.load_state_dict(ckpt["model_state_dict"])
    diffusion.eval()

    return DiffusionJSCC(vae, diffusion)


@torch.no_grad()
def evaluate_snr_steps(pipeline, dataloader, snr_db, num_steps, device):
    """Evaluate at given SNR and step count."""
    psnr_ref, ssim_ref, lpips_ref = [], [], []
    psnr_ini = []
    total_time = 0.0

    for x in dataloader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)

        t0 = time.time()
        x_refined, x_init = pipeline.sample(x, snr_db, num_steps=num_steps)
        total_time += time.time() - t0

        h, w = x.shape[2], x.shape[3]
        x_refined = x_refined[:, :, :h, :w].clamp(0, 1)
        x_init = x_init[:, :, :h, :w].clamp(0, 1)

        psnr_ref.append(compute_psnr(x, x_refined)["mean"].item())
        ssim_ref.append(compute_ssim(x, x_refined).item())
        lpips_ref.append(compute_lpips(x, x_refined).item())
        psnr_ini.append(compute_psnr(x, x_init)["mean"].item())

    n = len(psnr_ref)
    return {
        "psnr_refined": sum(psnr_ref) / n,
        "ssim_refined": sum(ssim_ref) / n,
        "lpips_refined": sum(lpips_ref) / n,
        "psnr_init": sum(psnr_ini) / n,
        "avg_time_s": total_time / n,
    }


def plot_comparison(results, baseline_path, bandwidth_ratio, output_dir):
    """Plot Diffusion-JSCC vs VAE-JSCC vs digital baselines."""
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "legend.fontsize": 9, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Diffusion results (use default steps=50 or highest available)
    diff_50 = [r for r in results if r.get("num_steps") == 50]
    if not diff_50:
        diff_50 = [r for r in results if r.get("num_steps") == max(r2.get("num_steps", 0) for r2 in results)]
    snrs = sorted(set(r["snr_db"] for r in diff_50))
    psnrs = [next(r["psnr_refined"] for r in diff_50 if r["snr_db"] == s) for s in snrs]
    ax.plot(snrs, psnrs, color=COLORS["diffusion"], marker="^", markersize=7,
            linewidth=2.5, label="Diffusion-JSCC (ours)", zorder=5)

    # VAE-JSCC results
    vae_psnrs = [next(r["psnr_init"] for r in diff_50 if r["snr_db"] == s) for s in snrs]
    ax.plot(snrs, vae_psnrs, color=COLORS["vae_jscc"], marker="D", markersize=6,
            linewidth=2, alpha=0.8, label="VAE-JSCC")

    # Digital baselines
    if Path(baseline_path).exists():
        with open(baseline_path) as f:
            baselines = json.load(f)
        for codec in ["jpeg", "webp"]:
            data = [r for r in baselines if r.get("codec") == codec
                    and abs(r["bandwidth_ratio"] - bandwidth_ratio) < 1e-6]
            if data:
                bs = sorted(data, key=lambda r: r["snr_db"])
                ax.plot([r["snr_db"] for r in bs], [r["psnr"] for r in bs],
                        color=COLORS[codec], marker="o" if codec == "jpeg" else "s",
                        markersize=5, linewidth=1.5, alpha=0.7,
                        label=f"{codec.upper()} + ideal code")

    # Shannon bound
    snr_fine = np.linspace(-5, 25, 200)
    ax.plot(snr_fine, [shannon_bound_psnr(s, bandwidth_ratio) for s in snr_fine],
            color=COLORS["bound"], linestyle="--", linewidth=2,
            label="Shannon bound (Gaussian)")

    rho_frac = f"1/{int(1/bandwidth_ratio)}" if bandwidth_ratio < 1 else str(bandwidth_ratio)
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Diffusion-JSCC vs Baselines (ρ = {rho_frac})")
    ax.legend(loc="lower right")
    ax.set_xlim(-6, 26)
    ax.set_ylim(bottom=5)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "diffusion_vs_baselines.pdf")
    fig.savefig(out / "diffusion_vs_baselines.png", dpi=150)
    plt.close(fig)


def plot_steps_vs_quality(results, output_dir):
    """Plot quality vs number of diffusion steps (Pareto frontier)."""
    plt.rcParams.update({
        "font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for snr in sorted(set(r["snr_db"] for r in results)):
        data = sorted([r for r in results if r["snr_db"] == snr], key=lambda r: r["num_steps"])
        steps = [r["num_steps"] for r in data]
        psnrs = [r["psnr_refined"] for r in data]
        times = [r["avg_time_s"] * 1000 for r in data]  # ms

        axes[0].plot(steps, psnrs, marker="o", markersize=5, label=f"SNR={snr}dB")
        axes[1].plot(times, psnrs, marker="o", markersize=5, label=f"SNR={snr}dB")

    axes[0].set_xlabel("Diffusion Steps")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Quality vs Steps")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Inference Time (ms)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Quality vs Latency (Pareto Frontier)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "steps_vs_quality.pdf")
    fig.savefig(out / "steps_vs_quality.png", dpi=150)
    plt.close(fig)


def main(config_path: str) -> None:
    """Run full Diffusion-JSCC evaluation sweep."""
    cfg = load_config(config_path)
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger("eval_diffusion")
    pipeline = load_diffusion_pipeline(cfg, device)
    logger.info("Pipeline loaded")

    loaders = get_loaders(cfg)
    eval_loader = loaders.test

    snr_list = [-5, -2, 0, 2, 5, 8, 10, 12, 15, 18, 20, 25]
    step_list = [1, 2, 5, 10, 25, 50]
    bandwidth_ratio = cfg.jscc.bandwidth_ratio if hasattr(cfg, "jscc") else 0.25

    results = []
    for snr in snr_list:
        for steps in step_list:
            metrics = evaluate_snr_steps(pipeline, eval_loader, snr, steps, device)
            result = {
                "snr_db": snr, "num_steps": steps, "bandwidth_ratio": bandwidth_ratio,
                **metrics,
            }
            results.append(result)
            logger.info(
                f"SNR={snr:5.1f}dB steps={steps:3d} | "
                f"init={metrics['psnr_init']:.1f} → refined={metrics['psnr_refined']:.1f}dB "
                f"SSIM={metrics['ssim_refined']:.3f} | {metrics['avg_time_s']*1000:.0f}ms"
            )

    # Save
    out_dir = Path("outputs/diffusion_jscc")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    fig_dir = str(out_dir / "figures")
    plot_comparison(results, "outputs/baselines/results.json", bandwidth_ratio, fig_dir)
    plot_steps_vs_quality(results, fig_dir)
    logger.info(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diffusion-JSCC")
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    main(args.config)
