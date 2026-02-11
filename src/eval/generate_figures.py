"""Publication-quality figure generation for all JSCC methods.

All figures are saved in both PDF (report) and PNG (README) formats.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eval.digital_baselines import shannon_bound_psnr

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
STYLE = {
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

BANDWIDTH_RATIO = 0.25

# Method styles: color, marker, linestyle, zorder
METHOD_STYLE = {
    "JPEG + Ideal Code":            {"color": "gray",    "marker": "s", "ls": "--", "zorder": 2},
    "WEBP + Ideal Code":            {"color": "gray",    "marker": "D", "ls": ":",  "zorder": 2},
    "Shannon Bound":                {"color": "black",   "marker": "",  "ls": "--", "zorder": 1},
    "DeepJSCC (Ours)":              {"color": "#1f77b4", "marker": "o", "ls": "-",  "zorder": 5},
    "Model-Based (Ours)":           {"color": "#d62728", "marker": "^", "ls": "-",  "zorder": 5},
    "DeepJSCC + Diffusion (Ours)":  {"color": "#2ca02c", "marker": "v", "ls": "-",  "zorder": 4},
}


def _save(fig: plt.Figure, output_dir: str, name: str) -> None:
    """Save figure as PDF and PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def _get_method_data(results: list[dict], method: str, key: str = "psnr"):
    """Extract sorted SNR and metric arrays for a method."""
    data = [r for r in results if r["method"] == method]
    if not data:
        return [], []
    data.sort(key=lambda r: r["snr_db"])
    snrs = [r["snr_db"] for r in data]
    vals = [r[key] for r in data]
    return snrs, vals


# Nicer display names for legends
DISPLAY_NAMES = {
    "WEBP + Ideal Code": "WebP + Ideal Code",
}


def _get_param_label(results: list[dict], method: str) -> str:
    """Get label with parameter count."""
    name = DISPLAY_NAMES.get(method, method)
    data = [r for r in results if r["method"] == method]
    if not data or data[0].get("params", 0) == 0:
        return name
    params = data[0]["params"]
    return f"{name}, {params/1e6:.1f}M"


# ---------------------------------------------------------------------------
# Figure 1: PSNR vs SNR (main figure)
# ---------------------------------------------------------------------------
def plot_psnr_vs_snr(results: list[dict], include_diffusion: bool, output_dir: str) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    methods = ["JPEG + Ideal Code", "WEBP + Ideal Code", "DeepJSCC (Ours)", "Model-Based (Ours)"]
    if include_diffusion:
        methods.append("DeepJSCC + Diffusion (Ours)")

    # Plot Shannon bound as smooth curve
    snr_fine = np.linspace(-5, 25, 200)
    bound = [shannon_bound_psnr(s, BANDWIDTH_RATIO) for s in snr_fine]
    s = METHOD_STYLE["Shannon Bound"]
    ax.plot(snr_fine, bound, color=s["color"], linestyle=s["ls"], linewidth=2,
            label=r"Shannon Bound ($\rho$ = 1/4)", zorder=s["zorder"])

    for method in methods:
        snrs, psnrs = _get_method_data(results, method, "psnr")
        if not snrs:
            continue
        s = METHOD_STYLE[method]
        label = _get_param_label(results, method)
        ax.plot(snrs, psnrs, color=s["color"], marker=s["marker"], linestyle=s["ls"],
                linewidth=2, markersize=7, label=label, zorder=s["zorder"])

    # Annotations
    # Cliff effect region
    ax.annotate(
        "Cliff Effect",
        xy=(8, 14), xytext=(2, 18),
        fontsize=10, fontstyle="italic", color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
    )

    # Graceful degradation
    deep_snrs, deep_psnrs = _get_method_data(results, "DeepJSCC (Ours)", "psnr")
    if deep_psnrs and len(deep_psnrs) > 1:
        ax.annotate(
            "Graceful\nDegradation",
            xy=(deep_snrs[0], deep_psnrs[0]),
            xytext=(deep_snrs[0] - 2, deep_psnrs[0] + 5),
            fontsize=9, fontstyle="italic", color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
        )

    # JSCC advantage shading
    ax.axvspan(-5, 8, alpha=0.06, color="#1f77b4", zorder=0)
    ax.text(-4.5, 38, "JSCC Advantage", fontsize=9, fontstyle="italic",
            color="#1f77b4", alpha=0.7)

    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(r"PSNR vs Channel SNR — All Methods ($\rho$ = 1/4, Kodak)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-6, 26)
    ax.set_ylim(10, 42)

    _save(fig, output_dir, "psnr_vs_snr_all_methods")


# ---------------------------------------------------------------------------
# Figure 2: SSIM vs SNR
# ---------------------------------------------------------------------------
def plot_ssim_vs_snr(results: list[dict], include_diffusion: bool, output_dir: str) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    methods = ["JPEG + Ideal Code", "WEBP + Ideal Code", "DeepJSCC (Ours)", "Model-Based (Ours)"]
    if include_diffusion:
        methods.append("DeepJSCC + Diffusion (Ours)")

    for method in methods:
        snrs, vals = _get_method_data(results, method, "ssim")
        if not snrs:
            continue
        s = METHOD_STYLE[method]
        label = _get_param_label(results, method)
        ax.plot(snrs, vals, color=s["color"], marker=s["marker"], linestyle=s["ls"],
                linewidth=2, markersize=7, label=label, zorder=s["zorder"])

    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("SSIM")
    ax.set_title(r"SSIM vs Channel SNR ($\rho$ = 1/4, Kodak)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-6, 26)
    ax.set_ylim(0, 1.05)

    _save(fig, output_dir, "ssim_vs_snr")


# ---------------------------------------------------------------------------
# Figure 3: LPIPS vs SNR
# ---------------------------------------------------------------------------
def plot_lpips_vs_snr(results: list[dict], include_diffusion: bool, output_dir: str) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    methods = ["DeepJSCC (Ours)", "Model-Based (Ours)"]
    if include_diffusion:
        methods.append("DeepJSCC + Diffusion (Ours)")

    for method in methods:
        snrs, vals = _get_method_data(results, method, "lpips")
        if not snrs:
            continue
        s = METHOD_STYLE[method]
        label = _get_param_label(results, method)
        ax.plot(snrs, vals, color=s["color"], marker=s["marker"], linestyle=s["ls"],
                linewidth=2, markersize=7, label=label, zorder=s["zorder"])

    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("LPIPS (lower is better)")
    ax.set_title(r"LPIPS vs Channel SNR ($\rho$ = 1/4, Kodak)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-6, 26)
    ax.invert_yaxis()

    _save(fig, output_dir, "lpips_vs_snr")


# ---------------------------------------------------------------------------
# Figure 4: Parameter efficiency
# ---------------------------------------------------------------------------
def plot_param_efficiency(
    results: list[dict], include_diffusion: bool, output_dir: str
) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = ["DeepJSCC (Ours)", "Model-Based (Ours)"]
    if include_diffusion:
        methods.append("DeepJSCC + Diffusion (Ours)")

    for method in methods:
        data = [r for r in results if r["method"] == method and r["snr_db"] == 10]
        if not data:
            continue
        params_m = data[0]["params"] / 1e6
        psnr = data[0]["psnr"]
        s = METHOD_STYLE[method]
        ax.scatter(params_m, psnr, color=s["color"], marker=s["marker"],
                   s=200, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(
            f"{method.split(' (')[0]}\n{params_m:.1f}M, {psnr:.1f}dB",
            xy=(params_m, psnr),
            xytext=(15, 10), textcoords="offset points",
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax.set_xlabel("Parameter Count (M)")
    ax.set_ylabel("PSNR at SNR=10dB")
    ax.set_title("Parameter Efficiency (Kodak, SNR=10dB)")
    ax.set_xscale("log")

    _save(fig, output_dir, "param_efficiency")


# ---------------------------------------------------------------------------
# Figure 5: Latency comparison
# ---------------------------------------------------------------------------
def plot_latency_comparison(
    results: list[dict], include_diffusion: bool, output_dir: str
) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["DeepJSCC (Ours)", "Model-Based (Ours)"]
    if include_diffusion:
        methods.append("DeepJSCC + Diffusion (Ours)")

    # Get timing at SNR=10dB
    labels = []
    enc_times = []
    dec_times = []
    colors = []

    for method in methods:
        data = [r for r in results if r["method"] == method and r["snr_db"] == 10]
        if not data:
            continue
        labels.append(method.split(" (")[0])
        enc_times.append(data[0]["enc_time_ms"])
        dec_times.append(data[0]["dec_time_ms"])
        colors.append(METHOD_STYLE[method]["color"])

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, enc_times, width, label="Encoding",
                   color=[c for c in colors], alpha=0.7, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dec_times, width, label="Decoding",
                   color=[c for c in colors], alpha=0.4, edgecolor="black", linewidth=0.5,
                   hatch="//")

    # Add total time labels
    for i, (e, d) in enumerate(zip(enc_times, dec_times)):
        total = e + d
        ax.text(i, max(e, d) + 2, f"Total: {total:.0f}ms",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Method")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Encoding/Decoding Latency (Kodak, SNR=10dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()

    _save(fig, output_dir, "latency_comparison")


# ---------------------------------------------------------------------------
# Figure 6: Cliff effect annotated
# ---------------------------------------------------------------------------
def plot_cliff_effect(results: list[dict], include_diffusion: bool, output_dir: str) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Digital baselines with cliff
    for method in ["JPEG + Ideal Code", "WEBP + Ideal Code"]:
        snrs, psnrs = _get_method_data(results, method, "psnr")
        if not snrs:
            continue
        s = METHOD_STYLE[method]
        label = DISPLAY_NAMES.get(method, method)
        ax.plot(snrs, psnrs, color=s["color"], marker=s["marker"], linestyle=s["ls"],
                linewidth=2.5, markersize=8, label=label, zorder=s["zorder"])

    # Learned methods (thicker lines)
    for method in ["DeepJSCC (Ours)", "Model-Based (Ours)"]:
        snrs, psnrs = _get_method_data(results, method, "psnr")
        if not snrs:
            continue
        s = METHOD_STYLE[method]
        label = _get_param_label(results, method)
        ax.plot(snrs, psnrs, color=s["color"], marker=s["marker"], linestyle=s["ls"],
                linewidth=3, markersize=8, label=label, zorder=s["zorder"])

    if include_diffusion:
        snrs, psnrs = _get_method_data(results, "DeepJSCC + Diffusion (Ours)", "psnr")
        if snrs:
            s = METHOD_STYLE["DeepJSCC + Diffusion (Ours)"]
            ax.plot(snrs, psnrs, color=s["color"], marker=s["marker"], linestyle=s["ls"],
                    linewidth=3, markersize=8, label="DeepJSCC + Diffusion (Ours)",
                    zorder=s["zorder"])

    # Annotate cliff
    ax.annotate(
        "CLIFF EFFECT\nDigital fails below\nchannel capacity",
        xy=(7, 13), xytext=(1, 22),
        fontsize=11, fontweight="bold", color="#d62728",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.9),
        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2),
    )

    # Annotate graceful degradation
    ax.annotate(
        "GRACEFUL DEGRADATION\nLearned JSCC still works",
        xy=(-3, 24), xytext=(8, 38),
        fontsize=11, fontweight="bold", color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9),
        arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2),
    )

    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("The Cliff Effect: Digital vs Learned JSCC (Kodak)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-6, 26)
    ax.set_ylim(8, 42)

    _save(fig, output_dir, "cliff_effect_annotated")


# ---------------------------------------------------------------------------
# Figure 7: Model-Based convergence
# ---------------------------------------------------------------------------
def plot_convergence(convergence: dict[int, float], output_dir: str) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))

    ks = sorted(convergence.keys())
    psnrs = [convergence[k] for k in ks]

    ax.plot(ks, psnrs, color="#d62728", marker="o", linewidth=2.5, markersize=9, zorder=5)

    # Fill area under curve
    ax.fill_between(ks, psnrs, alpha=0.1, color="#d62728")

    # Annotate each point
    for k, p in zip(ks, psnrs):
        ax.annotate(f"{p:.1f}", xy=(k, p), xytext=(0, 10),
                    textcoords="offset points", fontsize=9, ha="center")

    ax.set_xlabel("Number of Unrolled Iterations (K)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Model-Based JSCC: Iteration Convergence (SNR=10dB, Kodak)")
    ax.set_xticks(ks)
    ax.set_xlim(0.5, max(ks) + 0.5)

    _save(fig, output_dir, "model_based_convergence")


# ---------------------------------------------------------------------------
# Figure 8: Visual comparison
# ---------------------------------------------------------------------------
def plot_visual_comparison(visual_samples: dict, output_dir: str) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        """(1, C, H, W) or (C, H, W) tensor -> (H, W, C) numpy for display."""
        if t.ndim == 4:
            t = t[0]
        return t.permute(1, 2, 0).clamp(0, 1).numpy()

    original = _to_numpy(visual_samples["original"])
    image_name = visual_samples.get("image_name", "kodim01")

    for row, snr in enumerate([0, 10]):
        # Column 0: Original
        axes[row, 0].imshow(original)
        axes[row, 0].set_title("Original", fontsize=12, fontweight="bold")
        axes[row, 0].axis("off")

        # Column 1: JPEG
        jpeg_key = f"jpeg_snr{snr}"
        if jpeg_key in visual_samples:
            jpeg_img = _to_numpy(visual_samples[jpeg_key])
            jpeg_psnr = visual_samples.get(f"{jpeg_key}_psnr", 0)
            axes[row, 1].imshow(jpeg_img)
            axes[row, 1].set_title(f"JPEG ({jpeg_psnr:.1f} dB)", fontsize=12)
        axes[row, 1].axis("off")

        # Column 2: DeepJSCC
        djscc_key = f"deepjscc_snr{snr}"
        if djscc_key in visual_samples:
            djscc_img = _to_numpy(visual_samples[djscc_key])
            djscc_psnr = visual_samples.get(f"{djscc_key}_psnr", 0)
            axes[row, 2].imshow(djscc_img)
            axes[row, 2].set_title(f"DeepJSCC ({djscc_psnr:.1f} dB)", fontsize=12)
        axes[row, 2].axis("off")

        # Column 3: Model-Based
        mb_key = f"model_based_snr{snr}"
        if mb_key in visual_samples:
            mb_img = _to_numpy(visual_samples[mb_key])
            mb_psnr = visual_samples.get(f"{mb_key}_psnr", 0)
            axes[row, 3].imshow(mb_img)
            axes[row, 3].set_title(f"Model-Based ({mb_psnr:.1f} dB)", fontsize=12)
        axes[row, 3].axis("off")

        # Row label
        axes[row, 0].text(
            -0.15, 0.5, f"SNR = {snr} dB",
            transform=axes[row, 0].transAxes,
            fontsize=14, fontweight="bold", rotation=90,
            va="center", ha="center",
        )

    fig.suptitle(
        f"Visual Comparison on {image_name} at SNR=0dB (top) and SNR=10dB (bottom)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])

    _save(fig, output_dir, "visual_comparison")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_all_figures(
    all_results: list[dict],
    convergence: dict[int, float],
    visual_samples: dict,
    include_diffusion: bool,
    diffusion_results: list[dict],
    output_dir: str,
) -> None:
    """Generate all 8 publication-quality figures."""

    print(f"Generating Figure 1: PSNR vs SNR (main figure)...")
    plot_psnr_vs_snr(all_results, include_diffusion, output_dir)

    print(f"Generating Figure 2: SSIM vs SNR...")
    plot_ssim_vs_snr(all_results, include_diffusion, output_dir)

    print(f"Generating Figure 3: LPIPS vs SNR...")
    plot_lpips_vs_snr(all_results, include_diffusion, output_dir)

    print(f"Generating Figure 4: Parameter efficiency...")
    plot_param_efficiency(all_results, include_diffusion, output_dir)

    print(f"Generating Figure 5: Latency comparison...")
    plot_latency_comparison(all_results, include_diffusion, output_dir)

    print(f"Generating Figure 6: Cliff effect annotated...")
    plot_cliff_effect(all_results, include_diffusion, output_dir)

    print(f"Generating Figure 7: Model-Based convergence...")
    plot_convergence(convergence, output_dir)

    print(f"Generating Figure 8: Visual comparison...")
    plot_visual_comparison(visual_samples, output_dir)

    print(f"All 8 figures saved to {output_dir}")
