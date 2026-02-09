"""Run digital baseline evaluation across SNR ranges and bandwidth ratios."""

import argparse
import json
from pathlib import Path

import torch

from src.eval.digital_baselines import digital_baseline, shannon_bound_psnr
from src.eval.plot_baselines import (
    plot_cliff_effect,
    plot_all_rho,
    plot_codec_comparison,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


SNR_LIST = [-5, -2, 0, 2, 5, 8, 10, 12, 15, 18, 20, 25]
BANDWIDTH_RATIOS = [1 / 16, 1 / 8, 1 / 4, 1 / 2]
CODECS = ["jpeg", "webp"]


def _load_eval_images(cfg) -> list[torch.Tensor]:
    """Load evaluation images, preferring Kodak, falling back to CIFAR-10.

    Returns a list of (C, H, W) tensors (may have different spatial sizes).
    """
    kodak_dir = Path(cfg.data.root) / "kodak"
    if kodak_dir.exists() and len(list(kodak_dir.glob("*.png"))) == 24:
        from src.data.datasets import ImageFolderDataset
        from src.data.transforms import get_eval_transforms

        ds = ImageFolderDataset(str(kodak_dir), transform=get_eval_transforms())
        images = [ds[i] for i in range(len(ds))]
        print(f"Loaded Kodak dataset: {len(images)} images, {images[0].shape}")
        return images

    # Fall back to CIFAR-10
    from torchvision import datasets as tv_datasets
    from src.data.transforms import get_eval_transforms

    cifar_root = Path(cfg.data.root)
    if not (cifar_root / "cifar-10-batches-py").exists():
        raise FileNotFoundError("Neither Kodak nor CIFAR-10 found. Run scripts/setup_pod.sh")

    ds = tv_datasets.CIFAR10(
        root=str(cifar_root), train=False, download=False,
        transform=get_eval_transforms(),
    )
    images = [ds[i][0] for i in range(24)]
    print(f"Loaded CIFAR-10 (first 24): {images[0].shape}")
    return images


def main() -> None:
    """Run all digital baselines and generate figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config, overrides=unknown)
    set_seed(cfg.seed)

    image_list = _load_eval_images(cfg)
    print(f"Evaluating {len(image_list)} images, first: {image_list[0].shape}")

    output_dir = Path("outputs/baselines")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for rho in BANDWIDTH_RATIOS:
        rho_frac = f"1/{int(1/rho)}" if rho < 1 else str(rho)
        print(f"\n{'='*60}")
        print(f"Bandwidth ratio ρ = {rho_frac} ({rho:.4f})")
        print(f"{'='*60}")

        for codec in CODECS:
            print(f"\n  Codec: {codec.upper()}")

            for snr_db in SNR_LIST:
                set_seed(cfg.seed)  # deterministic per evaluation
                # Process each image individually (varying spatial sizes)
                per_img_results = []
                for img in image_list:
                    r = digital_baseline(
                        img.unsqueeze(0), snr_db=snr_db,
                        bandwidth_ratio=rho, codec=codec,
                    )
                    per_img_results.append(r)

                # Aggregate across images
                avg_psnr = sum(r["psnr"] for r in per_img_results) / len(per_img_results)
                avg_ssim = sum(r["ssim"] for r in per_img_results) / len(per_img_results)
                avg_rate = sum(r["source_rate_bpp"] for r in per_img_results) / len(per_img_results)
                avg_success = sum(r["success_rate"] for r in per_img_results) / len(per_img_results)
                capacity = per_img_results[0]["capacity"]
                quality = per_img_results[0]["quality"]
                req_rate = sum(r["required_channel_rate"] for r in per_img_results) / len(per_img_results)

                result_save = {
                    "psnr": avg_psnr,
                    "ssim": avg_ssim,
                    "success_rate": avg_success,
                    "source_rate_bpp": avg_rate,
                    "required_channel_rate": req_rate,
                    "capacity": capacity,
                    "codec": codec,
                    "quality": quality,
                    "snr_db": snr_db,
                    "bandwidth_ratio": rho,
                }
                all_results.append(result_save)

                status = "OK" if avg_success > 0.5 else "FAIL"
                print(
                    f"    SNR={snr_db:+5.0f}dB | PSNR={avg_psnr:5.1f}dB | "
                    f"SSIM={avg_ssim:.3f} | rate={avg_rate:.3f}bpp | "
                    f"C={capacity:.2f}bpc | {status}"
                )

        # Shannon bound for reference
        for snr_db in SNR_LIST:
            bound = shannon_bound_psnr(snr_db, rho)
            all_results.append({
                "codec": "shannon_bound",
                "snr_db": snr_db,
                "bandwidth_ratio": rho,
                "psnr": bound,
            })

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate figures
    print("\nGenerating figures...")
    codec_results = [r for r in all_results if r["codec"] != "shannon_bound"]

    # Main cliff effect at rho=1/4
    plot_cliff_effect(codec_results, 0.25, SNR_LIST, str(fig_dir))

    # All bandwidth ratios
    plot_all_rho(codec_results, BANDWIDTH_RATIOS, SNR_LIST, str(fig_dir))

    # Codec comparison at rho=1/4
    plot_codec_comparison(codec_results, 0.25, SNR_LIST, str(fig_dir))

    print(f"Figures saved to {fig_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
