"""Training script for Diffusion-JSCC refinement model."""

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.datasets import get_loaders
from src.eval.metrics import compute_psnr, compute_ssim
from src.models.diffusion_jscc.diffusion import GaussianDiffusion
from src.models.diffusion_jscc.model import DiffusionJSCC, load_vae_backbone
from src.models.diffusion_jscc.sampler import ddim_sample
from src.models.diffusion_jscc.unet import ConditionalUNet
from src.models.vae_jscc.model import pad_to_multiple
from src.utils.config import load_config
from src.utils.logging import get_logger, log_metrics
from src.utils.seed import set_seed


def get_device(cfg) -> torch.device:
    """Resolve device from config."""
    if hasattr(cfg, "device") and cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, optimizer, epoch, best_psnr, path):
    """Save a training checkpoint (diffusion UNet only)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.diffusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_psnr": best_psnr,
        },
        path,
    )


@torch.no_grad()
def evaluate(pipeline, dataloader, snr_db, device, num_steps=10):
    """Evaluate diffusion pipeline on a dataset at fixed SNR.

    Args:
        pipeline: DiffusionJSCC model.
        dataloader: Eval dataloader (batch_size=1).
        snr_db: Channel SNR in dB.
        device: Torch device.
        num_steps: DDIM sampling steps for eval.

    Returns:
        Dict with psnr_refined, ssim_refined, psnr_init, ssim_init.
    """
    psnr_ref, ssim_ref = [], []
    psnr_ini, ssim_ini = [], []

    for x in dataloader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)

        x_refined, x_init = pipeline.sample(x, snr_db, num_steps=num_steps)

        # Crop to original size if padded
        h, w = x.shape[2], x.shape[3]
        x_refined = x_refined[:, :, :h, :w].clamp(0, 1)
        x_init = x_init[:, :, :h, :w].clamp(0, 1)

        psnr_ref.append(compute_psnr(x, x_refined)["mean"].item())
        ssim_ref.append(compute_ssim(x, x_refined).item())
        psnr_ini.append(compute_psnr(x, x_init)["mean"].item())
        ssim_ini.append(compute_ssim(x, x_init).item())

    return {
        "psnr_refined": sum(psnr_ref) / len(psnr_ref),
        "ssim_refined": sum(ssim_ref) / len(ssim_ref),
        "psnr_init": sum(psnr_ini) / len(psnr_ini),
        "ssim_init": sum(ssim_ini) / len(ssim_ini),
    }


def train(config_path: str) -> None:
    """Run Diffusion-JSCC training."""
    cfg = load_config(config_path)
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)
    device = get_device(cfg)

    out_dir = Path("outputs/diffusion_jscc")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("train_diffusion", str(out_dir))
    metrics_file = str(out_dir / "train_metrics.jsonl")

    # Load frozen VAE-JSCC backbone
    backbone_cfg = cfg.backbone if hasattr(cfg, "backbone") else None
    ckpt_path = backbone_cfg.checkpoint if backbone_cfg else "outputs/vae_jscc/ckpt_best.pt"
    vae = load_vae_backbone(ckpt_path, device=device)
    logger.info(f"Loaded frozen VAE-JSCC from {ckpt_path}")

    # Build diffusion model
    model_cfg = cfg.model if hasattr(cfg, "model") else None
    T = model_cfg.num_timesteps if model_cfg and hasattr(model_cfg, "num_timesteps") else 1000

    unet = ConditionalUNet(
        in_channels=model_cfg.in_channels if model_cfg and hasattr(model_cfg, "in_channels") else 6,
        out_channels=model_cfg.out_channels if model_cfg and hasattr(model_cfg, "out_channels") else 3,
        base_channels=model_cfg.base_channels if model_cfg and hasattr(model_cfg, "base_channels") else 64,
        channel_mults=tuple(model_cfg.channel_mults) if model_cfg and hasattr(model_cfg, "channel_mults") else (1, 2, 4, 4),
        num_res_blocks=model_cfg.num_res_blocks if model_cfg and hasattr(model_cfg, "num_res_blocks") else 2,
        attention_levels=tuple(model_cfg.attention_levels) if model_cfg and hasattr(model_cfg, "attention_levels") else (2,),
        time_embed_dim=model_cfg.time_embed_dim if model_cfg and hasattr(model_cfg, "time_embed_dim") else 256,
        snr_embed_dim=model_cfg.snr_embed_dim if model_cfg and hasattr(model_cfg, "snr_embed_dim") else 256,
    )

    diffusion = GaussianDiffusion(unet, T=T).to(device)
    pipeline = DiffusionJSCC(vae, diffusion)
    logger.info(f"UNet parameters: {unet.count_parameters():,}")

    # Training config
    train_cfg = cfg.train if hasattr(cfg, "train") else None
    epochs = train_cfg.epochs if train_cfg else 100
    lr = train_cfg.lr if train_cfg else 2e-4
    weight_decay = train_cfg.weight_decay if train_cfg else 0.01
    grad_clip = train_cfg.grad_clip if train_cfg else 1.0
    snr_range = train_cfg.snr_range if train_cfg else [-5, 25]
    eval_every = train_cfg.eval_every if train_cfg else 10
    crop_size = train_cfg.crop_size if train_cfg and hasattr(train_cfg, "crop_size") else 128

    # Override crop size and batch size for diffusion training
    cfg.data.crop_size = crop_size
    if train_cfg and hasattr(train_cfg, "batch_size"):
        cfg.data.batch_size = train_cfg.batch_size

    optimizer = AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    loaders = get_loaders(cfg)
    train_loader = loaders.train
    eval_loader = loaders.test

    logger.info(f"Train batches: {len(train_loader)}, Epochs: {epochs}")
    logger.info(f"LR: {lr}, SNR range: {snr_range}, Crop: {crop_size}")

    best_psnr = 0.0
    eval_snrs = [0, 10, 20]

    for epoch in range(epochs):
        diffusion.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for x in train_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)

            snr_db = torch.empty(1).uniform_(snr_range[0], snr_range[1]).item()

            loss = pipeline.training_step(x, snr_db)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0
        avg_loss = epoch_loss / n_batches

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.6f} "
            f"lr={scheduler.get_last_lr()[0]:.6f} | {dt:.1f}s"
        )
        log_metrics(metrics_file, {
            "epoch": epoch, "loss": avg_loss,
            "lr": scheduler.get_last_lr()[0], "time_s": dt,
        })

        # Evaluation
        if eval_loader is not None and (epoch % eval_every == 0 or epoch == epochs - 1):
            diffusion.eval()
            for snr in eval_snrs:
                metrics = evaluate(pipeline, eval_loader, snr, device, num_steps=10)
                logger.info(
                    f"  SNR={snr:2d}dB | init PSNR={metrics['psnr_init']:.1f} "
                    f"→ refined PSNR={metrics['psnr_refined']:.1f} "
                    f"SSIM={metrics['ssim_refined']:.3f}"
                )
                log_metrics(metrics_file, {
                    "epoch": epoch, "eval_snr_db": snr,
                    "psnr_init": metrics["psnr_init"],
                    "psnr_refined": metrics["psnr_refined"],
                    "ssim_refined": metrics["ssim_refined"],
                })

            # Track best at SNR=10
            mid = evaluate(pipeline, eval_loader, 10.0, device, num_steps=10)
            if mid["psnr_refined"] > best_psnr:
                best_psnr = mid["psnr_refined"]
                save_checkpoint(pipeline, optimizer, epoch, best_psnr,
                                str(out_dir / "ckpt_best.pt"))
                logger.info(f"  ** New best refined PSNR={best_psnr:.1f}dB **")

        # Periodic save
        if epoch % 20 == 0:
            save_checkpoint(pipeline, optimizer, epoch, best_psnr,
                            str(out_dir / f"ckpt_epoch{epoch}.pt"))
        save_checkpoint(pipeline, optimizer, epoch, best_psnr,
                        str(out_dir / "ckpt_last.pt"))

    logger.info(f"Training complete. Best refined PSNR={best_psnr:.1f}dB at SNR=10dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion-JSCC")
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    train(args.config)
