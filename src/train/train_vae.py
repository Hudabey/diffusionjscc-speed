"""Training script for DeepJSCC model (v2: no VAE, MSE+MS-SSIM loss)."""

import argparse
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.datasets import get_loaders
from src.eval.metrics import compute_psnr, compute_ssim
from src.models.vae_jscc.model import VAEJSCC, pad_to_multiple
from src.utils.config import load_config
from src.utils.logging import get_logger, log_metrics
from src.utils.seed import set_seed


def get_device(cfg) -> torch.device:
    """Resolve device from config."""
    if hasattr(cfg, "device") and cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: VAEJSCC,
    optimizer: Adam,
    epoch: int,
    best_psnr: float,
    path: str,
) -> None:
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_psnr": best_psnr,
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model: VAEJSCC,
    dataloader: torch.utils.data.DataLoader,
    snr_db: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a dataset at a fixed SNR."""
    model.eval()
    all_psnr = []
    all_ssim = []

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

    model.train()
    return {
        "psnr": sum(all_psnr) / len(all_psnr),
        "ssim": sum(all_ssim) / len(all_ssim),
    }


def train(config_path: str) -> None:
    """Run DeepJSCC training.

    Args:
        config_path: Path to YAML config file.
    """
    cfg = load_config(config_path)

    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)
    device = get_device(cfg)

    # Output directory — v2 to keep old results
    out_dir = Path("outputs/vae_jscc_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_vae", str(out_dir))
    metrics_file = str(out_dir / "train_metrics.jsonl")

    # Model config
    model_cfg = cfg.model if hasattr(cfg, "model") else None
    latent_channels = getattr(model_cfg, "latent_channels", 192)
    base_channels = getattr(model_cfg, "base_channels", 64)

    model = VAEJSCC(
        latent_channels=latent_channels,
        base_channels=base_channels,
    ).to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Latent channels: {latent_channels}, Base channels: {base_channels}, Device: {device}")

    # Training config
    train_cfg = cfg.train if hasattr(cfg, "train") else None
    epochs = getattr(train_cfg, "epochs", 500)
    lr = getattr(train_cfg, "lr", 1e-4)
    weight_decay = getattr(train_cfg, "weight_decay", 1e-5)
    grad_clip = getattr(train_cfg, "grad_clip", 1.0)
    snr_range = getattr(train_cfg, "snr_range", [-2, 20])
    warmup_epochs = getattr(train_cfg, "warmup_epochs", 10)
    eval_every = getattr(train_cfg, "eval_every", 25)
    save_every = getattr(train_cfg, "save_every", 50)

    # Override data batch_size from train config if present
    if train_cfg and hasattr(train_cfg, "batch_size"):
        cfg.data.batch_size = train_cfg.batch_size

    # Override crop_size from data config
    if hasattr(cfg.data, "crop_size"):
        cfg.data.crop_size = cfg.data.crop_size

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Checkpoint resume
    start_epoch = 0
    best_psnr = 0.0
    ckpt_last = out_dir / "ckpt_last.pt"
    if ckpt_last.exists():
        logger.info(f"Resuming from {ckpt_last}")
        ckpt = torch.load(str(ckpt_last), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt["best_psnr"]
        for _ in range(start_epoch):
            scheduler.step()
        logger.info(f"Resumed at epoch {start_epoch}, best PSNR={best_psnr:.1f}dB")

    # Data loaders
    loaders = get_loaders(cfg)
    train_loader = loaders.train
    eval_loader = loaders.test

    if train_loader is None:
        raise RuntimeError("No training data. Check config data.train_dataset.")

    logger.info(f"Train batches: {len(train_loader)}, Eval loader: {eval_loader is not None}")
    logger.info(f"Epochs: {epochs}, LR: {lr}, SNR range: {snr_range}")

    best_psnr = max(best_psnr, 0.0)
    eval_snrs = [0, 5, 10, 15, 20]

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_msssim = 0.0
        n_batches = 0

        # Linear LR warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        t0 = time.time()
        for x in train_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)

            # Random SNR for this batch
            snr_db = torch.empty(1).uniform_(snr_range[0], snr_range[1]).item()

            x_hat, z, _ = model(x, snr_db)

            loss, mse_loss, msssim_loss = VAEJSCC.compute_loss(x, x_hat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_msssim += msssim_loss.item()
            n_batches += 1

        # Only step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        dt = time.time() - t0

        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse / n_batches
        avg_msssim = epoch_msssim / n_batches
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} "
            f"mse={avg_mse:.4f} msssim={avg_msssim:.4f} "
            f"lr={current_lr:.6f} | {dt:.1f}s"
        )

        log_metrics(metrics_file, {
            "epoch": epoch,
            "loss": avg_loss,
            "mse_loss": avg_mse,
            "msssim_loss": avg_msssim,
            "lr": current_lr,
            "time_s": dt,
        })

        # Validation
        if eval_loader is not None and (epoch % eval_every == 0 or epoch == epochs - 1):
            for eval_snr in eval_snrs:
                metrics = evaluate(model, eval_loader, eval_snr, device)
                logger.info(
                    f"  eval SNR={eval_snr:2d}dB | "
                    f"PSNR={metrics['psnr']:.1f}dB SSIM={metrics['ssim']:.3f}"
                )
                log_metrics(metrics_file, {
                    "epoch": epoch,
                    "eval_snr_db": eval_snr,
                    "eval_psnr": metrics["psnr"],
                    "eval_ssim": metrics["ssim"],
                })

            # Track best at SNR=10dB
            mid_snr_metrics = evaluate(model, eval_loader, 10.0, device)
            if mid_snr_metrics["psnr"] > best_psnr:
                best_psnr = mid_snr_metrics["psnr"]
                save_checkpoint(model, optimizer, epoch, best_psnr,
                                str(out_dir / "ckpt_best.pt"))
                logger.info(f"  ** New best PSNR={best_psnr:.1f}dB at SNR=10dB **")

        # Periodic save
        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, best_psnr,
                            str(out_dir / f"ckpt_epoch{epoch}.pt"))

        # Always save latest
        save_checkpoint(model, optimizer, epoch, best_psnr,
                        str(out_dir / "ckpt_last.pt"))

    logger.info(f"Training complete. Best PSNR={best_psnr:.1f}dB at SNR=10dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepJSCC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args, _ = parser.parse_known_args()
    train(args.config)
