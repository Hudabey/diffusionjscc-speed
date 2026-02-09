"""Training script for VAE-JSCC model."""

import argparse
import json
import math
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


def get_beta(epoch: int, beta_start: float, beta_end: float, warmup_epochs: int) -> float:
    """Linearly warm up beta from beta_start to beta_end over warmup_epochs."""
    if epoch >= warmup_epochs:
        return beta_end
    return beta_start + (beta_end - beta_start) * epoch / warmup_epochs


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
    """Evaluate model on a dataset at a fixed SNR.

    Args:
        model: VAE-JSCC model.
        dataloader: Evaluation dataloader (batch_size=1 for variable sizes).
        snr_db: Channel SNR in dB.
        device: Torch device.

    Returns:
        Dict with 'psnr' and 'ssim' averaged over the dataset.
    """
    model.eval()
    all_psnr = []
    all_ssim = []

    for x in dataloader:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)

        # Pad to multiple of 16 for the encoder's strided convolutions
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
    """Run VAE-JSCC training.

    Args:
        config_path: Path to YAML config file.
    """
    cfg = load_config(config_path)

    # Seed for reproducibility
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    device = get_device(cfg)

    # Output directory
    out_dir = Path("outputs/vae_jscc")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("train_vae", str(out_dir))
    metrics_file = str(out_dir / "train_metrics.jsonl")

    # Model config
    model_cfg = cfg.model if hasattr(cfg, "model") else None
    latent_channels = model_cfg.latent_channels if model_cfg and hasattr(model_cfg, "latent_channels") else 192
    snr_embed_dim = model_cfg.snr_embed_dim if model_cfg else 256

    model = VAEJSCC(
        latent_channels=latent_channels,
        snr_embed_dim=snr_embed_dim,
    ).to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Latent channels: {latent_channels}, Device: {device}")

    # Training config
    train_cfg = cfg.train if hasattr(cfg, "train") else None
    epochs = train_cfg.epochs if train_cfg else 200
    lr = train_cfg.lr if train_cfg else 1e-4
    weight_decay = train_cfg.weight_decay if train_cfg else 1e-5
    grad_clip = train_cfg.grad_clip if train_cfg else 1.0
    snr_range = train_cfg.snr_range if train_cfg else [0, 20]
    eval_every = train_cfg.eval_every if train_cfg else 10
    save_every = train_cfg.save_every if train_cfg else 20

    beta_start = model_cfg.beta_start if model_cfg and hasattr(model_cfg, "beta_start") else 0.0001
    beta_end = model_cfg.beta_end if model_cfg and hasattr(model_cfg, "beta_end") else 0.001
    beta_warmup = model_cfg.beta_warmup_epochs if model_cfg and hasattr(model_cfg, "beta_warmup_epochs") else 50

    # Override data batch_size from train config if present
    if train_cfg and hasattr(train_cfg, "batch_size"):
        cfg.data.batch_size = train_cfg.batch_size

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Data loaders
    loaders = get_loaders(cfg)
    train_loader = loaders.train
    eval_loader = loaders.test

    if train_loader is None:
        raise RuntimeError("No training data. Check config data.train_dataset.")

    logger.info(f"Train batches: {len(train_loader)}, Eval loader: {eval_loader is not None}")
    logger.info(f"Epochs: {epochs}, LR: {lr}, SNR range: {snr_range}")
    logger.info(f"Beta: {beta_start} → {beta_end} over {beta_warmup} epochs")

    best_psnr = 0.0
    eval_snrs = [0, 5, 10, 15, 20]

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0
        beta = get_beta(epoch, beta_start, beta_end, beta_warmup)

        t0 = time.time()
        for x in train_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)

            # Random SNR for this batch
            snr_db = torch.empty(1).uniform_(snr_range[0], snr_range[1]).item()

            x_hat, mu, log_sigma = model(x, snr_db)

            loss, recon_loss, kl_loss = VAEJSCC.compute_loss(
                x, x_hat, mu, log_sigma, beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} "
            f"recon={avg_recon:.4f} kl={avg_kl:.2f} beta={beta:.5f} "
            f"lr={scheduler.get_last_lr()[0]:.6f} | {dt:.1f}s"
        )

        log_metrics(metrics_file, {
            "epoch": epoch,
            "loss": avg_loss,
            "recon_loss": avg_recon,
            "kl_loss": avg_kl,
            "beta": beta,
            "lr": scheduler.get_last_lr()[0],
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
    parser = argparse.ArgumentParser(description="Train VAE-JSCC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args, _ = parser.parse_known_args()
    train(args.config)
