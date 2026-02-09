"""Tests for data pipeline."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.data.datasets import get_loaders
from src.data.transforms import get_train_transforms, get_eval_transforms


def _make_cfg(**overrides) -> SimpleNamespace:
    """Create a minimal config for testing."""
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            root="./data",
            train_dataset="cifar10",
            eval_dataset="cifar10",
            batch_size=4,
            num_workers=0,
            crop_size=32,
        )
    )
    for k, v in overrides.items():
        setattr(cfg.data, k, v)
    return cfg


# ── CIFAR-10 tests ────────────────────────────────────────────────────────────

_cifar10_available = (Path("./data") / "cifar-10-batches-py").exists()


@pytest.mark.skipif(not _cifar10_available, reason="CIFAR-10 not downloaded")
def test_cifar10_loads():
    cfg = _make_cfg(train_dataset="cifar10", eval_dataset="cifar10")
    loaders = get_loaders(cfg)
    batch = next(iter(loaders.test))
    images = batch[0]  # CIFAR-10 returns (images, labels)
    assert images.shape == (4, 3, 32, 32)
    assert images.min() >= 0.0
    assert images.max() <= 1.0


# ── Kodak tests ───────────────────────────────────────────────────────────────

_kodak_available = (Path("./data/kodak").exists()
                    and len(list(Path("./data/kodak").glob("*.png"))) == 24)


@pytest.mark.skipif(not _kodak_available, reason="Kodak not downloaded")
def test_kodak_loads():
    cfg = _make_cfg(train_dataset="kodak", eval_dataset="kodak")
    loaders = get_loaders(cfg)
    assert loaders.train is None
    count = 0
    for img in loaders.test:
        assert img.min() >= 0.0
        assert img.max() <= 1.0
        count += 1
    assert count == 24


# ── DIV2K tests ───────────────────────────────────────────────────────────────

_div2k_available = (Path("./data/DIV2K_train_HR").exists()
                    and Path("./data/DIV2K_valid_HR").exists())


@pytest.mark.skipif(not _div2k_available, reason="DIV2K not downloaded")
def test_div2k_loads():
    cfg = _make_cfg(
        train_dataset="div2k", eval_dataset="div2k",
        crop_size=256, batch_size=2,
    )
    loaders = get_loaders(cfg)
    batch = next(iter(loaders.train))
    assert batch.shape[1] == 3
    assert batch.shape[2] == 256
    assert batch.shape[3] == 256
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


# ── Transform tests ───────────────────────────────────────────────────────────


def test_transforms_crop():
    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(128, 64, 32))
    transform = get_train_transforms(crop_size=128)
    out = transform(img)
    assert out.shape == (3, 128, 128)


def test_transforms_range():
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(255, 128, 0))
    transform = get_eval_transforms()
    out = transform(img)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
