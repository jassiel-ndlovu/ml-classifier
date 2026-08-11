"""Dataset loading, stratified splitting and augmentation.

The processed dataset is a single ``.npz`` produced by
``scripts/reconstruct_dataset.py``. Images are stored as uint8 {0, 1} in their
canonical (orientation-corrected) upright pose. This module wraps them in a
PyTorch ``Dataset``, scales pixels to the {0, 255} range the model expects, and
provides light, label-preserving augmentation for the tiny training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import PIXEL_SCALE, Config


@dataclass
class Splits:
    train: GlyphDataset
    val: GlyphDataset
    test: GlyphDataset


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, idx) from the processed dataset file."""
    data = np.load(path)
    return data["X"].astype(np.uint8), data["y"].astype(np.int64), data["idx"].astype(np.int64)


def stratified_split(
    y: np.ndarray, val_fraction: float, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class index split so every class appears in every partition.

    With as few as four samples in a class this keeps at least one example in
    train and distributes the rest, which a naive random split cannot guarantee.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_fraction))) if n - n_test >= 2 else 0
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test : n_test + n_val])
        train_idx.extend(idx[n_test + n_val :])
    return (
        np.array(sorted(train_idx)),
        np.array(sorted(val_idx)),
        np.array(sorted(test_idx)),
    )


def _augment(img: np.ndarray, translate_px: int, noise_flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Random small translation + salt-and-pepper noise on a {0,1} image."""
    out = img
    if translate_px > 0:
        dx = int(rng.integers(-translate_px, translate_px + 1))
        dy = int(rng.integers(-translate_px, translate_px + 1))
        out = np.roll(out, shift=(dy, dx), axis=(0, 1))
        if dy > 0:
            out[:dy, :] = 0
        elif dy < 0:
            out[dy:, :] = 0
        if dx > 0:
            out[:, :dx] = 0
        elif dx < 0:
            out[:, dx:] = 0
    if noise_flip_prob > 0:
        flip = rng.random(out.shape) < noise_flip_prob
        out = np.where(flip, 1 - out, out)
    return out


class GlyphDataset(Dataset):
    """Holds a subset of glyphs and yields model-ready (image, label) tensors."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        augment: bool = False,
        translate_px: int = 2,
        noise_flip_prob: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.X = X[indices]
        self.y = y[indices]
        self.orig_index = indices
        self.augment = augment
        self.translate_px = translate_px
        self.noise_flip_prob = noise_flip_prob
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = self.X[i].astype(np.float32)
        if self.augment:
            img = _augment(img, self.translate_px, self.noise_flip_prob, self._rng).astype(np.float32)
        img = img * PIXEL_SCALE  # match the model's training-time input range
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 32, 32)
        return tensor, int(self.y[i])


def build_splits(cfg: Config) -> Splits:
    X, y, _ = load_npz(cfg.data_path)
    tr, va, te = stratified_split(y, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    return Splits(
        train=GlyphDataset(
            X, y, tr, augment=cfg.augment, translate_px=cfg.translate_px,
            noise_flip_prob=cfg.noise_flip_prob, seed=cfg.seed,
        ),
        val=GlyphDataset(X, y, va, augment=False, seed=cfg.seed),
        test=GlyphDataset(X, y, te, augment=False, seed=cfg.seed),
    )


def build_loaders(cfg: Config, splits: Splits) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return (
        DataLoader(splits.train, shuffle=True, drop_last=len(splits.train) > cfg.batch_size, **common),
        DataLoader(splits.val, shuffle=False, **common),
        DataLoader(splits.test, shuffle=False, **common),
    )
