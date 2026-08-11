"""Inference utilities: load a checkpoint and classify glyph arrays.

This is the modern replacement for the original ``classifyall.py``. It accepts a
processed ``.npz`` (or a numpy array of upright {0,1} glyphs) and writes predicted
labels, matching the behaviour the original assignment harness expected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import PIXEL_SCALE, Config
from .model import GlyphCNN
from .utils import resolve_device


def load_model(checkpoint: Path, num_classes: int, dropout: float, device: torch.device) -> GlyphCNN:
    model = GlyphCNN(num_classes=num_classes, dropout=dropout).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def classify_array(model: GlyphCNN, X: np.ndarray, device: torch.device, batch_size: int = 64) -> dict:
    """Classify upright {0,1} glyphs of shape (N, 32, 32)."""
    X = (X.astype(np.float32) * PIXEL_SCALE).reshape(-1, 1, 32, 32)
    preds, confs = [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            p = torch.softmax(model(batch), dim=1).cpu().numpy()
            preds.append(p.argmax(1))
            confs.append(p.max(1))
    return {"pred": np.concatenate(preds), "confidence": np.concatenate(confs)}


def classify_file(cfg: Config, checkpoint: Path, npz_path: Path, out_txt: Path) -> Path:
    device = resolve_device(cfg.device)
    model = load_model(checkpoint, cfg.num_classes, cfg.dropout, device)
    X = np.load(npz_path)["X"]
    result = classify_array(model, X, device)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_txt, result["pred"].astype(int), fmt="%d")
    return out_txt
