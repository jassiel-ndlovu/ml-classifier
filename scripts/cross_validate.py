"""Multi-seed cross-validation of the from-scratch pipeline.

The recovered dataset is tiny (~5 images/class), so a single train/test split is
high-variance. This runs the full training pipeline across several seeds and writes
a stable mean +/- std to ``reports/scratch_cv.json`` (consumed by the dashboard).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from glyphcnn.config import Config
from glyphcnn.data import build_loaders, build_splits
from glyphcnn.engine import infer, train
from glyphcnn.metrics import accuracy
from glyphcnn.model import GlyphCNN
from glyphcnn.utils import resolve_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="reports/scratch_cv.json")
    args = ap.parse_args()
    logging.getLogger("glyphcnn").setLevel(logging.WARNING)

    accs = []
    for seed in range(args.seeds):
        cfg = Config(seed=seed)
        set_seed(seed)
        device = resolve_device(cfg.device)
        splits = build_splits(cfg)
        tl, vl, te = build_loaders(cfg, splits)
        model = GlyphCNN(cfg.num_classes, cfg.dropout).to(device)
        train(cfg, model, tl, vl, device)
        r = infer(model, te, device)
        a = accuracy(r["y_true"], r["y_pred"])
        accs.append(float(a))
        print(f"seed {seed}: test accuracy {a:.3f}")

    summary = {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "runs": accs,
        "n_seeds": len(accs),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"MEAN {100*np.mean(accs):.1f}% +/- {100*np.std(accs):.1f}%  -> {args.out}")


if __name__ == "__main__":
    main()
