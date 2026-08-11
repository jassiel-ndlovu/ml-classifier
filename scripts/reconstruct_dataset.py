"""Reconstruct the 32x32 binary glyph dataset from rendered matplotlib PNGs.

The original raw pixel text files for this project were lost; all that remained
were 151 matplotlib-rendered PNGs (viridis colormap) with ``idx``, ``orient`` and
``label`` encoded in each filename. This script recovers the underlying 32x32
binary arrays by detecting the axes data-region (viridis purple background +
yellow foreground), cropping it, and resampling to 32x32.

Outputs
-------
- ``data/processed/dataset.npz`` : X (N,32,32) uint8 {0,1}, y, idx, orient
- ``data/processed/samples/class_XX/*.png`` : clean upscaled PNGs for the gallery
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

FNAME_RE = re.compile(r"idx=(?P<idx>[\d.]+)-orient=(?P<orient>[\d.]+)-label=(?P<label>[\d.]+)")


def reconstruct_one(path: Path) -> np.ndarray:
    """Return a 32x32 uint8 {0,1} array recovered from a rendered PNG."""
    im = np.array(Image.open(path).convert("RGB")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]

    # viridis background ~ (68,1,84): blue-ish, low green; foreground ~ yellow.
    purple = (b > 50) & (b < 140) & (r < 110) & (g < 70) & (b > r)
    yellow = (r > 150) & (g > 150) & (b < 130)
    data_region = purple | yellow
    ys, xs = np.where(data_region)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    fg = yellow[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255
    small = np.array(Image.fromarray(fg).resize((32, 32), Image.NEAREST))
    return (small > 127).astype(np.uint8)


def correct_orientation(img: np.ndarray, orient: int) -> np.ndarray:
    """Rotate a raw glyph to its canonical upright pose.

    Mirrors the original project's convention (validated against the pretrained
    model, which scores 93% once this correction is applied):
    orient 1 -> rot90 k=3, orient 2 -> k=2, orient 3 -> k=1, orient 0 -> none.
    """
    k = {0: 0, 1: 3, 2: 2, 3: 1}[int(orient)]
    return np.rot90(img, k).copy()


def parse_meta(path: Path) -> dict:
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot parse metadata from {path.name}")
    return {
        "idx": int(float(m["idx"])),
        "orient": int(float(m["orient"])),
        "label": int(float(m["label"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="folder of rendered PNGs")
    ap.add_argument("--out", default=Path("data/processed"), type=Path)
    args = ap.parse_args()

    files = sorted(args.src.glob("*.png"))
    if not files:
        raise SystemExit(f"no PNGs found in {args.src}")

    X, X_raw, y, idxs, orients = [], [], [], [], []
    sample_dir = args.out / "samples"
    for f in files:
        meta = parse_meta(f)
        raw = reconstruct_one(f)
        img = correct_orientation(raw, meta["orient"])  # canonical upright pose
        X.append(img)
        X_raw.append(raw)
        y.append(meta["label"])
        idxs.append(meta["idx"])
        orients.append(meta["orient"])

        cls_dir = sample_dir / f"class_{meta['label']:02d}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        up = Image.fromarray((img * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST)
        up.save(cls_dir / f"idx={meta['idx']}.png")

    X = np.stack(X).astype(np.uint8)
    X_raw = np.stack(X_raw).astype(np.uint8)
    y = np.array(y, dtype=np.int64)
    idxs = np.array(idxs, dtype=np.int64)
    orients = np.array(orients, dtype=np.int64)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "dataset.npz", X=X, X_raw=X_raw, y=y, idx=idxs, orient=orients
    )
    print(f"reconstructed {len(X)} images, {len(np.unique(y))} classes")
    print("class counts:", dict(zip(*np.unique(y, return_counts=True))))


if __name__ == "__main__":
    main()
