"""Assemble the self-contained static results dashboard (site/index.html).

Reads the JSON artifacts under reports/ and the processed glyph images, encodes
every glyph as a base64 PNG, and injects the whole payload into a single HTML file
that renders training curves, a confusion matrix, per-class metrics and an
interactive gallery of classified samples with no external dependencies.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SITE = ROOT / "site"


def glyph_data_uri(img01: np.ndarray, size: int = 64) -> str:
    im = Image.fromarray((img01 * 255).astype("uint8")).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mean_glyph_uri(X: np.ndarray, size: int = 64) -> str:
    m = X.mean(0)
    m = (m / m.max() * 255).astype("uint8") if m.max() else m.astype("uint8")
    im = Image.fromarray(m).resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_payload() -> dict:
    data = np.load(ROOT / "data" / "processed" / "dataset.npz")
    X, y = data["X"], data["y"]

    pretrained = json.loads((REPORTS / "metrics_pretrained_all.json").read_text())
    samples = json.loads((REPORTS / "samples_pretrained_all.json").read_text())
    history = json.loads((REPORTS / "history.json").read_text())
    scratch = json.loads((REPORTS / "metrics_test.json").read_text())
    cv = json.loads((REPORTS / "scratch_cv.json").read_text())
    class_names = pretrained["class_names"]

    # embed a thumbnail for every classified sample (index aligns with dataset order)
    for s in samples:
        s["img"] = glyph_data_uri(X[s["index"]])

    # one clean mean-glyph per class + support counts
    classes = []
    for c, name in enumerate(class_names):
        mask = y == c
        classes.append({
            "id": c,
            "name": name,
            "count": int(mask.sum()),
            "mean_img": mean_glyph_uri(X[mask]),
        })

    return {
        "class_names": class_names,
        "pretrained": pretrained,
        "samples": samples,
        "history": history,
        "scratch": scratch,
        "scratch_cv": cv,
        "classes": classes,
        "n_glyphs": int(len(y)),
    }


def main() -> None:
    payload = build_payload()
    template = (Path(__file__).parent / "dashboard_template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(payload))
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "index.html").write_text(html)
    size_kb = len((SITE / "index.html").read_text()) / 1024
    print(f"wrote {SITE/'index.html'} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
