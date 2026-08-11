"""Static matplotlib figures for the reports/ folder and README.

The interactive dashboard is built separately (site/); these PNGs are the
lightweight, embeddable versions used in the README and reports.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1f2933"
ACCENT = "#2f6f8f"
GRID = "#e4e9ee"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_history(history: dict, out: Path) -> Path:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=140)
    a.plot(epochs, history["train_loss"], color=ACCENT, label="train")
    a.plot(epochs, history["val_loss"], color="#c0563b", label="val")
    a.set_title("Loss", color=INK, fontsize=11)
    a.set_xlabel("epoch")
    a.legend(frameon=False)
    b.plot(epochs, history["train_acc"], color=ACCENT, label="train")
    b.plot(epochs, history["val_acc"], color="#c0563b", label="val")
    b.set_title("Accuracy", color=INK, fontsize=11)
    b.set_xlabel("epoch")
    b.set_ylim(0, 1)
    b.legend(frameon=False)
    for ax in (a, b):
        _style(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion(cm: np.ndarray, class_names: list[str], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix", color=INK, fontsize=11)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else INK)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
