"""Training and evaluation loops.

Implements a single-epoch train/eval step, a full training routine with
ReduceLROnPlateau scheduling, early stopping on validation loss, best-checkpoint
saving, and a per-sample inference helper used to build the results dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .metrics import accuracy
from .model import GlyphCNN
from .utils import get_logger

log = get_logger()


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_true, all_pred = 0.0, [], []
    torch.set_grad_enabled(train)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        all_true.append(labels.cpu().numpy())
        all_pred.append(outputs.argmax(1).cpu().numpy())
    torch.set_grad_enabled(True)
    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    n = max(1, len(y_true))
    return total_loss / n, accuracy(y_true, y_pred)


def train(
    cfg: Config,
    model: GlyphCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.lr_patience, min_lr=cfg.min_lr
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val, best_state, epochs_no_improve, best_epoch = float("inf"), None, 0, 0
    ckpt_path = Path(cfg.models_dir) / f"{cfg.run_name}.pth"

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr)

        if va_loss < best_val - 1e-4:
            best_val, best_epoch, epochs_no_improve = va_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                f"epoch {epoch:3d}/{cfg.epochs} | train {tr_loss:.3f}/{tr_acc:.2%} "
                f"| val {va_loss:.3f}/{va_acc:.2%} | lr {lr:.1e}"
            )
        if epochs_no_improve >= cfg.early_stop_patience:
            log.info(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    history["checkpoint"] = str(ckpt_path)
    return history


@torch.no_grad()
def infer(model, loader, device) -> dict:
    """Return per-sample predictions, probabilities and true labels."""
    model.eval()
    y_true, y_pred, confidence, probs = [], [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        p = torch.softmax(model(inputs), dim=1).cpu().numpy()
        y_pred.append(p.argmax(1))
        confidence.append(p.max(1))
        probs.append(p)
        y_true.append(labels.numpy())
    return {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "confidence": np.concatenate(confidence),
        "probs": np.concatenate(probs),
    }
