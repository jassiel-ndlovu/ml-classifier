"""Command-line interface for the GlyphCNN pipeline.

Examples
--------
    python -m glyphcnn train --epochs 80
    python -m glyphcnn evaluate --checkpoint models/glyphcnn.pth
    python -m glyphcnn evaluate --checkpoint models/pretrained_model.pth --split all
    python -m glyphcnn predict  --checkpoint models/glyphcnn.pth --input data/processed/dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import Config
from .data import GlyphDataset, build_loaders, build_splits, load_npz
from .engine import infer, train
from .metrics import accuracy, confusion_matrix, per_class_metrics
from .model import GlyphCNN
from .predict import classify_file, load_model
from .utils import get_logger, resolve_device, set_seed, write_json
from .viz import plot_confusion, plot_history

log = get_logger()


def _cfg_from_args(args) -> Config:
    cfg = Config()
    for field in vars(cfg):
        if hasattr(args, field) and getattr(args, field) is not None:
            setattr(cfg, field, getattr(args, field))
    cfg.data_path = Path(cfg.data_path)
    cfg.models_dir = Path(cfg.models_dir)
    cfg.reports_dir = Path(cfg.reports_dir)
    return cfg


def _export_reports(cfg: Config, result: dict, tag: str) -> None:
    """Write metrics JSON + figures + per-sample records for the dashboard."""
    y_true, y_pred = result["y_true"], result["y_pred"]
    cm = confusion_matrix(y_true, y_pred, cfg.num_classes)
    metrics = per_class_metrics(cm)
    acc = accuracy(y_true, y_pred)

    reports = Path(cfg.reports_dir)
    write_json(reports / f"metrics_{tag}.json", {
        "tag": tag,
        "accuracy": acc,
        "n_samples": int(len(y_true)),
        "class_names": cfg.class_names,
        "confusion_matrix": cm.tolist(),
        **metrics,
    })

    samples = [
        {
            "index": int(i),
            "true": int(t),
            "pred": int(p),
            "true_name": cfg.class_names[int(t)],
            "pred_name": cfg.class_names[int(p)],
            "confidence": float(c),
            "correct": bool(t == p),
        }
        for i, (t, p, c) in enumerate(zip(y_true, y_pred, result["confidence"]))
    ]
    write_json(reports / f"samples_{tag}.json", samples)
    plot_confusion(cm, cfg.class_names, reports / "figures" / f"confusion_{tag}.png")
    log.info(f"[{tag}] accuracy {acc:.2%} over {len(y_true)} samples -> {reports}")


def cmd_train(args) -> None:
    cfg = _cfg_from_args(args)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    log.info(f"device={device} seed={cfg.seed}")

    splits = build_splits(cfg)
    log.info(f"split sizes: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")
    train_loader, val_loader, test_loader = build_loaders(cfg, splits)

    model = GlyphCNN(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    history = train(cfg, model, train_loader, val_loader, device)

    write_json(Path(cfg.reports_dir) / "history.json", history)
    write_json(Path(cfg.reports_dir) / "config.json", cfg.to_dict())
    plot_history(history, Path(cfg.reports_dir) / "figures" / "history.png")

    result = infer(model, test_loader, device)
    _export_reports(cfg, result, tag="test")


def cmd_evaluate(args) -> None:
    cfg = _cfg_from_args(args)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = load_model(Path(args.checkpoint), cfg.num_classes, cfg.dropout, device)

    X, y, idx = load_npz(cfg.data_path)
    if args.split == "all":
        indices = np.arange(len(y))
        tag = "pretrained_all" if "pretrained" in str(args.checkpoint) else "all"
    else:
        splits = build_splits(cfg)
        ds = {"train": splits.train, "val": splits.val, "test": splits.test}[args.split]
        indices = ds.orig_index
        tag = args.split
    from torch.utils.data import DataLoader

    loader = DataLoader(GlyphDataset(X, y, indices, augment=False), batch_size=cfg.batch_size)
    result = infer(model, loader, device)
    _export_reports(cfg, result, tag=tag)


def cmd_predict(args) -> None:
    cfg = _cfg_from_args(args)
    out = classify_file(cfg, Path(args.checkpoint), Path(args.input), Path(args.output))
    log.info(f"wrote predictions -> {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glyphcnn", description="GlyphCNN pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--data-path", dest="data_path")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--device", choices=["auto", "cpu", "cuda"])
        sp.add_argument("--dropout", type=float)
        sp.add_argument("--reports-dir", dest="reports_dir")
        sp.add_argument("--models-dir", dest="models_dir")

    t = sub.add_parser("train", help="train from scratch")
    common(t)
    t.add_argument("--epochs", type=int)
    t.add_argument("--batch-size", dest="batch_size", type=int)
    t.add_argument("--learning-rate", dest="learning_rate", type=float)
    t.add_argument("--run-name", dest="run_name")
    t.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a checkpoint")
    common(e)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    e.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("predict", help="write predicted labels for an input npz")
    common(pr)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default="predlabels.txt")
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
