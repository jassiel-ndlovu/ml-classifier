"""Central configuration for the GlyphCNN pipeline.

Everything that governs a run (paths, hyper-parameters, split ratios, seed) lives
here as a single dataclass so runs are reproducible and self-documenting. The CLI
overrides fields from the command line; nothing is hard-coded deep in the code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# The 21 classes are the capital letters A-Y, excluding D, H, K, X and Z.
# Index position == integer class label used throughout the project.
CLASS_NAMES: list[str] = [
    "A", "B", "C", "E", "F", "G", "I", "J", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y",
]

NUM_CLASSES: int = len(CLASS_NAMES)
IMAGE_SIZE: int = 32
# The pretrained model was trained on inputs scaled to {0, 255}; we match that.
PIXEL_SCALE: float = 255.0


@dataclass
class Config:
    """A single, serialisable description of a training/evaluation run."""

    # --- data ---
    data_path: Path = Path("data/processed/dataset.npz")
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    # --- model / optimisation ---
    num_classes: int = NUM_CLASSES
    batch_size: int = 16
    epochs: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.30
    label_smoothing: float = 0.05

    # --- scheduling / early stopping ---
    lr_patience: int = 8      # ReduceLROnPlateau patience (epochs)
    early_stop_patience: int = 30
    min_lr: float = 1e-5

    # --- augmentation (train only) ---
    augment: bool = True
    translate_px: int = 3     # max random shift in pixels
    noise_flip_prob: float = 0.05  # per-pixel salt-and-pepper flip probability

    # --- runtime ---
    seed: int = 0
    device: str = "auto"      # "auto" | "cpu" | "cuda"
    num_workers: int = 0

    # --- output locations ---
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    run_name: str = "glyphcnn"

    # populated at load time; not set by the user
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
