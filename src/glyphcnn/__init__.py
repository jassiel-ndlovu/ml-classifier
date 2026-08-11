"""GlyphCNN: a small CNN that classifies noisy 32x32 binary letter glyphs.

A revamp of a 3rd-year supervised-learning project. The package exposes a
config-driven, reproducible pipeline for data loading, training, evaluation and
inference.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import CLASS_NAMES, Config
from .model import GlyphCNN

__all__ = ["Config", "GlyphCNN", "CLASS_NAMES", "__version__"]
