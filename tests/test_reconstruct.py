import importlib.util
import sys
from pathlib import Path

import numpy as np

# load the standalone reconstruction script as a module
_spec = importlib.util.spec_from_file_location(
    "reconstruct_dataset", Path(__file__).resolve().parents[1] / "scripts" / "reconstruct_dataset.py"
)
recon = importlib.util.module_from_spec(_spec)
sys.modules["reconstruct_dataset"] = recon
_spec.loader.exec_module(recon)


def test_parse_meta():
    meta = recon.parse_meta(Path("idx=100-orient=2.0-label=12.0-32x32.png"))
    assert meta == {"idx": 100, "orient": 2, "label": 12}


def test_correct_orientation_is_invertible():
    img = np.arange(16).reshape(4, 4)
    # applying orientation-3 then its inverse (orientation-1) returns the original
    once = recon.correct_orientation(img, 3)
    back = recon.correct_orientation(once, 1)
    assert np.array_equal(back, img)


def test_correct_orientation_180_twice_identity():
    img = np.random.default_rng(0).integers(0, 2, (5, 5))
    twice = recon.correct_orientation(recon.correct_orientation(img, 2), 2)
    assert np.array_equal(twice, img)
