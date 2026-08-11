import numpy as np
import pytest


@pytest.fixture
def tiny_dataset(tmp_path):
    """A small synthetic dataset (.npz) with all 21 classes represented."""
    rng = np.random.default_rng(0)
    per_class = 5
    n = 21 * per_class
    X = (rng.random((n, 32, 32)) > 0.5).astype(np.uint8)
    y = np.repeat(np.arange(21), per_class).astype(np.int64)
    idx = np.arange(n, dtype=np.int64)
    path = tmp_path / "dataset.npz"
    np.savez_compressed(path, X=X, X_raw=X, y=y, idx=idx, orient=np.zeros(n, dtype=np.int64))
    return path
