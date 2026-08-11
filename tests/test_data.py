import numpy as np

from glyphcnn.config import PIXEL_SCALE, Config
from glyphcnn.data import GlyphDataset, build_splits, load_npz, stratified_split


def test_stratified_split_covers_all_classes_in_train():
    y = np.repeat(np.arange(21), 5)
    tr, va, te = stratified_split(y, val_fraction=0.15, test_fraction=0.15, seed=0)
    # no overlap
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert len(tr) + len(va) + len(te) == len(y)
    # every class present in train
    assert set(np.unique(y[tr])) == set(range(21))


def test_dataset_scales_to_model_range(tiny_dataset):
    X, y, idx = load_npz(tiny_dataset)
    ds = GlyphDataset(X, y, np.arange(len(y)), augment=False)
    img, label = ds[0]
    assert img.shape == (1, 32, 32)
    assert float(img.max()) in (0.0, PIXEL_SCALE)
    assert 0 <= label < 21


def test_build_splits_from_file(tiny_dataset):
    cfg = Config(data_path=tiny_dataset)
    splits = build_splits(cfg)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == 21 * 5
    assert len(splits.train) > 0
