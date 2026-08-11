import torch

from glyphcnn.config import NUM_CLASSES
from glyphcnn.model import GlyphCNN


def test_forward_output_shape():
    model = GlyphCNN(num_classes=NUM_CLASSES)
    x = torch.rand(4, 1, 32, 32)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_predict_proba_sums_to_one():
    model = GlyphCNN(num_classes=NUM_CLASSES)
    x = torch.rand(3, 1, 32, 32)
    p = model.predict_proba(x)
    assert p.shape == (3, NUM_CLASSES)
    assert torch.allclose(p.sum(dim=1), torch.ones(3), atol=1e-5)


def test_pretrained_checkpoint_loads_if_present():
    # architecture must stay compatible with the historical checkpoint
    import pathlib

    ckpt = pathlib.Path("models/pretrained_model.pth")
    if not ckpt.exists():
        return
    model = GlyphCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
