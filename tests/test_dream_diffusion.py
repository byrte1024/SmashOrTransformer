import numpy as np
import torch
from PIL import Image
from model.model import SmashRanker
from model import dream_diffusion as dd


def _tiny_model():
    return SmashRanker("vit_tiny_patch16_224", resolution=32, dropout=0.0, pretrained=False).eval()


def test_score_tensor_shape_and_range():
    m = _tiny_model()
    img01 = torch.rand(2, 3, 96, 96)            # B,3,H,W in [0,1]
    s = dd._score_tensor(m, img01, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 32)
    assert s.shape == (2,) and float(s.min()) >= 0 and float(s.max()) <= 1


def test_smash_loss_grad_flows_and_direction():
    m = _tiny_model()
    img = torch.rand(1, 3, 96, 96, requires_grad=True)
    loss = dd.smash_loss(m, img, target=1.0, mean=(0.5,) * 3, std=(0.5,) * 3, resolution=32)
    loss.backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


def test_to_pil_roundtrip():
    img01 = torch.zeros(1, 3, 8, 8); img01[:, 0] = 1.0   # red
    pil = dd.to_pil(img01)
    assert isinstance(pil, Image.Image) and pil.size == (8, 8)
    assert np.asarray(pil)[0, 0, 0] == 255
