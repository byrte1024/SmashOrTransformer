import torch
from model.model import SmashRanker


def _tiny():
    return SmashRanker(backbone="vit_tiny_patch16_224", resolution=32,
                       dropout=0.0, pretrained=False)


def test_forward_returns_logit_per_sample():
    m = _tiny()
    x = torch.randn(2, 3, 32, 32)
    out = m(x)
    assert out.shape == (2,)
    assert out.dtype == torch.float32


def test_data_config_has_mean_std():
    m = _tiny()
    assert len(m.data_config["mean"]) == 3
    assert len(m.data_config["std"]) == 3


def test_freeze_unfreeze_toggles_grad():
    m = _tiny()
    m.freeze_backbone()
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    m.unfreeze_backbone()
    assert all(p.requires_grad for p in m.backbone.parameters())
