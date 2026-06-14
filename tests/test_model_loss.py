import torch
from model.loss import soft_bce


def test_soft_bce_finite_scalar():
    logits = torch.tensor([0.0, 2.0, -1.0])
    targets = torch.tensor([0.5, 0.9, 0.1])
    loss = soft_bce(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_soft_bce_gradients_flow():
    logits = torch.tensor([0.3, -0.4], requires_grad=True)
    targets = torch.tensor([0.7, 0.2])
    soft_bce(logits, targets).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_perfect_prediction_low_loss():
    logits = torch.tensor([10.0, -10.0])
    targets = torch.tensor([1.0, 0.0])
    assert soft_bce(logits, targets) < 1e-3
