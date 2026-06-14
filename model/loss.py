from __future__ import annotations
import torch
import torch.nn.functional as F


def soft_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with soft (fractional) targets in [0,1]. Mean scalar."""
    return F.binary_cross_entropy_with_logits(logits, targets)
