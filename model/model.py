from __future__ import annotations
import timm
import torch
import torch.nn as nn
from timm.data import resolve_model_data_config


class SmashRanker(nn.Module):
    """timm backbone (ViT) + scalar regression head. forward -> logit [B]."""

    def __init__(self, backbone: str, resolution: int, dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, img_size=resolution)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))
        self.data_config = resolve_model_data_config(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats).squeeze(-1)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
