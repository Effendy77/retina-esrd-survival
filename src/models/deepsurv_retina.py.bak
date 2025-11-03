import os
import torch
import torch.nn as nn
from .retfound_backbone import RETFoundBackbone

class DeepSurvRetina(nn.Module):
    """
    RETFound ViT backbone + simple DeepSurv head (linear risk).
    Head in_features is taken from backbone.num_features (1024 for ViT-L).
    """
    def __init__(self, weights_path: str, model_name: str | None = None, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = RETFoundBackbone(weights_path=weights_path, model_name=model_name)
        feat_dim = getattr(self.backbone.encoder, "num_features", None)
        if feat_dim is None:
            # fallback for older timm
            feat_dim = self.backbone.encoder.num_features if hasattr(self.backbone.encoder, "num_features") else 1024
        # DeepSurv head: MLP → 1 (log-risk)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        f = self.backbone(x)         # (B, feat_dim)
        log_risk = self.head(f).squeeze(1)
        return log_risk
