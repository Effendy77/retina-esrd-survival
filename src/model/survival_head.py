
import torch
import torch.nn as nn
class CoxHead(nn.Module):
    def __init__(self, in_dim, hidden=256, tabular_dim=0, fusion="concat"):
        super().__init__()
        fuse_dim = in_dim + (tabular_dim if tabular_dim>0 else 0)
        self.mlp = nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
    def forward(self, img_embed, tab=None):
        x = torch.cat([img_embed, tab], dim=1) if tab is not None else img_embed
        return self.mlp(x).squeeze(1)
