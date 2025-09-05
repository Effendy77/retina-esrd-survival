
from .retfound_backbone import load_backbone
from .survival_head import CoxHead
import torch.nn as nn
class MultiModalSurvival(nn.Module):
    def __init__(self, backbone="retfound_vit_base", checkpoint="", tabular_dim=0, fusion="concat"):
        super().__init__()
        self.backbone, emb_dim = load_backbone(backbone, checkpoint)
        self.head = CoxHead(in_dim=emb_dim, hidden=256, tabular_dim=tabular_dim, fusion=fusion)
    def forward(self, images, tabular=None):
        z = self.backbone(images)
        return self.head(z, tabular)
def build_model(backbone="retfound_vit_base", checkpoint="", tabular_dim=0, fusion="concat"):
    return MultiModalSurvival(backbone=backbone, checkpoint=checkpoint, tabular_dim=tabular_dim, fusion=fusion)
