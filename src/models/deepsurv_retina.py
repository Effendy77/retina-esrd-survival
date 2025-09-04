from torch import nn
from .retfound_backbone import RETFoundBackbone
from .survival_head import DeepSurvHead

class DeepSurvRetina(nn.Module):
    def __init__(self, weights_path, in_dim=768):
        super().__init__()
        self.backbone = RETFoundBackbone(weights_path=weights_path)
        self.head = DeepSurvHead(in_dim=in_dim)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)
