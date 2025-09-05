
import torch
import torch.nn as nn
class DummyBackbone(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()
        self.embed_dim = embed_dim
    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.pool(x).flatten(1)
        return x
def load_backbone(name="retfound_vit_base", checkpoint=""):
    return DummyBackbone(embed_dim=768), 768
