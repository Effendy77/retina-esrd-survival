import torch, timm
from torch import nn

class RETFoundBackbone(nn.Module):
    def __init__(self, weights_path: str, img_size: int = 224, freeze_until: str | None = None):
        super().__init__()
        self.encoder = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        self.encoder.load_state_dict(sd, strict=False)
        if freeze_until:
            for name, p in self.encoder.named_parameters():
                if name.startswith(freeze_until):
                    p.requires_grad = False

    def forward(self, x):
        z = self.encoder.forward_features(x)  # [B,768]
        return z
