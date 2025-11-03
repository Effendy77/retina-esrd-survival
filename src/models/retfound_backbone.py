import os
import torch
import torch.nn as nn
import timm

class RETFoundBackbone(nn.Module):
    """
    ViT backbone for RETFound.
    - Defaults to ViT-L/16 (1024-dim) to match the RETFound_mae_natureCFP weights.
    - Safely loads only matching keys from the checkpoint (handles prefixes & shape mismatches).
    """
    def __init__(self, weights_path: str, model_name: str | None = None):
        super().__init__()
        # allow override via env var RETF_BACKBONE, else default to ViT-L/16
        model_name = model_name or os.getenv("RETF_BACKBONE", "vit_large_patch16_224")
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")

        # Load checkpoint
        if weights_path and os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                sd = sd["model"]

            # strip common prefixes and keep only matching shapes
            enc_sd = self.encoder.state_dict()
            new_sd = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("module."): nk = nk[len("module."):]
                for pref in ("encoder.", "backbone.", "model."):
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                if nk in enc_sd and enc_sd[nk].shape == v.shape:
                    new_sd[nk] = v
            # merge and load
            enc_sd.update(new_sd)
            missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        else:
            raise FileNotFoundError(f"RETFound weights not found at: {weights_path}")

    def forward(self, x):
        # timm model with num_classes=0, global_pool='avg' returns feature vector
        return self.encoder(x)
