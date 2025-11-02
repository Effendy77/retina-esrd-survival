import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from src.models.deepsurv_retina import DeepSurvRetina
from src.data.ukb_survival_dataset import UKBSurvivalDataset

VAL="data/metadata/cv/fold0_val.csv"
IMG="data/images"
CKPT="outputs/surv_smoke_f0/best_cindex.pth"

# dataset (builds proper val transforms internally if needed)
ds = UKBSurvivalDataset(VAL, IMG, None, train=False)
dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

m = DeepSurvRetina(weights_path="checkpoints/retfound_encoder.bin")
sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("model", sd.get("state_dict", sd))
m.load_state_dict(sd, strict=False)
m = m.cuda() if torch.cuda.is_available() else m
m.eval()

all_p, all_t, all_e = [], [], []
with torch.no_grad():
    for imgs, t, e in dl:
        imgs = imgs.cuda() if torch.cuda.is_available() else imgs
        p = m(imgs).detach().cpu().numpy()
        all_p.append(p); all_t.append(t.numpy()); all_e.append(e.numpy())
import numpy as np
p = np.concatenate(all_p); T = np.concatenate(all_t); E = np.concatenate(all_e)
c_pos = concordance_index(T, p,  event_observed=E)
c_neg = concordance_index(T, -p, event_observed=E)
print(f"C-index using risk:  {c_pos:.4f}")
print(f"C-index using -risk: {c_neg:.4f}")
