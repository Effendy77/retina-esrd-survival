#!/usr/bin/env python
import argparse, os, json, numpy as np
import torch
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

from src.data.ukb_survival_dataset import UKBSurvivalDataset
from src.models.deepsurv_retina import DeepSurvRetina
from src.train.train_survival import build_transforms

@torch.no_grad()
def evaluate(model, dl, device, flip_sign=False):
    model.eval()
    all_r, all_t, all_e = [], [], []
    for imgs, t, e in dl:
        imgs = imgs.to(device)
        r = model(imgs).cpu().numpy().reshape(-1)
        all_r.append(r); all_t.append(t.numpy()); all_e.append(e.numpy())
    all_r = np.concatenate(all_r); all_t = np.concatenate(all_t); all_e = np.concatenate(all_e)
    if flip_sign:
        all_r = -all_r
    c = concordance_index(all_t, all_r, event_observed=all_e)
    return float(c), all_r, all_t, all_e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="Fold checkpoint (best_cindex.pth)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cindex_flip", action="store_true")
    ap.add_argument("--retfound_ckpt", default="checkpoints/retfound_encoder.bin", help="Path to RETFound encoder weights")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_val = build_transforms(args.img_size, train=False)
    ds = UKBSurvivalDataset(args.test_csv, args.img_dir, tf_val)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Initialize model with RETFound backbone (required)
    model = DeepSurvRetina(weights_path=args.retfound_ckpt).to(device)

    # Load fold checkpoint (trained head + possibly partial backbone)
    ck = torch.load(args.checkpoint, map_location=device)
    sd = ck.get("model", ck.get("state_dict", ck))
    model.load_state_dict(sd, strict=False)

    # Evaluate: flip sign if requested (keeps behavior identical to training with --cindex_flip)
    c, r, t, e = evaluate(model, dl, device, flip_sign=args.cindex_flip)
    print(f"Holdout C-index: {c:.4f}")

    # Save outputs
    np.savez(os.path.join(args.outdir, "preds.npz"), risk=r, time=t, event=e)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"cindex": float(c)}, f, indent=2)

if __name__ == "__main__":
    main()
