import os, argparse, time, json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from lifelines.utils import concordance_index

from src.data.ukb_survival_dataset import UKBSurvivalDataset
from src.models.deepsurv_retina import DeepSurvRetina


def seed_everything(seed=77):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = "0"


def build_transforms(img_size, train=True):
    img_size = int(img_size)
    if train:
        return T.Compose([
            T.Resize(int(img_size*1.05)),
            T.RandomResizedCrop(img_size, scale=(0.9,1.0), ratio=(0.95,1.05)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            # (Optional) consider commenting these for very imbalanced outcomes:
            # T.RandomGrayscale(p=0.05),
            # T.GaussianBlur(kernel_size=3, sigma=(0.1,1.2)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])


class CoxPHLoss(nn.Module):
    """Negative partial log-likelihood for Cox proportional hazards."""
    def __init__(self): super().__init__()
    def forward(self, preds, times, events):
        # sort by descending time so risk sets are cumulative
        order = torch.argsort(times, descending=True)
        preds = preds[order]; events = events[order]
        log_cumsum_exp = torch.logcumsumexp(preds, dim=0)
        diff = preds - log_cumsum_exp
        return -torch.sum(diff * events) / (events.sum() + 1e-8)


def evaluate(model, dl, device, flip=False):
    model.eval()
    all_r, all_t, all_e = [], [], []
    with torch.no_grad():
        for imgs, t, e in dl:
            imgs = imgs.to(device)
            r = model(imgs).detach().cpu().numpy()
            all_r.append(r); all_t.append(t.numpy()); all_e.append(e.numpy())
    all_r = np.concatenate(all_r); all_t = np.concatenate(all_t); all_e = np.concatenate(all_e)
    scores = -all_r if flip else all_r
    c = concordance_index(all_t, scores, event_observed=all_e)
    return float(c), all_r, all_t, all_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--img_dir",   required=True)
    ap.add_argument("--weights_path", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="outputs/run")
    # extras (stable defaults)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=77)
    # NEW: reproducible toggle for C-index orientation
    ap.add_argument("--cindex_flip", action="store_true",
                    help="Compute C-index with negative scores (flip orientation).")
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_train = build_transforms(args.img_size, train=True)
    tf_val   = build_transforms(args.img_size, train=False)

    tr_ds = UKBSurvivalDataset(args.train_csv, args.img_dir, tf_train)
    va_ds = UKBSurvivalDataset(args.val_csv,   args.img_dir, tf_val)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = DeepSurvRetina(weights_path=args.weights_path).to(device)
    loss_fn = CoxPHLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, verbose=True)

    best_c = -1.0; patience_ct = 0
    rows = []
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_losses = []
        t0 = time.time()
        for imgs, t, e in tr_dl:
            imgs, t, e = imgs.to(device), t.to(device), e.to(device)
            opt.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss = loss_fn(preds, t, e)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        # validation
        val_c, _, _, _ = evaluate(model, va_dl, device, flip=args.cindex_flip)
        # scheduler minimizes, so step on negative C
        sched.step(-val_c)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_cindex": val_c,
            "lr": float(opt.param_groups[0]["lr"]),
            "epoch_sec": round(time.time()-t0, 2),
        }
        rows.append(row)
        print(f"Epoch {epoch:03d}: loss={row['train_loss']:.4f}  valC={val_c:.4f}  lr={row['lr']:.2e}")

        # checkpointing
        last_path = os.path.join(args.out, "last.pth")
        best_path = os.path.join(args.out, "best_cindex.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_cindex": val_c, "args": vars(args)}, last_path)

        improved = (val_c - best_c) >= args.min_delta
        if improved:
            best_c = val_c
            torch.save({"model": model.state_dict(), "best_val_cindex": best_c, "args": vars(args)}, best_path)
            patience_ct = 0
        else:
            patience_ct += 1

        # persist metrics each epoch
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics_epoch.csv"), index=False)
        with open(os.path.join(args.out, "metrics_last.json"), "w") as f:
            json.dump(row, f, indent=2)

        if patience_ct >= args.patience:
            print(f"Early stopping (patience={args.patience}). Best C={best_c:.4f}")
            break

    print(f"Done. Best val C-index: {best_c:.4f} | saved to {best_path}")


if __name__ == "__main__":
    main()
