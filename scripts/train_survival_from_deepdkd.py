#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Train Cox Survival Model using DeepDKD pretrained encoder.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class CoxPHLoss(nn.Module):
    """Negative partial log-likelihood loss for Cox proportional hazards model."""
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        """
        preds: tensor of shape (N,) or (N,1) - predicted risk scores (higher -> higher risk)
        target: tuple (durations, events) where durations and events are tensors of shape (N,)
        Implements -sum_{i: e_i=1} [pred_i - log(sum_{j: t_j >= t_i} exp(pred_j))] / num_events
        """
        durations, events = target
        preds = preds.view(-1)
        durations = durations.view(-1)
        events = events.view(-1)

        # sort by descending durations so each risk set is a suffix
        order = torch.argsort(durations, descending=True)
        preds = preds[order]
        events = events[order]

        # compute log cumulative sum of exp(preds) for risk sets efficiently
        # reverse preds, compute logcumsumexp, then flip back
        rev_preds = preds.flip(0)
        log_cumsum = torch.logcumsumexp(rev_preds, dim=0).flip(0)

        # contribution only from observed events
        diff = preds - log_cumsum
        denom = events.sum().clamp_min(1.0)  # avoid division by zero
        loss = - (diff * events).sum() / denom
        return loss

from lifelines.utils import concordance_index
from tqdm import tqdm

# Ensure repo root is on sys.path so we can import deepdkd as a package
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from deepdkd.model import Model as DeepDKD

class CoxHead(nn.Module):
    """Cox proportional hazards head."""
    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeepDKD_Survival(nn.Module):
    """DeepDKD backbone + Cox head for survival."""
    def __init__(self, encoder_path=None, freeze_encoder=True):
        super().__init__()
        self.encoder = DeepDKD(output_size=512)
        # infer encoder embedding size dynamically
        import torch
        try:
            with torch.no_grad():
                _d = torch.randn(1,3,224,224)
                _o = self.encoder(_d)
                if isinstance(_o, (tuple,list)):
                    _o = _o[0]
                # handle single-dim outputs
                if hasattr(_o, 'shape') and len(_o.shape)>1:
                    self.emb_dim = int(_o.shape[1])
                else:
                    self.emb_dim = int(getattr(_o,'shape',(_o,))[0])
        except Exception:
            self.emb_dim = 512

        if encoder_path and os.path.exists(encoder_path):
            ckpt = torch.load(encoder_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            # clean keys if saved from DDP
            state = {k.replace('module.', ''): v for k, v in state.items()}
            # filter encoder keys if present
            filtered = {}
            for k, v in state.items():
                if k.startswith("model.encoder.backbone.") or k.startswith("encoder.backbone.") or k.startswith("backbone.") or k.startswith("module.model.encoder.backbone."):
                    nk = k.replace("module.", "")
                    nk = nk.replace("model.encoder.backbone.", "").replace("encoder.backbone.", "").replace("backbone.", "")
                    filtered[nk] = v
            load_state = filtered if len(filtered)>0 else state
            self.encoder.load_state_dict(load_state, strict=False)
            print(f"Loaded pretrained DeepDKD weights from: {encoder_path}")
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # get embedding dimension
        self.embed_dim = getattr(self.encoder, "out_dim", 512)
        self.head = CoxHead(self.embed_dim)

    def forward(self, x):
        emb = self.encoder(x)
        if isinstance(emb, (list, tuple)):
            emb = emb[-1]
        return self.head(emb), emb


class SurvivalDataset(Dataset):
    """
    Expected CSV format:
    image_path, duration, event
    """
    def __init__(self, csv_path, img_dir, transform=None):
        # set default transform if none provided
        if transform is None:
            from torchvision import transforms as _transforms
            transform = _transforms.Compose([
                _transforms.Resize((224,224)),
                _transforms.RandomHorizontalFlip(),
                _transforms.ToTensor(),
                _transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row['image_path'])
        from PIL import Image
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        duration = torch.tensor(row['duration'], dtype=torch.float32)
        event = torch.tensor(row['event'], dtype=torch.float32)
        return img, duration, event


def save_checkpoint(outdir, epoch, model, optimizer):
    path = os.path.join(outdir, f"checkpoint_epoch_{epoch:03d}.pth")
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, path)


def save_val_predictions(outdir, epoch, preds, durations, events):
    np.savez(
        os.path.join(outdir, f"val_preds_epoch_{epoch:03d}.npz"),
        preds=preds,
        durations=durations,
        events=events
    )


def train(args):
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reproducibility
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # transforms
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # datasets and loaders
    train_ds = SurvivalDataset(args.train_csv, args.img_dir, transform=train_tfms)
    val_ds = SurvivalDataset(args.val_csv, args.img_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    model = DeepDKD_Survival(args.encoder_path, args.freeze_encoder).to(device)

    # optimizer and loss
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-6)
    criterion = CoxPHLoss()

    best_val_c = -1.0
    metrics_file = os.path.join(args.outdir, "metrics_epoch.csv")
    with open(metrics_file, "w") as mf:
        mf.write("epoch,train_loss,val_cindex\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, durations, events in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs = imgs.to(device)
            durations = durations.to(device)
            events = events.to(device)
            optimizer.zero_grad()
            preds, _ = model(imgs)
            loss = criterion(preds, (durations, events))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        all_preds, all_dur, all_ev = [], [], []
        with torch.no_grad():
            for imgs, durations, events in DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers):
                imgs = imgs.to(device)
                preds, _ = model(imgs)
                preds = preds.detach().cpu().numpy()
                all_preds.extend(preds)
                all_dur.extend(durations.numpy())
                all_ev.extend(events.numpy())

        val_c = concordance_index(all_dur, -np.array(all_preds), all_ev)
        save_val_predictions(args.outdir, epoch, np.array(all_preds), np.array(all_dur), np.array(all_ev))
        save_checkpoint(args.outdir, epoch, model, optimizer)

        with open(metrics_file, "a") as mf:
            mf.write(f"{epoch},{train_loss:.6f},{val_c:.6f}\n")

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val C-Index: {val_c:.4f}")

        if val_c > best_val_c:
            best_val_c = val_c
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_cindex.pth"))

    print(f"Training complete. Best val C-index = {best_val_c:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cox Survival Model from DeepDKD Encoder")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train CSV (image_path,duration,event)")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to val CSV (image_path,duration,event)")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--encoder_path", type=str, default=None, help="Pretrained DeepDKD encoder weights")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder weights")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold index")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
