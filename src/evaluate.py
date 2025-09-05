
import argparse, os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.fundus_survival_dataset import FundusSurvivalDataset as SurvivalImageDataset
from src.model.multimodal_model import build_model
from src.utils.metrics import concordance_index, time_dependent_auc

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['survival'], default='survival')
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--image_root', default='')
    ap.add_argument('--target', default='time_to_event')
    ap.add_argument('--event_col', default='event_occurred')
    ap.add_argument('--tabular_cols', default='')
    ap.add_argument('--tabular_norm', choices=['none','standardize'], default='standardize')
    ap.add_argument('--fusion', choices=['concat'], default='concat')
    ap.add_argument('--backbone', default='retfound_vit_base')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=448)
    ap.add_argument('--eval_at_years', nargs='+', type=float, default=[2.0, 5.0])
    ap.add_argument('--out_scores', default='outputs/val_scores.csv')
    ap.add_argument('--flip_right_eye', type=str, default='true')
    return ap.parse_args()

def str2bool(x): return str(x).lower() in ['1','true','t','yes','y']

def collate(batch):
    imgs = torch.stack([b['image'] for b in batch], dim=0)
    times = torch.tensor([b['time'] for b in batch], dtype=torch.float32)
    events = torch.tensor([b['event'] for b in batch], dtype=torch.float32)
    tabs = None
    if batch[0]['tab'] is not None:
        tabs = torch.stack([b['tab'] for b in batch], dim=0)
    return imgs, times, events, tabs

def main():
    args = parse_args()
    flip_right = str2bool(args.flip_right_eye)
    tab_cols = [c.strip() for c in args.tabular_cols.split(',') if c.strip()]
    df = pd.read_csv(args.val_csv)

    ds = SurvivalImageDataset(df, image_root=args.image_root,
                              img_size=args.img_size, time_col=args.target,
                              event_col=args.event_col, tabular_cols=tab_cols,
                              tabular_norm=args.tabular_norm, fit_stats=False,
                              flip_right_eye=flip_right)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)

    model = build_model(backbone=args.backbone, checkpoint='', tabular_dim=len(tab_cols), fusion=args.fusion)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')); model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)

    all_r, all_t, all_e = [], [], []
    with torch.no_grad():
        for imgs, times, events, tabs in loader:
            imgs, times, events = imgs.to(device), times.to(device), events.to(device)
            tabs = tabs.to(device) if tabs is not None else None
            risk = model(imgs, tabs)
            all_r.append(risk.detach().cpu().flatten())
            all_t.append(times.cpu())
            all_e.append(events.cpu())
    import torch as _torch
    risk = _torch.cat(all_r).numpy(); times = _torch.cat(all_t).numpy(); events = _torch.cat(all_e).numpy()

    cidx = concordance_index(times, -risk, events)
    print(f'C-index: {cidx:.4f}')

    out = {'eid': df['eid'].tolist(), 'risk': risk.tolist(), 'time': times.tolist(), 'event': events.tolist()}
    os.makedirs(os.path.dirname(args.out_scores), exist_ok=True)
    pd.DataFrame(out).to_csv(args.out_scores, index=False)

    for t in args.eval_at_years:
        auc_t = time_dependent_auc(times, events, -risk, t)
        print(f'Time-dependent AUC at {t} years: {auc_t:.4f}')

if __name__ == '__main__':
    main()
