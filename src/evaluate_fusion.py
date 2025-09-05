
# src/evaluate_fusion.py
import argparse, os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.model.multimodal_model import build_model
from src.utils.metrics import concordance_index, time_dependent_auc

def str2bool(x):
    return str(x).lower() in ['1','true','t','yes','y']

class EyeImageDataset(Dataset):
    # Minimal dataset for paired evaluation.
    # Expects a DataFrame with columns: 'eid', 'image_path', 'side' ('left' or 'right').
    # Applies right-eye flip if requested.
    def __init__(self, df, img_size=448, flip_right_eye=True):
        self.df = df.reset_index(drop=True)
        self.flip_right_eye = flip_right_eye
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        side = row['side']
        img = Image.open(path).convert('RGB')
        if self.flip_right_eye and side == 'right':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.tf(img)
        return img, row['eid'], side

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_csv', required=True, help='PAIRED CSV with both-eye columns (from build_esrd_survival_with_both_eyes.py)')
    ap.add_argument('--image_root', default='')
    ap.add_argument('--backbone', default='retfound_vit_base')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=448)
    ap.add_argument('--fusion', choices=['mean','max'], default='mean')
    ap.add_argument('--flip_right_eye', type=str, default='true')
    ap.add_argument('--tabular_cols', default='')
    ap.add_argument('--tabular_norm', choices=['none','standardize'], default='standardize')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--eval_at_years', nargs='+', type=float, default=[2.0, 5.0])
    ap.add_argument('--out_scores', default='outputs/val_scores_fused.csv')
    args = ap.parse_args()

    flip_right = str2bool(args.flip_right_eye)
    tab_cols = [c.strip() for c in args.tabular_cols.split(',') if c.strip()]  # not used here (image-only eval)

    df = pd.read_csv(args.val_csv)
    rows = []
    if 'left_image_path' in df.columns and 'right_image_path' in df.columns:
        for _, r in df.iterrows():
            if isinstance(r.get('left_image_path', ''), str) and len(str(r['left_image_path']))>0:
                p = r['left_image_path']
                if not os.path.isabs(p) and args.image_root:
                    p = os.path.join(args.image_root, p)
                rows.append({'eid': r['eid'], 'image_path': p, 'side': 'left'})
            if isinstance(r.get('right_image_path', ''), str) and len(str(r['right_image_path']))>0:
                p = r['right_image_path']
                if not os.path.isabs(p) and args.image_root:
                    p = os.path.join(args.image_root, p)
                rows.append({'eid': r['eid'], 'image_path': p, 'side': 'right'})
        times = df['time_to_event'].values
        events = df['event_occurred'].values
        eids = df['eid'].values
    else:
        raise ValueError('Expected paired CSV with left_image_path and right_image_path columns.')

    eye_df = pd.DataFrame(rows)
    ds = EyeImageDataset(eye_df, img_size=args.img_size, flip_right_eye=flip_right)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(backbone=args.backbone, checkpoint='', tabular_dim=0, fusion='concat')
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_scores = []
    with torch.no_grad():
        for imgs, batch_eids, sides in loader:
            imgs = imgs.to(device)
            risk = model(imgs, None).detach().cpu().numpy().reshape(-1)
            for sc, eid, sd in zip(risk, batch_eids, sides):
                all_scores.append((str(eid), sd, float(sc)))
    score_df = pd.DataFrame(all_scores, columns=['eid','side','risk_eye'])

    pivot = score_df.pivot_table(index='eid', columns='side', values='risk_eye', aggfunc='mean')
    pivot = pivot.reindex([str(x) for x in eids])

    if args.fusion == 'mean':
        fused = pivot.mean(axis=1, skipna=True)
    else:
        fused = pivot.max(axis=1, skipna=True)

    out = pd.DataFrame({
        'eid': [str(x) for x in eids],
        'risk_left': pivot.get('left'),
        'risk_right': pivot.get('right'),
        f'risk_{args.fusion}': fused.values,
        'time': times,
        'event': events
    })

    cidx = concordance_index(out['time'].values, -out[f'risk_{args.fusion}'].values, out['event'].values)
    print(f'Fused ({args.fusion}) C-index: {cidx:.4f}')
    for t in args.eval_at_years:
        auc_t = time_dependent_auc(out['time'].values, out['event'].values, -out[f'risk_{args.fusion}'].values, t)
        print(f'Fused ({args.fusion}) time-dependent AUC at {t} years: {auc_t:.4f}')

    os.makedirs(os.path.dirname(args.out_scores), exist_ok=True)
    out.to_csv(args.out_scores, index=False)
    print(f'Saved fused scores to {args.out_scores}')

if __name__ == '__main__':
    main()
