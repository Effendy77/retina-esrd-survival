
import argparse, os, json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.datasets.fundus_survival_dataset import FundusSurvivalDataset as SurvivalImageDataset
from src.model.multimodal_model import build_model
from src.utils.losses import cox_ph_loss
from src.utils.metrics import concordance_index

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['survival','classification'], default='survival')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--image_root', default='')
    ap.add_argument('--target', default='time_to_event')
    ap.add_argument('--event_col', default='event_occurred')
    ap.add_argument('--tabular_cols', default='')
    ap.add_argument('--tabular_norm', choices=['none','standardize'], default='standardize')
    ap.add_argument('--fusion', choices=['concat'], default='concat')
    ap.add_argument('--backbone', default='retfound_vit_base')
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--img_size', type=int, default=448)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--outdir', default='outputs/run')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--flip_right_eye', type=str, default='true')
    return ap.parse_args()

def str2bool(x): return str(x).lower() in ['1','true','t','yes','y']

def save_config(args):
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, 'config_snapshot.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

def collate_survival(batch):
    import torch
    imgs = torch.stack([b['image'] for b in batch], dim=0)
    times = torch.tensor([b['time'] for b in batch], dtype=torch.float32)
    events = torch.tensor([b['event'] for b in batch], dtype=torch.float32)
    tabs = None
    if batch[0]['tab'] is not None:
        tabs = torch.stack([b['tab'] for b in batch], dim=0)
    return imgs, times, events, tabs

def main():
    args = parse_args(); save_config(args)
    flip_right = str2bool(args.flip_right_eye)
    tab_cols = [c.strip() for c in args.tabular_cols.split(',') if c.strip()]
    train_df = pd.read_csv(args.train_csv); val_df = pd.read_csv(args.val_csv)

    train_ds = SurvivalImageDataset(train_df, image_root=args.image_root, img_size=args.img_size,
                                    time_col=args.target, event_col=args.event_col, tabular_cols=tab_cols,
                                    tabular_norm=args.tabular_norm, fit_stats=True, flip_right_eye=flip_right)
    val_ds   = SurvivalImageDataset(val_df, image_root=args.image_root, img_size=args.img_size,
                                    time_col=args.target, event_col=args.event_col, tabular_cols=tab_cols,
                                    tabular_norm=args.tabular_norm, stats=train_ds.stats_, flip_right_eye=flip_right)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_survival)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_survival)

    model = build_model(backbone=args.backbone, checkpoint=args.checkpoint,
                        tabular_dim=len(tab_cols), fusion=args.fusion)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_c = -1.0
    for epoch in range(1, args.epochs+1):
        model.train(); tr_loss = 0.0
        for imgs, times, events, tabs in train_loader:
            imgs, times, events = imgs.to(device), times.to(device), events.to(device); tabs = tabs.to(device) if tabs is not None else None
            risk = model(imgs, tabs); loss = cox_ph_loss(risk, times, events)
            optim.zero_grad(); loss.backward(); optim.step(); tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        model.eval(); all_r, all_t, all_e = [], [], []
        with torch.no_grad():
            for imgs, times, events, tabs in val_loader:
                imgs, times, events = imgs.to(device), times.to(device), events.to(device); tabs = tabs.to(device) if tabs is not None else None
                risk = model(imgs, tabs)
                all_r.append(risk.detach().cpu().flatten()); all_t.append(times.cpu()); all_e.append(events.cpu())
        import torch as _torch
        all_r = _torch.cat(all_r).numpy(); all_t = _torch.cat(all_t).numpy(); all_e = _torch.cat(all_e).numpy()
        from src.utils.metrics import concordance_index
        cidx = concordance_index(all_t, -all_r, all_e)
        print(f'Epoch {epoch:03d} | train_loss {tr_loss:.4f} | val_c-index {cidx:.4f}')
        if cidx > best_c:
            best_c = cidx; os.makedirs(args.outdir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best.pt'))
    print(f'Best val C-index: {best_c:.4f}')

if __name__ == '__main__':
    main()
