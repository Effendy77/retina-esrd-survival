import argparse, os, pandas as pd
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--eid_col", default="eid")
ap.add_argument("--event_col", default="event")
ap.add_argument("--n_splits", type=int, default=5)
ap.add_argument("--seed", type=int, default=77)
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
for col in (args.eid_col, args.event_col):
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}. Found: {list(df.columns)}")

df[args.event_col] = df[args.event_col].astype(int)
eid_event = (df.groupby(args.eid_col)[args.event_col].max() > 0).astype(int).reset_index()

skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
os.makedirs(args.out_dir, exist_ok=True)
for fold, (tr_idx, va_idx) in enumerate(skf.split(eid_event[args.eid_col], eid_event[args.event_col])):
    tr_eids = set(eid_event.loc[tr_idx, args.eid_col])
    va_eids = set(eid_event.loc[va_idx, args.eid_col])
    tr = df[df[args.eid_col].isin(tr_eids)].reset_index(drop=True)
    va = df[df[args.eid_col].isin(va_eids)].reset_index(drop=True)
    tr.to_csv(os.path.join(args.out_dir, f"fold{fold}_train.csv"), index=False)
    va.to_csv(os.path.join(args.out_dir, f"fold{fold}_val.csv"), index=False)
    print(f"FOLD {fold}: train={tr[args.eid_col].nunique()} eids ({len(tr)} rows) | val={va[args.eid_col].nunique()} eids ({len(va)} rows)")
