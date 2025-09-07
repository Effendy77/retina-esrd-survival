import argparse, os, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)
ap.add_argument("--out_csv", required=True)
ap.add_argument("--eid_col", default="eid")
ap.add_argument("--event_col", default="event")
ap.add_argument("--time_col", default=None)
ap.add_argument("--test_size", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=77)
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
def find(cands):
    for c in cands:
        if c in df.columns: return c
    return None

eid_col   = args.eid_col   if args.eid_col   in df.columns else find(["eid","EID","participant_id"])
event_col = args.event_col if args.event_col in df.columns else find(["event","event_occurred","status","label","esrd_event"])
if eid_col is None or event_col is None:
    raise SystemExit(f"Missing required columns. Found: {list(df.columns)}")

df[event_col] = df[event_col].astype(int)
eid_event = (df.groupby(eid_col)[event_col].max() > 0).astype(int).reset_index()
sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
tr, te = next(sss.split(eid_event[eid_col], eid_event[event_col]))
test_eids = set(eid_event.loc[te, eid_col])
holdout = df[df[eid_col].isin(test_eids)].reset_index(drop=True)
os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
holdout.to_csv(args.out_csv, index=False)
print(f"Holdout: {holdout[eid_col].nunique()} eids | {len(holdout)} rows")
print(f"Columns used -> eid:{eid_col} event:{event_col}")
