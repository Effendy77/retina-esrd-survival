#!/usr/bin/env bash
echo "Resolved IMG_DIR: ${IMG_DIR}"; echo "Resolved NUM_WORKERS: ${NUM_WORKERS}"
# DEFAULT_IMG_DIR_PLACEHOLDER
: "${IMG_DIR:=/home/$USER/data/retina_images}"
: "${NUM_WORKERS:=4}"

set -euo pipefail

# ----------------------------
# Configuration — edit as needed
# ----------------------------
REPO="${REPO:-$HOME/projects/retina-esrd-survival}"
IMG_DIR="${IMG_DIR:-/mnt/d/DATA/main_data/bilateralclean/}"                     # <<-- REPLACE this with your actual image root path
BATCH="${BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
PENALIZERS=(${PENALIZERS:-0.01 0.1 1.0})
export PENALIZERS="${PENALIZERS:-0.01,0.1,1.0}"  # lifelines CoxPHFitter penalizers to try
OUT_BASE="${OUT_BASE:-outputs/cv_pipeline}"
FINAL_OUT="${FINAL_OUT:-outputs/final_cox}"
CHOICE="${CHOICE:-0.01}"                    # default chosen penalizer (overridden by STEP 3 result if you change)
# ----------------------------

echo "=== CV pipeline starting ==="
echo "Repo: $REPO"
cd "$REPO"
echo "Git commit: $(git rev-parse --verify HEAD)"
echo "Python: $(python -V 2>&1)"
echo "Torch: $(python -c 'import torch,sys; print(getattr(torch, \"__version__\", \"N/A\"))' 2>/dev/null || echo N/A)"

# Basic checks
if [ ! -d "$REPO" ]; then
  echo "Error: repo directory not found: $REPO" >&2
  exit 1
fi

if [ ! -f data/metadata/cv/fold0_train.csv ]; then
  echo "Error: expected CV CSVs under data/metadata/cv/ but fold0_train.csv missing" >&2
  ls -la data/metadata/cv || true
  exit 1
fi

if [ ! -d "$IMG_DIR" ]; then
  echo "Warning: IMG_DIR does not exist: $IMG_DIR"
  echo "If your CSV contains absolute WSL paths, ensure IMG_DIR is set correctly before running."
fi

mkdir -p "$OUT_BASE"
mkdir -p "$FINAL_OUT"

echo
echo "=== STEP 1: Extract embeddings for each fold train/val and holdout ==="
FILES=( \
  data/metadata/cv/fold0_train.csv data/metadata/cv/fold0_val.csv \
  data/metadata/cv/fold1_train.csv data/metadata/cv/fold1_val.csv \
  data/metadata/cv/fold2_train.csv data/metadata/cv/fold2_val.csv \
  data/metadata/cv/fold3_train.csv data/metadata/cv/fold3_val.csv \
  data/metadata/cv/fold4_train.csv data/metadata/cv/fold4_val.csv \
  data/metadata/cv/holdout_test.csv )

for CSV in "${FILES[@]}"; do
  stem=$(basename "$CSV" .csv)
  out_npz="$OUT_BASE/emb_${stem}.npz"
  echo "-> Extracting: $CSV -> $out_npz"
  export CSV OUT_NPZ IMG_DIR BATCH NUM_WORKERS SEED
  OUT_NPZ="$out_npz" python - <<'PY'
import os, numpy as np, torch
from torch.utils.data import DataLoader
# Local import of project scripts
from scripts.train_survival_from_deepdkd import DeepDKD_Survival, SurvivalDataset

CSV = os.environ['CSV']
OUT_NPZ = os.environ['OUT_NPZ']
IMG_DIR = os.environ['IMG_DIR']
BATCH = int(os.environ['BATCH'])
NUM_WORKERS = int(os.environ['NUM_WORKERS'])
SEED = int(os.environ['SEED'])

torch.manual_seed(SEED)
model = DeepDKD_Survival(None, freeze_encoder=True)
model.eval()

ds = SurvivalDataset(CSV, img_dir=IMG_DIR)
loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

embs, durs, evs = [], [], []
with torch.no_grad():
    for xb, d, e in loader:
        _, emb = model(xb)
        embs.append(emb.cpu().numpy()); durs.append(d.numpy()); evs.append(e.numpy())

if len(embs) == 0:
    raise RuntimeError(f"No batches produced for {CSV}; check paths and CSV content")

embs = np.concatenate(embs)
durs = np.concatenate(durs)
evs = np.concatenate(evs)

np.savez(OUT_NPZ, emb=embs, dur=durs, ev=evs)
print(f"Saved {OUT_NPZ}; shape: {embs.shape}")
PY
done

echo
echo "=== STEP 2: Sanity — event counts per split ==="
python - <<'PY'
import glob, numpy as np
files = sorted(glob.glob("outputs/cv_pipeline/emb_*.npz"))
for f in files:
    d = np.load(f)
    ev = d['ev']
    print(f, "rows", len(ev), "events", int(ev.sum()), "censored", int((len(ev)-int(ev.sum()))))
PY

echo
echo "=== STEP 3: Penalizer selection (CV across fold0..4) ==="
export PENALIZERS
python - <<'PY'
import os, numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

PEN = [float(x) for x in os.environ['PENALIZERS'].split()]
folds = [(f"outputs/cv_pipeline/emb_fold{i}_train.npz", f"outputs/cv_pipeline/emb_fold{i}_val.npz") for i in range(5)]

best_pen = None
best_mean = -1.0
for pen in PEN:
    cis = []
    for tr, val in folds:
        dtr = np.load(tr); dval = np.load(val)
        Xtr, d_tr, e_tr = dtr['emb'], dtr['dur'], dtr['ev']
        Xval, d_val, e_val = dval['emb'], dval['dur'], dval['ev']
        df_tr = pd.DataFrame(Xtr); df_tr['duration']=d_tr; df_tr['event']=e_tr
        df_val = pd.DataFrame(Xval); df_val['duration']=d_val; df_val['event']=e_val
        cph = CoxPHFitter(penalizer=pen)
        cph.fit(df_tr, duration_col='duration', event_col='event', show_progress=False)
        preds = cph.predict_partial_hazard(df_val).values.ravel()
        ci = concordance_index(df_val['duration'].values, -preds, df_val['event'].values)
        cis.append(ci)
    mean_ci = float(np.mean(cis)); std_ci = float(np.std(cis))
    print(f"penalizer={pen}: mean C-index = {mean_ci:.4f} ± {std_ci:.4f}")
    if mean_ci > best_mean:
        best_mean = mean_ci; best_pen = pen

print("\nSelected penalizer by CV mean-best:", best_pen, "with mean CI", best_mean)
# write pick to OUT_BASE/selected_pen.txt
open("outputs/cv_pipeline/selected_pen.txt","w").write(f"{best_pen}")
PY

SEL_PEN=$(cat outputs/cv_pipeline/selected_pen.txt)
echo "Selected penalizer saved to outputs/cv_pipeline/selected_pen.txt -> $SEL_PEN"
echo

echo "=== STEP 4: Pool fold train embeddings, fit final Cox, evaluate on holdout ==="
export CHOICE="${SEL_PEN:-$CHOICE}"
python - <<'PY'
import glob, numpy as np, pandas as pd, joblib, os
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

files = sorted(glob.glob("outputs/cv_pipeline/emb_fold*_train.npz"))
if not files:
    raise SystemExit("No pooled train embeddings found in outputs/cv_pipeline")

embs_list, durs_list, evs_list = [], [], []
for f in files:
    d = np.load(f)
    embs_list.append(d['emb']); durs_list.append(d['dur']); evs_list.append(d['ev'])
X = np.concatenate(embs_list); dur = np.concatenate(durs_list); ev = np.concatenate(evs_list)
df = pd.DataFrame(X); df['duration']=dur; df['event']=ev

pen = float(os.environ.get('CHOICE', 0.01))
cph = CoxPHFitter(penalizer=pen)
cph.fit(df, duration_col='duration', event_col='event', show_progress=True)

os.makedirs("outputs/final_cox", exist_ok=True)
joblib.dump(cph, "outputs/final_cox/cox_final.joblib")
print("Saved final Cox to outputs/final_cox/cox_final.joblib")
print("Apparent concordance (training):", getattr(cph, "concordance_index_", None))

# Evaluate on holdout
dtest = np.load("outputs/cv_pipeline/emb_holdout_test.npz")
df_test = pd.DataFrame(dtest['emb']); df_test['duration']=dtest['dur']; df_test['event']=dtest['ev']
preds = cph.predict_partial_hazard(df_test).values.ravel()
from lifelines.utils import concordance_index
ci = concordance_index(df_test['duration'].values, -preds, df_test['event'].values)
print("Holdout C-index:", ci)
# save holdout preds
os.makedirs("outputs/final_cox", exist_ok=True)
import numpy as _np
_np.savez("outputs/final_cox/holdout_preds.npz", preds=preds, durations=dtest['dur'], events=dtest['ev'])
PY

echo
echo "=== STEP 5: Diagnostics: metadata and crude calibration (KM quintiles) ==="
python - <<'PY'
import json, subprocess, os, numpy as np, pandas as pd
from lifelines import KaplanMeierFitter
import joblib

meta = {
  "git": subprocess.check_output(["git","rev-parse","--verify","HEAD"]).decode().strip(),
  "seed": int(os.environ.get('SEED',42)),
  "penalizer_choice": float(open("outputs/cv_pipeline/selected_pen.txt").read().strip())
}
os.makedirs("outputs/final_cox", exist_ok=True)
open("outputs/final_cox/metadata.json","w").write(json.dumps(meta, indent=2))

cph = joblib.load("outputs/final_cox/cox_final.joblib")
dtest = np.load("outputs/cv_pipeline/emb_holdout_test.npz")
df_test = pd.DataFrame(dtest['emb']); df_test['duration']=dtest['dur']; df_test['event']=dtest['ev']
preds = cph.predict_partial_hazard(df_test).values.ravel()
df_test['risk']=preds
# create quintiles, drop duplicates if any
df_test['q'] = pd.qcut(df_test['risk'], 5, labels=False, duplicates='drop')
km = KaplanMeierFitter()
for q in sorted(df_test['q'].unique()):
    sub = df_test[df_test['q']==q]
    km.fit(sub['duration'], sub['event'])
    # print last survival point (most extreme time)
    sf = km.survival_function_
    last = float(sf.iloc[-1,0]) if not sf.empty else float('nan')
    print("quintile", int(q), "n", len(sub), "KM tail survival:", round(last,4))
PY

echo
echo "Pipeline finished. Artifacts:"
ls -la outputs/cv_pipeline || true
ls -la outputs/final_cox || true
echo "=== END ==="