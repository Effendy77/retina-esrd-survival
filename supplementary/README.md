# Supplementary Materials: RetBio-Kidney-DL Pipeline

This folder contains reproducible materials for the RetBio-Kidney-DL pipeline — a deep learning framework for retinal biomarker-based kidney outcome modeling. These materials support survival analysis experiments and will extend to binary classification and eGFR prediction in future work.

## Contents
- `scripts/`: Core pipeline scripts for embedding extraction and CoxPH modeling
- `outputs/`: Metadata and metrics from full pipeline run
- `figures/`: KM curves and penalizer sweep plots (to be added)
- `environment.yml`: Conda environment for reproducibility
- `run_full_pipeline_20251103.log`: Full log for provenance

## Reproducibility
To reproduce the full pipeline:
```bash
conda env create -f environment.yml
conda activate retina-surv
IMG_DIR=~/data/retina_images NUM_WORKERS=4 ./scripts/run_cv_pipeline.sh

Citation
If using this pipeline, please cite: Effendy et al., 2025. RetBio-Kidney-DL: Deep Learning for Retinal Biomarker-Based Kidney Outcome Prediction.

📊 Embedding Summary
To generate a CSV summary of all .npz files (embedding shape, event/censor counts), run:
python - <<'PY'
import numpy as np, glob, pandas as pd
rows = []
for f in sorted(glob.glob("outputs/cv_pipeline/*.npz")):
    d = np.load(f)
    rows.append({
        "file": f,
        "emb_shape": d["emb"].shape,
        "events": int(d["ev"].sum()),
        "censored": int((1 - d["ev"]).sum())
    })
pd.DataFrame(rows).to_csv("supplementary/outputs/emb_summary.csv", index=False)
print("Saved supplementary/outputs/emb_summary.csv")
PY
Notes
This supplement supports the survival modeling component of RetBio-Kidney-DL.

Future versions will include classification metrics, eGFR prediction modules, and additional figures.


---

## ✅ Next Steps

1. Save this as `supplementary/README.md`:
```bash
nano supplementary/README.md

