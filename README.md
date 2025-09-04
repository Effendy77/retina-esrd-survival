# Retina-ESRD-Survival (RETFound + DeepSurv)

**Target journal:** *EHJ Digital Health*

This repo provides a fully reproducible pipeline to predict **time to end-stage renal disease (ESRD)** from **retinal fundus images** using a **RETFound** backbone with a **DeepSurv-style survival head**, compared against the **Kidney Failure Risk Equation (KFRE)** and enhanced baselines. It includes training, evaluation, calibration, decision curve analysis, reclassification metrics, and interpretable Grad-CAM casebooks.

## Highlights
- Vision Transformer (RETFound) embeddings + Cox/DeepSurv survival head
- UK Biobank: left-eye images, linked ESRD outcomes; comparison vs KFRE (4-/8-var)
- Calibration (risk & time), IBS, C-index, DCA, NRI/IDI
- Publication-ready figures and scripts
- Prospective validation template for **LHCH Hypertension Clinic**

## Quickstart
```bash
conda env create -f environment.yml
conda activate retina-esrd
python -m pip install -e .
```

### Data layout
```
data/
  images_left/                # left-eye images (e.g., eid_21015_0.0.png)
  metadata/train.csv          # eid,image_path,time,event,age,sex,...
  metadata/val.csv
  metadata/test.csv
  survival_labels.csv         # one row per eid: time,event
```

### Train (5-fold CV)
```bash
bash scripts/02_train_cv.sh
```

### Evaluate + Figures
```bash
bash scripts/03_eval_cv.sh
bash scripts/04_make_figures.sh
```

### Compare vs KFRE
```bash
python scripts/05_kfre_baselines.py --cfg configs/kfre.yaml
python -m src.eval.kfre_compare --pred results/cv_survival_preds.csv --kfre results/kfre/kfre_scores.csv
```

## Reproducibility
- `environment.yml` (exact versions)
- fixed seeds, deterministic cuDNN
- scripted figure generation under `results/figures`

## Citation
See `CITATION.cff`.
