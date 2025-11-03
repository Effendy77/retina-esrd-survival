# Supplementary Materials: DeepDKD Survival Pipeline

This repository contains reproducible materials for the DeepDKD survival analysis pipeline, including image-based embedding extraction, CoxPH modeling, and cross-validation.

## Contents
- `scripts/`: Core pipeline scripts and model wrappers
- `outputs/`: Metadata and metrics from full pipeline run
- `figures/`: KM curves and penalizer sweep plots
- `environment.yml`: Conda environment for reproducibility
- `run_full_pipeline_YYYYMMDD.log`: Full log for provenance

## Pipeline Overview
1. **Image preprocessing**: ResNet50 encoder applied to 40k retinal images
2. **Embedding extraction**: 512-dim features saved per fold
3. **Survival modeling**: CoxPH with penalizer sweep across folds
4. **Holdout evaluation**: Concordance index and KM curve

## Reproducibility
To reproduce the full pipeline:
```bash
conda env create -f environment.yml
conda activate retina-surv
IMG_DIR=~/data/retina_images NUM_WORKERS=4 ./scripts/run_cv_pipeline.sh
