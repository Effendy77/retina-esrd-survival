#!/bin/bash
# run_full_experiment.sh
# Full 5-fold Retina-ESRD Survival Experiment using train_survival.py

# -------- CONFIG --------
IMG_DIR="/home/fendy77/projects/retina-esrd-survival/data/images"
DATA_DIR="/home/fendy77/projects/retina-esrd-survival/data/metadata/cv"
OUTPUT_DIR="/home/fendy77/projects/retina-esrd-survival/outputs"
CHECKPOINT="/home/fendy77/projects/retina-esrd-survival/checkpoints/retfound_encoder.bin"
SCRIPT_DIR="/home/fendy77/projects/retina-esrd-survival/src/train"

EPOCHS=50
BATCH_SIZE=16
LR=1e-4
PATIENCE=8
IMG_SIZE=224
SEED=77
NUM_WORKERS=4

# -------- TRAIN 5-Fold CV --------
echo "Starting 5-fold cross-validation survival training..."
for FOLD in 0 1 2 3 4; do
    echo "Training fold ${FOLD}..."
    python $SCRIPT_DIR/train_survival.py \
        --train_csv $DATA_DIR/fold${FOLD}_train.csv \
        --val_csv   $DATA_DIR/fold${FOLD}_val.csv \
        --img_dir   $IMG_DIR \
        --weights_path $CHECKPOINT \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --patience $PATIENCE \
        --img_size $IMG_SIZE \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --out $OUTPUT_DIR/surv_vitl16_f${FOLD}
done
echo "5-fold training complete."

# -------- EVALUATION ON HELD-OUT TEST SET --------
TEST_CSV="$DATA_DIR/holdout_test.csv"
echo "Evaluating held-out test set..."
for FOLD in 0 1 2 3 4; do
    echo "Evaluating fold ${FOLD}..."
    python /home/fendy77/projects/retina-esrd-survival/scripts/eval_holdout.py \
        --test_csv $TEST_CSV \
        --img_dir $IMG_DIR \
        --checkpoint $OUTPUT_DIR/surv_vitl16_f${FOLD}/best_cindex.pth \
        --outdir $OUTPUT_DIR/surv_vitl16_f${FOLD}/test_holdout \
        --cindex_flip
done
echo "Held-out evaluation complete."
echo "Full survival experiment finished successfully!"
