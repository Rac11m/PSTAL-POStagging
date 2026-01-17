#!/bin/bash

# ==========================
# Fixed files
# ==========================
TRAIN_FILE="../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train"
DEV_FILE="../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev"

# ==========================
# Grid of hyperparameters
# ==========================
EPOCHS_LIST=(20 30 35 40)
MAX_LEN_LIST=(20 24 30 32 40)
BATCH_SIZE_LIST=(32 64)

# ==========================
# Loop over configurations
# ==========================
for EPOCHS in "${EPOCHS_LIST[@]}"; do
  for MAX_LEN in "${MAX_LEN_LIST[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do

      echo "======================================"
      echo "Training: epochs=${EPOCHS}, max_len=${MAX_LEN}, batch_size=${BATCH_SIZE}"
      echo "======================================"

      python train_postag.py \
        -t "$TRAIN_FILE" \
        -d "$DEV_FILE" \
        -e "$EPOCHS" \
        -l "$MAX_LEN" \
        -bs "$BATCH_SIZE"

    done
  done
done
