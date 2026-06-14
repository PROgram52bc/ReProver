#!/bin/bash

# Exit on error
set -e

# 1. Download APRIL dataset
echo "Step 1: Downloading APRIL dataset..."
python scripts/download_april.py

# 2. Train the repair model using APRIL data
# Defaulting to byt5-small for T5 training
echo "Step 2: Training the repair model on APRIL..."
python -m generation.train_april_model \
    --data-dir data/april \
    --model-name google/byt5-small \
    --epochs 5 \
    --batch-size 4 \
    --log-dir logs/april_repair_train

# 3. (Optional) Evaluate the new repair model
# Replace the paths as needed
echo "Step 3: Evaluating the new APRIL-trained repair model..."
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random \
    --split val \
    --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
    --repair-ckpt-path logs/april_repair_train/final_april_model \
    --num-theorems 100 \
    --verbose
