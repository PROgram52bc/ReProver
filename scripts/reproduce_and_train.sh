#!/bin/bash

# Exit on error
set -e

# 1. Harvest errors using the existing generator
# We set FAILED_TACTIC_LOG to capture errors during evaluation
export FAILED_TACTIC_LOG="logs/failed_tactics.jsonl"
mkdir -p logs

echo "Step 1: Harvesting failed tactics..."
# Adjust parameters (e.g., --num-theorems) as needed for a quick run or full harvest
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random \
    --split val \
    --gen_ckpt_path hf_models/kaiyuy/leandojo-lean4-tacgen-byt5-small \
    --num-theorems 100 \
    --verbose

# 2. Curate the dataset
echo "Step 2: Curating the dataset..."
python scripts/curate_dataset.py \
    --log-file logs/failed_tactics.jsonl \
    --leandojo-dataset-path data/leandojo_benchmark_4/random/val.json \
    --output-file data/curated_failed_tactics.json

# 3. Train the repair model
echo "Step 3: Training the repair model..."
python -m generation.train_repair_model \
    --dataset-path data/curated_failed_tactics.json \
    --epochs 5 \
    --batch-size 4 \
    --log-dir logs/repair_model_train

# 4. Evaluate the new error-guided pipeline
echo "Step 4: Evaluating the error-guided pipeline..."
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random \
    --split val \
    --gen_ckpt_path hf_models/kaiyuy/leandojo-lean4-tacgen-byt5-small \
    --repair-ckpt-path logs/repair_model_train/final_repair_model \
    --num-theorems 100 \
    --verbose
