#!/bin/bash

# Configuration
DATA_PATH="data/leandojo_benchmark_4/random/"
GEN_CKPT="kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
RET_CKPT="kaiyuy/leandojo-lean4-retriever-byt5-small"
INDEXED_CORPUS="data/leandojo_benchmark_4/indexed_corpus.pkl"
WALL_TIMEOUT=100000  # Fair cutoff time for all runs in an experiment

# Experiment sets (modify these arrays to scale your experiments)
THEOREMS_LIST=(2000)
TACTICS_LIST=(1 3 5 8)
REPAIR_COUNTS_LIST=(1 2 3) # Number of recursive repair attempts to try

mkdir -p logs/experiments

for thm in "${THEOREMS_LIST[@]}"; do
    for tac in "${TACTICS_LIST[@]}"; do
        
        EXP_BASE_ID="thm${thm}_tac${tac}"
        LOGS_TO_COMPARE=()
        
        echo "========================================================="
        echo "EXPERIMENT: Theorems=$thm | Tactics=$tac | Timeout=${WALL_TIMEOUT}s"
        echo "========================================================="

        # 1. Run WITHOUT repair model (Baseline)
        LOG_NO_REPAIR="logs/experiments/${EXP_BASE_ID}_norepair.log"
        echo "Running [BASELINE - NO REPAIR]..."
        python prover/evaluate.py \
            --data-path "$DATA_PATH" \
            --gen_ckpt_path "$GEN_CKPT" \
            --ret_ckpt_path "$RET_CKPT" \
            --indexed-corpus-path "$INDEXED_CORPUS" \
            --num-sampled-tactics "$tac" \
            --num-theorems "$thm" \
            --wall-timeout "$WALL_TIMEOUT" \
            --log-file "$LOG_NO_REPAIR" \
            --exp-id "${EXP_BASE_ID}_norepair" \
            --save-results
        
        LOGS_TO_COMPARE+=("$LOG_NO_REPAIR")

        # 2. Run WITH repair model for each count in the list
        for rep in "${REPAIR_COUNTS_LIST[@]}"; do
            LOG_REPAIR="logs/experiments/${EXP_BASE_ID}_repair_c${rep}.log"
            echo "Running [REPAIR - COUNT $rep]..."
            python prover/evaluate.py \
                --data-path "$DATA_PATH" \
                --gen_ckpt_path "$GEN_CKPT" \
                --ret_ckpt_path "$RET_CKPT" \
                --indexed-corpus-path "$INDEXED_CORPUS" \
                --num-sampled-tactics "$tac" \
                --num-theorems "$thm" \
                --repair-ckpt-path "uw-math-ai/gAPRIL-wo-exp" \
                --repair-count "$rep" \
                --wall-timeout "$WALL_TIMEOUT" \
                --log-file "$LOG_REPAIR" \
                --exp-id "${EXP_BASE_ID}_repair_c${rep}" \
                --save-results
            
            LOGS_TO_COMPARE+=("$LOG_REPAIR")
        done

        # 3. Compare all variants for this (thm, tac) configuration
        echo "Generating multi-way comparison for configuration: $EXP_BASE_ID"
        python compare_outcomes.py "${LOGS_TO_COMPARE[@]}" --output "logs/experiments/${EXP_BASE_ID}_comparison.csv"
        
        echo -e "\n\n"
    done
done

echo "All experiments completed. Results are in logs/experiments/"
