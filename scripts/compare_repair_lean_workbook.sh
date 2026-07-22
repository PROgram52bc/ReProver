#!/bin/bash

# Configuration
DATA_PATH="data/lean_workbook_reprover"
DATASET="lean_workbook"
SPLIT="val"
GEN_CKPT="kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
RET_CKPT="kaiyuy/leandojo-lean4-retriever-byt5-small"
INDEXED_CORPUS="data/leandojo_benchmark_4/indexed_corpus.pkl"
GLOBAL_WALL_TIMEOUT=100000  # Fair cutoff time for all runs in an experiment
LOCAL_TIMEOUT=600
UNBOUNDED_RUNTIME=true # Set to true to run until exhaustion (ignores GLOBAL_WALL_TIMEOUT)

# Construct timeout argument
GLOBAL_TIMEOUT_ARG="--global-wall-timeout $GLOBAL_WALL_TIMEOUT"
if [ "$UNBOUNDED_RUNTIME" = true ]; then
    GLOBAL_TIMEOUT_ARG=""
    echo "Running in UNBOUNDED mode (no wall timeout)"
fi

# Experiment sets (modify these arrays to scale your experiments)
THEOREMS_LIST=(90)
TACTICS_LIST=(3) # (3 5 8)
REPAIR_COUNTS_LIST=(3) # (1 2 3) # Number of recursive repair attempts to try

mkdir -p logs/experiments

for thm in "${THEOREMS_LIST[@]}"; do
    for tac in "${TACTICS_LIST[@]}"; do
        
        EXP_BASE_ID="thm${thm}_tac${tac}"
        LOGS_TO_COMPARE=()
        SUMMARIES_TO_ANALYZE=()
        
        DISPLAY_TIMEOUT="${GLOBAL_WALL_TIMEOUT}s"
        if [ "$UNBOUNDED_RUNTIME" = true ]; then
            DISPLAY_TIMEOUT="Unbounded"
        fi

        echo "========================================================="
        echo "EXPERIMENT: Theorems=$thm | Tactics=$tac | Timeout=${DISPLAY_TIMEOUT}"
        echo "========================================================="

        # 1. Run WITHOUT repair model (Baseline)
        LOG_NO_REPAIR="logs/experiments/${EXP_BASE_ID}_norepair.log"
        SUMMARY_NO_REPAIR="logs/experiments/${EXP_BASE_ID}_norepair.jsonl"
        SUMMARIES_TO_ANALYZE+=("$SUMMARY_NO_REPAIR")
        echo "Running [BASELINE - NO REPAIR]..."
        # python -m prover.evaluate \
        #     --data-path "$DATA_PATH" \
        #     --dataset "$DATASET" \
        #     --split "$SPLIT" \
        #     --gen_ckpt_path "$GEN_CKPT" \
        #     --ret_ckpt_path "$RET_CKPT" \
        #     --num-sampled-tactics "$tac" \
        #     --num-theorems "$thm" \
        #     --timeout "$LOCAL_TIMEOUT" \
        #     --timeout-accounting effective \
        #     $GLOBAL_TIMEOUT_ARG \
        #     --summary-jsonl "$SUMMARY_NO_REPAIR" \
        #     --log-file "$LOG_NO_REPAIR" \
        #     --exp-id "${EXP_BASE_ID}_norepair" \
        #     --save-results
        
        # LOGS_TO_COMPARE+=("$LOG_NO_REPAIR")

        # 2. Run WITH repair model for each count in the list
        for rep in "${REPAIR_COUNTS_LIST[@]}"; do
            LOG_REPAIR="logs/experiments/${EXP_BASE_ID}_repair_c${rep}.log"
            SUMMARY_REPAIR="logs/experiments/${EXP_BASE_ID}_repair_c${rep}.jsonl"
            SUMMARIES_TO_ANALYZE+=("$SUMMARY_REPAIR")
            echo "Running [REPAIR - COUNT $rep]..."
            python -m prover.evaluate \
                --data-path "$DATA_PATH" \
                --dataset "$DATASET" \
                --split "$SPLIT" \
                --gen_ckpt_path "$GEN_CKPT" \
                --ret_ckpt_path "$RET_CKPT" \
                --num-sampled-tactics "$tac" \
                --num-theorems "$thm" \
                --repair-ckpt-path "uw-math-ai/gAPRIL-wo-exp" \
                --repair-count "$rep" \
                --timeout "$LOCAL_TIMEOUT" \
                --timeout-accounting effective \
                $GLOBAL_TIMEOUT_ARG \
                --summary-jsonl "$SUMMARY_REPAIR" \
                --log-file "$LOG_REPAIR" \
                --exp-id "${EXP_BASE_ID}_repair_c${rep}" \
                --save-results
            
            LOGS_TO_COMPARE+=("$LOG_REPAIR")
        done

        # 3. Compare all variants for this (thm, tac) configuration
        echo "Generating multi-way comparison for configuration: $EXP_BASE_ID"
        python compare_outcomes.py "${LOGS_TO_COMPARE[@]}" --output "logs/experiments/${EXP_BASE_ID}_comparison.csv"
        
        python -m scripts.analyze_timeout_variants \
            "${SUMMARIES_TO_ANALYZE[@]}" \
            --local-timeout "$LOCAL_TIMEOUT" \
            --global-budget "$GLOBAL_WALL_TIMEOUT" \
            --output "logs/experiments/${EXP_BASE_ID}_timeout_variants.csv"
        
        echo -e "\n\n"
    done
done

echo "All experiments completed. Results are in logs/experiments/"
