#!/bin/bash
# Reproducible baseline-vs-repair comparison on the Lean Workbook dataset.
#
# Runs prover/evaluate.py twice over the *same* theorem set (same split,
# same --num-theorems, same --num-sampled-tactics, same non-retrieval
# checkpoint) -- once without a repair model, once with -- and reports
# Pass@1 for each plus which theorems flipped. See LEAN_WORKBOOK_SETUP_NOTES.md
# for why NUM_PROCS=4 and env.sh are required on this cluster.
#
# Prerequisites (one-time): data/lean_workbook_reprover/<SPLIT>.json must
# exist -- run `python scripts/setup_lean_workbook.py` first if it doesn't.
#
# Usage: ./scripts/compare_repair_lean_workbook.sh
# Override any of the variables below via env, e.g.:
#   NUM_THEOREMS=100 REPAIR_COUNT=3 ./scripts/compare_repair_lean_workbook.sh

set -uo pipefail
cd "$(dirname "$0")/.."

if [ -f env.sh ]; then
    source env.sh
else
    echo "WARNING: env.sh not found -- GITHUB_ACCESS_TOKEN/HF_ACCESS_TOKEN may be missing." >&2
fi

# See LEAN_WORKBOOK_SETUP_NOTES.md #3: default NUM_PROCS (= cpu_count(), often
# 24+) causes a thundering herd against a shared/networked scratch filesystem
# during LeanDojo's extraction step. Override with NUM_PROCS=<n> if your
# filesystem can sustain more concurrent readers.
export NUM_PROCS="${NUM_PROCS:-4}"

DATA_PATH="${DATA_PATH:-data/lean_workbook_reprover}"
SPLIT="${SPLIT:-val}"
GEN_CKPT="${GEN_CKPT:-kaiyuy/leandojo-lean4-tacgen-byt5-small}"
REPAIR_CKPT="${REPAIR_CKPT:-uw-math-ai/gAPRIL-wo-exp}"
NUM_TACTICS="${NUM_TACTICS:-5}"
NUM_THEOREMS="${NUM_THEOREMS:-50}"
REPAIR_COUNT="${REPAIR_COUNT:-2}"
TIMEOUT="${TIMEOUT:-600}"

if [ ! -f "$DATA_PATH/$SPLIT.json" ]; then
    echo "ERROR: $DATA_PATH/$SPLIT.json not found." >&2
    echo "Run 'python scripts/setup_lean_workbook.py' first (see README Lean Workbook section)." >&2
    exit 1
fi

OUT_DIR="logs/experiments/lean_workbook_repair_compare"
mkdir -p "$OUT_DIR"

EXP_ID="lw_${SPLIT}_thm${NUM_THEOREMS}_tac${NUM_TACTICS}"
LOG_BASELINE="$OUT_DIR/${EXP_ID}_baseline.log"
SUMMARY_BASELINE="$OUT_DIR/${EXP_ID}_baseline.jsonl"
LOG_REPAIR="$OUT_DIR/${EXP_ID}_repair_c${REPAIR_COUNT}.log"
SUMMARY_REPAIR="$OUT_DIR/${EXP_ID}_repair_c${REPAIR_COUNT}.jsonl"

echo "========================================================="
echo "Lean Workbook baseline-vs-repair comparison"
echo "  data=$DATA_PATH split=$SPLIT theorems=$NUM_THEOREMS tactics=$NUM_TACTICS"
echo "  repair_ckpt=$REPAIR_CKPT repair_count=$REPAIR_COUNT NUM_PROCS=$NUM_PROCS"
echo "========================================================="

echo
echo "--- [1/2] BASELINE (no repair) ---"
python prover/evaluate.py \
    --data-path "$DATA_PATH" \
    --dataset lean_workbook \
    --split "$SPLIT" \
    --gen_ckpt_path "$GEN_CKPT" \
    --num-sampled-tactics "$NUM_TACTICS" \
    --num-theorems "$NUM_THEOREMS" \
    --timeout "$TIMEOUT" \
    --summary-jsonl "$SUMMARY_BASELINE" \
    --log-file "$LOG_BASELINE" \
    --exp-id "${EXP_ID}_baseline"
BASELINE_STATUS=$?

echo
echo "--- [2/2] REPAIR (repair-count=$REPAIR_COUNT) ---"
python prover/evaluate.py \
    --data-path "$DATA_PATH" \
    --dataset lean_workbook \
    --split "$SPLIT" \
    --gen_ckpt_path "$GEN_CKPT" \
    --repair-ckpt-path "$REPAIR_CKPT" \
    --repair-count "$REPAIR_COUNT" \
    --num-sampled-tactics "$NUM_TACTICS" \
    --num-theorems "$NUM_THEOREMS" \
    --timeout "$TIMEOUT" \
    --summary-jsonl "$SUMMARY_REPAIR" \
    --log-file "$LOG_REPAIR" \
    --exp-id "${EXP_ID}_repair_c${REPAIR_COUNT}"
REPAIR_STATUS=$?

if [ $BASELINE_STATUS -ne 0 ]; then
    echo "WARNING: baseline run exited with status $BASELINE_STATUS -- see $LOG_BASELINE" >&2
fi
if [ $REPAIR_STATUS -ne 0 ]; then
    echo "WARNING: repair run exited with status $REPAIR_STATUS -- see $LOG_REPAIR" >&2
fi

echo
echo "========================================================="
echo "COMPARISON (from $SUMMARY_BASELINE vs $SUMMARY_REPAIR)"
echo "========================================================="
python scripts/compare_repair_results.py "$SUMMARY_BASELINE" "$SUMMARY_REPAIR"

echo
echo "Full logs:"
echo "  baseline: $LOG_BASELINE"
echo "  repair:   $LOG_REPAIR"
