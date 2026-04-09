#!/bin/bash
set -e

MFAR_PYTHON=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
export LD_LIBRARY_PATH=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/lib:$LD_LIBRARY_PATH
export PATH=/home/xxxchen/miniforge3/bin:/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval
BASE_DIR=data

# ── Usage ────────────────────────────────────────────────────────────────────
# bash eval_negation_memory.sh <alpha> [boost] [suppress]
# Examples:
#   bash eval_negation_memory.sh 2.0 boost           → boost only
#   bash eval_negation_memory.sh 2.0 suppress         → suppress only
#   bash eval_negation_memory.sh 2.0 boost suppress   → both
#   bash eval_negation_memory.sh 5.0 boost            → boost with alpha=5.0

ALPHA=${1:-1.0}
shift || true

# Parse boost/suppress from remaining args
USE_BOOST=False
USE_SUPPRESS=False
MODES=""
for arg in "$@"; do
    case $arg in
        boost) USE_BOOST=True; MODES="${MODES}_boost" ;;
        suppress) USE_SUPPRESS=True; MODES="${MODES}_suppress" ;;
    esac
done

# Default to boost if nothing specified
if [ "$USE_BOOST" = "False" ] && [ "$USE_SUPPRESS" = "False" ]; then
    USE_BOOST=True
    MODES="_boost"
fi

RUNS_DIR=output/failure_analysis/type_b_memory/runs/alpha_${ALPHA}${MODES}
CACHE_DIR=output/failure_analysis/type_b_memory/cache

mkdir -p $RUNS_DIR

# ── Evaluate with negation memory ────────────────────────────────────────────
echo "=== alpha=${ALPHA}, boost=${USE_BOOST}, suppress=${USE_SUPPRESS} ==="
echo "=== Results → $RUNS_DIR ==="
$MFAR_PYTHON -m mfar.commands.mask_fields \
    --dataset_name prime \
    --data $BASE_DIR/prime \
    --lexical-index $BASE_DIR/prime \
    --temp-dir /tmp/mfar_temp/prime_negmem \
    --out $RUNS_DIR \
    --checkpoint_dir ./output/prime \
    --partition val \
    --additional_partition test \
    --field_names "all_dense,all_sparse,single_dense,single_sparse" \
    --dev-batch-size 32 \
    --num_gpus ${NUM_GPUS:-6} \
    --debug \
    --negation_cache "${CACHE_DIR}/qwen3_cache_val.jsonl,${CACHE_DIR}/qwen3_cache_test.jsonl" \
    --memory_alpha $ALPHA \
    --use_boost $USE_BOOST \
    --use_suppress $USE_SUPPRESS \
    2>&1 | tee $RUNS_DIR/eval.log

echo "=== Done at $(date) ==="

# ── Run comparison analysis ──────────────────────────────────────────────────
echo "=== Running comparison analysis ==="
$MFAR_PYTHON failure_analysis/type_b_memory/evaluate_memory.py \
    --baseline_dir output/prime_eval \
    --memory_dir $RUNS_DIR \
    --splits val test \
    2>&1 | tee $RUNS_DIR/comparison.log

echo "=== Analysis complete ==="
