#!/bin/bash
set -e

# Step 1: Run mFAR inference on FULL train split to generate .qres
# Then filter to train-dev qids for Meta-Harness evaluation.
#
# Usage: bash eval_train.sh

MFAR_PYTHON=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
export LD_LIBRARY_PATH=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/lib:$LD_LIBRARY_PATH
export PATH=/home/xxxchen/miniforge3/bin:/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval
BASE_DIR=data

mkdir -p output/contriever/prime_train_eval

echo "=== Running mFAR inference on train split at $(date) ==="
$MFAR_PYTHON -m mfar.commands.mask_fields \
    --dataset_name prime \
    --data $BASE_DIR/prime \
    --lexical-index $BASE_DIR/prime \
    --temp-dir /tmp/mfar_temp/prime_train \
    --out ./output/contriever/prime_train_eval \
    --checkpoint_dir ./output/prime \
    --partition train \
    --field_names "all_dense,all_sparse,single_dense,single_sparse" \
    --dev-batch-size 32 \
    --num_gpus 8 \
    --debug \
    2>&1 | tee output/contriever/prime_train_eval/eval.log

echo "=== Train inference done at $(date) ==="

# Filter to train-dev qids
echo "=== Filtering to train-dev qids ==="
$MFAR_PYTHON failure_analysis/type_b_memory/meta_harness/filter_qres.py \
    --qres output/contriever/prime_train_eval/final-all-0.qres \
    --keep_qids data/prime/train-dev.queries \
    --output output/contriever/prime_eval/final-train-dev-all-0.qres

echo "=== Done. Output: output/contriever/prime_eval/final-train-dev-all-0.qres ==="
