#!/bin/bash
set -e

MFAR_PYTHON=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
export LD_LIBRARY_PATH=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/lib:$LD_LIBRARY_PATH
export PATH=/home/xxxchen/miniforge3/bin:/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval
BASE_DIR=data

mkdir -p output/prime_eval

# ============ Evaluate Prime ============
echo "=== Evaluating Prime at $(date) ==="
$MFAR_PYTHON -m mfar.commands.mask_fields \
    --dataset_name prime \
    --data $BASE_DIR/prime \
    --lexical-index $BASE_DIR/prime \
    --temp-dir /tmp/mfar_temp/prime \
    --out ./output/prime_eval \
    --checkpoint_dir ./output/prime \
    --partition val \
    --additional_partition test \
    --field_names "all_dense,all_sparse,single_dense,single_sparse" \
    --dev-batch-size 32 \
    --num_gpus 8 \
    2>&1 | tee output/prime_eval/eval.log

echo "=== Prime evaluation done at $(date) ==="

mkdir -p output/mag_eval

# ============ Evaluate MAG ============
echo "=== Evaluating MAG at $(date) ==="
$MFAR_PYTHON -m mfar.commands.mask_fields \
    --dataset_name mag \
    --data $BASE_DIR/mag \
    --lexical-index $BASE_DIR/mag \
    --temp-dir /tmp/mfar_temp/mag \
    --out ./output/mag_eval \
    --checkpoint_dir ./output/mag \
    --partition val \
    --additional_partition test \
    --field_names "all_dense,all_sparse,single_dense,single_sparse" \
    --dev-batch-size 32 \
    --num_gpus 8 \
    2>&1 | tee output/mag_eval/eval.log

echo "=== MAG evaluation done at $(date) ==="

mkdir -p output/amazon_eval

# ============ Evaluate Amazon ============
echo "=== Evaluating Amazon at $(date) ==="
$MFAR_PYTHON -m mfar.commands.mask_fields \
    --dataset_name amazon \
    --data $BASE_DIR/amazon \
    --lexical-index $BASE_DIR/amazon \
    --temp-dir /tmp/mfar_temp/amazon \
    --out ./output/amazon_eval \
    --checkpoint_dir ./output/amazon \
    --partition val \
    --additional_partition test \
    --field_names "all_dense,all_sparse,single_dense,single_sparse" \
    --dev-batch-size 32 \
    --num_gpus 8 \
    2>&1 | tee output/amazon_eval/eval.log

echo "=== Amazon evaluation done at $(date) ==="
