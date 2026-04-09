#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate mfar
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval
BASE_DIR=data

mkdir -p output/prime output/mag output/amazon
rm -rf /tmp/mfar_temp

# ============ Prime (最小，约13万文档) ============
echo "=== Training Prime at $(date) ==="
python -m mfar.commands.train \
    --corpus $BASE_DIR/prime \
    --queries $BASE_DIR/prime \
    --lexical-index $BASE_DIR/prime \
    --temp-dir /tmp/mfar_temp/prime \
    --out ./output/prime \
    --dataset_name prime \
    --encoder_lr 5e-5 \
    --weights_lr 1e-2 \
    --train-batch-size 16 \
    --field_names "all_dense,all_sparse,single_dense,single_sparse" \
    --trec_val_freq 0 \
    --num_gpus 8 \
    --use_batchnorm \
    --additional_partition test \
    2>&1 | tee output/prime/train.log

# ============ MAG (约70万文档) ============
echo "=== Training MAG at $(date) ==="
python -m mfar.commands.train \
    --corpus $BASE_DIR/mag \
    --queries $BASE_DIR/mag \
    --lexical-index $BASE_DIR/mag \
    --temp-dir /tmp/mfar_temp/mag \
    --out ./output/mag \
    --dataset_name mag \
    --encoder_lr 5e-5 \
    --weights_lr 1e-2 \
    --train-batch-size 32 \
    --dev-batch-size 32 \
    --field_names "all_dense,all_sparse,single_dense,single_sparse" \
    --trec_val_freq 0 \
    --num_gpus 8 \
    --use_batchnorm \
    --additional_partition test \
    2>&1 | tee output/mag/train.log

# ============ Amazon (约95万文档) ============
echo "=== Training Amazon at $(date) ==="
python -m mfar.commands.train \
    --corpus $BASE_DIR/amazon \
    --queries $BASE_DIR/amazon \
    --lexical-index $BASE_DIR/amazon \
    --temp-dir /tmp/mfar_temp/amazon \
    --out ./output/amazon \
    --dataset_name amazon \
    --encoder_lr 1e-5 \
    --weights_lr 5e-3 \
    --train-batch-size 16 \
    --dev-batch-size 32 \
    --field_names "all_dense,all_sparse,single_dense,single_sparse" \
    --trec_val_freq 0 \
    --num_gpus 8 \
    --additional_partition test \
    2>&1 | tee output/amazon/train.log

echo "=== All training done at $(date) ==="

# ============ Auto-run eval after training ============
echo "=== Starting eval at $(date) ==="
bash eval_all.sh
echo "=== All done (train + eval) at $(date) ==="
