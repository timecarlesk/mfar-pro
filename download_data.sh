#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate mfar

# Fix libstdc++ issue - use conda's version
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /scratch/mcity_project_root/mcity_project/xxxchen/multifield-adaptive-retrieval
BASE_DIR=/scratch/mcity_project_root/mcity_project/xxxchen/multifield-adaptive-retrieval/data

download_dataset() {
    local name=$1
    local dir=$BASE_DIR/$name
    mkdir -p $dir
    echo "=== [$name] Starting at $(date) ==="
    python -m mfar.commands.stark.stark_to_trec --out $dir --dataset_name $name
    echo "=== [$name] Corpus done, downloading queries ==="
    python -m mfar.commands.stark.download_queries --out $dir --dataset_name $name
    echo "=== [$name] Queries done, building BM25 index ==="
    python -m mfar.commands.create_bm25s_index --data_path $dir --dataset_name $name --output_path $dir
    echo "=== [$name] All done at $(date) ==="
}

download_dataset prime &
download_dataset mag &
download_dataset amazon &

wait
echo "=== All datasets done at $(date) ==="
