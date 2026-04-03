#!/bin/bash
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --job-name=qwen3_train
#SBATCH --output=output/failure_analysis/type_b_memory/slurm_train_%j.log

cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval
PY=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python

# Start ollama
ollama serve &
sleep 5

# Pull model if needed
ollama pull qwen3:8b

# Clean old cache and run
rm -f output/failure_analysis/type_b_memory/qwen3_cache_train.jsonl
$PY failure_analysis/type_b_memory/batch_qwen3_inference.py --splits train --workers 1

echo "Done at $(date)"
