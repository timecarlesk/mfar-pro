#!/bin/bash
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:8
#SBATCH --mem=128GB
#SBATCH --time=36:00:00
#SBATCH --job-name=meta_harness
#SBATCH --output=output/failure_analysis/type_b_memory/meta_harness_sbatch_%j.log

set -e

# ── Paths ───────────────────────────────────────────────────────────────────
OLLAMA=/home/xxxchen/ollama/bin/ollama
PY=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
WORKDIR=/scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval
API_KEY_FILE=/scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/api_keys

export PATH=/home/xxxchen/ollama/bin:/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin:$PATH
export LD_LIBRARY_PATH=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/lib:$LD_LIBRARY_PATH
export ANTHROPIC_API_KEY=$(cat $API_KEY_FILE | sed 's/^anthropic://')

cd $WORKDIR

# ── Start 8 Ollama instances (one per GPU) ──────────────────────────────────
echo "=== Starting 8 Ollama instances at $(date) ==="

for i in 0 1 2 3 4 5 6 7; do
    PORT=$((11434 + i))
    CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=0.0.0.0:$PORT \
        $OLLAMA serve &> /tmp/ollama_${SLURM_JOB_ID}_${i}.log &
done

echo "  Waiting for Ollama to start..."
sleep 10

# ── Pull model if needed (skip if no network) ──────────────────────────────
OLLAMA_HOST=0.0.0.0:11434 $OLLAMA pull qwen3:8b 2>/dev/null || echo "  Pull skipped (model already cached or no network)"

# ── Health check ────────────────────────────────────────────────────────────
echo "=== Health check ==="
ALIVE=0
for i in 0 1 2 3 4 5 6 7; do
    PORT=$((11434 + i))
    if curl -s http://127.0.0.1:$PORT/api/tags >/dev/null 2>&1; then
        echo "  Port $PORT: OK"
        ALIVE=$((ALIVE + 1))
    else
        echo "  Port $PORT: FAIL"
    fi
done

if [ $ALIVE -eq 0 ]; then
    echo "ERROR: No Ollama endpoints alive. Exiting."
    exit 1
fi
echo "  $ALIVE/8 endpoints alive"

# ── Warm up: preload model on all GPUs ──────────────────────────────────────
echo "=== Warming up model on all GPUs at $(date) ==="
for i in 0 1 2 3 4 5 6 7; do
    PORT=$((11434 + i))
    curl -s http://127.0.0.1:$PORT/api/generate \
        -d '{"model":"qwen3:8b","prompt":"hi","stream":false}' \
        >/dev/null 2>&1 &
done
wait
echo "  Model loaded on all GPUs"

# ── Run Meta-Harness loop ───────────────────────────────────────────────────
ENDPOINTS="http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436,http://127.0.0.1:11437,http://127.0.0.1:11438,http://127.0.0.1:11439,http://127.0.0.1:11440,http://127.0.0.1:11441"

echo ""
echo "=== Starting Meta-Harness loop at $(date) ==="
echo ""

$PY failure_analysis/type_b_memory/meta_harness/loop.py \
    --max_rounds 5 \
    --start_round 2 \
    --split val \
    --model qwen3:8b \
    --endpoints $ENDPOINTS \
    --workers 8

echo ""
echo "=== Meta-Harness loop completed at $(date) ==="

# ── Cleanup: kill Ollama instances ──────────────────────────────────────────
pkill -f "ollama serve" 2>/dev/null || true
echo "  Ollama instances stopped"
