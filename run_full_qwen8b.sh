#!/bin/bash
# run_full_qwen8b.sh — qwen3:8b 端到端 pipeline，4x H100，只跑 v1 / v2 / no_memory

# ═══════════════════════════════════════════════════════════
# 启动 4 路 Ollama（每张 H100 一个实例）
# ═══════════════════════════════════════════════════════════
pkill -9 ollama || true
sleep 5
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=127.0.0.1:$((11434+i)) ollama serve &
  sleep 2
done
sleep 15

PY=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
EP=http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436,http://127.0.0.1:11437
MODEL=qwen3:8b
SWEEP="0.3 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 1.0"

cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval

set -e  # Python step 失败立即退出

# ═══════════════════════════════════════════════════════════
# Step 1: Train → 挖 memory v1
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/train_memory/run_stage1_stage2.py \
  --splits train --model $MODEL --endpoints $EP

$PY failure_analysis/type_b_memory/rerank/train_memory/extract_rerouted.py \
  --splits train --detect_model $MODEL

$PY failure_analysis/type_b_memory/rerank/train_memory/build_memory_context.py \
  --splits train

# ═══════════════════════════════════════════════════════════
# Step 2: val/test Stage 1+2 with memory_v1
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/detect_route/run_stage1_stage2.py \
  --splits val test --model $MODEL --endpoints $EP --memory_version memory_v1

# ═══════════════════════════════════════════════════════════
# Step 3: Rerank v1 on val → verify → 生成 memory_v2
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits val --model $MODEL --endpoints $EP --memory_version memory_v1
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val --model $MODEL --alpha 0.7 --memory_version memory_v1

$PY failure_analysis/type_b_memory/rerank/scoring/verify.py verify \
  --splits val --model $MODEL --endpoints $EP --memory_version memory_v1

$PY failure_analysis/type_b_memory/rerank/scoring/verify.py update-memory \
  --splits val --model $MODEL --memory_version memory_v1

# ═══════════════════════════════════════════════════════════
# Step 4: val/test Stage 1+2 with memory_v2
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/detect_route/run_stage1_stage2.py \
  --splits val test --model $MODEL --endpoints $EP --memory_version memory_v2

# ═══════════════════════════════════════════════════════════
# Step 5: Rerank — v1, v2, no_memory 三条线
# ═══════════════════════════════════════════════════════════
# v1: score test + merge val+test
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits test --model $MODEL --endpoints $EP --memory_version memory_v1
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model $MODEL --alpha_sweep $SWEEP --memory_version memory_v1

# v2: score val + test + merge val+test
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits val --model $MODEL --endpoints $EP --memory_version memory_v2
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits test --model $MODEL --endpoints $EP --memory_version memory_v2
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model $MODEL --alpha_sweep $SWEEP --memory_version memory_v2

# no_memory: score val + test + merge val+test
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits val test --model $MODEL --endpoints $EP --memory_version memory_v1 --no_memory
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model $MODEL --alpha_sweep $SWEEP --memory_version memory_v1 --no_memory

echo "======== ALL DONE (qwen3:8b) ========"
