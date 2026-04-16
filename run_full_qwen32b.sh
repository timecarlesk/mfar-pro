#!/bin/bash
# run_full_qwen32b.sh — qwen3:32b 端到端 pipeline，只跑 v1 / v2 / no_memory

# ═══════════════════════════════════════════════════════════
# 启动 8 路 Ollama（每张 H100 一个实例）
# 注意：pkill 不加 set -e，否则没有 ollama 进程时会直接退出
# ═══════════════════════════════════════════════════════════
pkill -9 ollama || true
sleep 5
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=127.0.0.1:$((11434+i)) ollama serve &
  sleep 2
done
sleep 15

set -e  # ollama 起来之后，任何 Python step 失败立即退出

PY=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
EP=http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436,http://127.0.0.1:11437,http://127.0.0.1:11438,http://127.0.0.1:11439,http://127.0.0.1:11440,http://127.0.0.1:11441
MODEL=qwen3:32b

cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval

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
# Step 5: Rerank — v1, v2, no_memory 三条线，val+test + α sweep
# ═══════════════════════════════════════════════════════════
# v1
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits test --model $MODEL --endpoints $EP --memory_version memory_v1
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model $MODEL --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version memory_v1

# v2
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits test --model $MODEL --endpoints $EP --memory_version memory_v2
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model $MODEL --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version memory_v2

# no_memory（用 memory_v1 的 Stage 1+2 cache 做 routing gate，但 Stage 3 不加 [RELEVANT] 标签）
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits val test --model $MODEL --endpoints $EP --memory_version memory_v1 --no_memory
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model $MODEL --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version memory_v1 --no_memory

echo "======== ALL DONE ========"
