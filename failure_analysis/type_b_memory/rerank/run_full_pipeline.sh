pkill -9 ollama; sleep 5
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=127.0.0.1:$((11434+i)) ollama serve &
  sleep 2
done
sleep 15

PY=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
EP=http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436,http://127.0.0.1:11437,http://127.0.0.1:11438,http://127.0.0.1:11439,http://127.0.0.1:11440,http://127.0.0.1:11441
cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval

# ═══════════════════════════════════════════════════════════
# Step 1: Stage 1+2 on train → build memory v1
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/train_memory/run_stage1_stage2.py \
  --splits train --model gemma4:31b --endpoints $EP

$PY failure_analysis/type_b_memory/rerank/train_memory/extract_rerouted.py \
  --splits train --detect_model gemma4:31b

$PY failure_analysis/type_b_memory/rerank/train_memory/build_memory_context.py \
  --splits train

# ═══════════════════════════════════════════════════════════
# Step 2: Build memory KG
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/train_memory/build_memory_kg.py build

# ═══════════════════════════════════════════════════════════
# Step 3: Stage 1+2 on val/test with memory v1
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/detect_route/run_stage1_stage2.py \
  --splits val test --model gemma4:31b --endpoints $EP --memory_version memory_v1

# ═══════════════════════════════════════════════════════════
# Step 4: Rerank v1 on val → verify → memory v2
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits val --model gemma4:31b --endpoints $EP --memory_version memory_v1
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val --model gemma4:31b --alpha 0.7 --memory_version memory_v1

$PY failure_analysis/type_b_memory/rerank/scoring/verify.py verify \
  --splits val --model gemma4:31b --endpoints $EP --memory_version memory_v1

$PY failure_analysis/type_b_memory/rerank/scoring/verify.py update-memory \
  --splits val --model gemma4:31b --memory_version memory_v1

# ═══════════════════════════════════════════════════════════
# Step 5: Stage 1+2 on val/test with memory v2 + KG
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/detect_route/run_stage1_stage2.py \
  --splits val test --model gemma4:31b --endpoints $EP --memory_version memory_v2

$PY failure_analysis/type_b_memory/rerank/detect_route/run_stage1_stage2.py \
  --splits val test --model gemma4:31b --endpoints $EP --memory_version memory_kg

# ═══════════════════════════════════════════════════════════
# Step 6: Rerank scoring — all conditions on test
# ═══════════════════════════════════════════════════════════
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits test --model gemma4:31b --endpoints $EP --memory_version memory_v1
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model gemma4:31b --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version memory_v1

$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits test --model gemma4:31b --endpoints $EP --memory_version memory_v2
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model gemma4:31b --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version memory_v2

$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits test --model gemma4:31b --endpoints $EP --memory_version memory_kg
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model gemma4:31b --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version memory_kg

$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
  --splits test --model gemma4:31b --endpoints $EP --memory_version memory_v1 --no_memory
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
  --splits val test --model gemma4:31b --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version memory_v1 --no_memory
