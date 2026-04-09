# Type B Memory: Negation Query Field Re-Routing for mFAR

## Problem

mFAR learns query-conditioned field weights via `score = sum(softmax(q @ W) * field_scores)`. For negation queries (~20% of all queries), these weights route to the **wrong field**. For example, "drugs NOT indicated for diabetes" causes mFAR to search the `indication` field when the answer is in `contraindication`.

**Impact**: Rerouted queries have 0.332-0.347 MRR vs 0.493-0.505 for non-negation queries.

## Two Approaches Attempted

### 1. Logit Bias (`logit_bias/`) — Failed

Added `±α` bias to the softmax weight logits to boost/suppress specific fields.

**Why it failed**:
- Bias applies uniformly to ALL candidate documents — can't distinguish "this doc has the right field" from "this doc doesn't"
- 43-48% of suppressed fields are populated in gold docs → actively harms ranking
- Boosting an empty field on a doc = multiplying by zero → no effect

### 2. LLM Re-Ranking (`rerank/`) — Works (+0.057 MRR)

Post-hoc re-ranking of mFAR's top-50 results using Qwen3 for negation-aware relevance scoring. Only applied to flagged negation queries; non-negation queries keep baseline rankings unchanged (zero regression).

**Three-stage pipeline**:

```
Stage 1 (Detect):  Qwen3 yes/no — does this query need field re-routing?
                   Runs on ALL queries. ~95% get "no" and are skipped.

Stage 2 (Route):   Qwen3 identifies boost_fields — which fields should the
                   reranker focus on? Uses memory context from training data.

Stage 3 (Rerank):  For each (query, top-50 doc) pair, Qwen3 scores 0-10
                   for negation-aware relevance. Doc formatting prioritizes
                   boost_fields from Stage 2. Final score = hybrid of
                   LLM score and mFAR score.
```

**Memory context**: Stage 2's prompt is augmented with training data statistics — for each (answer_type, negation_pattern) combination observed in training, the memory records which fields are actually populated in gold docs. This gives the LLM data-driven guidance instead of relying on parametric knowledge alone.

## Results

| Model | α | Val Reroute ΔMRR | Test Reroute ΔMRR | Test Overall ΔMRR | Non-neg Δ |
|-------|---|-----------------|------------------|-------------------|-----------|
| Qwen3-8B | 0.7 | +0.035 | +0.043 | +0.008 | 0.000 |
| **Qwen3-32B** | **0.7** | **+0.029** | **+0.057** | **+0.011** | **0.000** |

Best config: Qwen3-32B, α=0.7 (70% LLM + 30% mFAR), top-50 re-ranking.

## Directory Structure

```
type_b_memory/
├── rerank/                              # Rerank approach (works)
│   ├── shared/
│   │   └── qwen3_client.py             #   Ollama API, prompts, parsing
│   ├── train_memory/                    #   Step 1: Generate memory from train
│   │   ├── run_stage1_stage2.py         #     Stage 1+2 on train (default prompt)
│   │   ├── extract_rerouted.py          #     Extract rerouted queries + gold docs
│   │   └── build_memory_context.py      #     Field confusion → memory_context.txt
│   ├── detect_route/                    #   Step 2: Detect + route val/test
│   │   └── run_stage1_stage2.py         #     Stage 1+2 with memory context
│   ├── scoring/                         #   Step 3: LLM scoring + merge
│   │   ├── rerank.py                    #     score / merge / pilot commands
│   │   └── evaluate.py                  #     MRR/Hit@k comparison
│   └── analysis/                        #   Step 4: Validation
│       └── validate_boost_precision.py  #     Boost precision/recall vs gold docs
│
├── logit_bias/                          # Logit bias approach (failed)
│   ├── negation_memory_module.py        #   Bias injection into LinearWeights
│   ├── eval.sh                          #   Evaluation script
│   └── dump_logit_range.py              #   W matrix analysis
│
└── rerank_results_standalone.tex/pdf    # Results writeup
```

## Output Structure

```
output/failure_analysis/type_b_memory/
├── cache/
│   ├── qwen3_8b/                        # Stage 1+2 detection cache
│   │   ├── qwen3_cache_train.jsonl
│   │   ├── qwen3_cache_val.jsonl
│   │   └── qwen3_cache_test.jsonl
│   └── rerank/                          # Rerank scores by model
│       ├── qwen3_8b/
│       │   ├── rerank_cache_val.jsonl
│       │   └── rerank_cache_test.jsonl
│       └── qwen3_32b/
│           ├── rerank_cache_val.jsonl
│           └── rerank_cache_test.jsonl
├── analysis/
│   ├── memory_context_train.txt         # Memory context (injected into Stage 2)
│   ├── rerouted_train.json              # Rerouted training queries
│   └── field_confusion_train.json       # Per-group field population stats
└── runs/
    ├── rerank/
    │   ├── qwen3_8b/alpha_0.7_top50/    # Results per model + alpha
    │   └── qwen3_32b/alpha_0.7_top50/
    └── logit_bias/                      # Old logit bias results
```

## How to Run

### Prerequisites

```bash
PY=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval

# Start 6 Ollama instances (one per GPU)
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=127.0.0.1:$((11434+i)) ollama serve &
done
sleep 10
EP=http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436,http://127.0.0.1:11437,http://127.0.0.1:11438,http://127.0.0.1:11439
```

### Step 1: Generate Memory from Training Data

```bash
# Stage 1+2 on train (uses default prompt, no memory context yet)
$PY failure_analysis/type_b_memory/rerank/train_memory/run_stage1_stage2.py \
    --splits train --endpoints $EP

# Extract rerouted queries
$PY failure_analysis/type_b_memory/rerank/train_memory/extract_rerouted.py \
    --splits train

# Build field confusion → memory_context_train.txt
$PY failure_analysis/type_b_memory/rerank/train_memory/build_memory_context.py \
    --splits train
```

### Step 2: Detect + Route Val/Test (with Memory)

```bash
$PY failure_analysis/type_b_memory/rerank/detect_route/run_stage1_stage2.py \
    --splits val test --endpoints $EP
```

### Step 3: Rerank

```bash
# Pilot (50 queries, sanity check)
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py pilot \
    --splits val --model qwen3:8b --endpoint http://127.0.0.1:11434

# Full scoring
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py score \
    --splits val test --model qwen3:8b --endpoints $EP

# Merge + evaluate (instant, sweep alpha)
$PY failure_analysis/type_b_memory/rerank/scoring/rerank.py merge \
    --splits val test --model qwen3:8b --alpha_sweep 0.3 0.5 0.7 1.0
```

### Step 4: Validate

```bash
$PY failure_analysis/type_b_memory/rerank/analysis/validate_boost_precision.py
```

## Key Design Decisions

1. **Regex → Qwen3 for detection**: Initially used regex for Stage 1, switched to Qwen3 for higher recall (catches negation patterns regex misses like "preclude the use of", "renders unsuitable").

2. **Boost-only doc formatting**: Stage 3's document representation shows boost_fields with actual content, other fields as `[has data]`. This focuses the LLM on the fields that matter for the negation constraint.

3. **Hybrid scoring**: `final = α × LLM + (1-α) × mFAR`. Pure LLM (α=1.0) underperforms hybrid (α=0.7), showing mFAR's retrieval signal is complementary.

4. **Zero regression guarantee**: Non-flagged queries (~80%) bypass re-ranking entirely — their baseline rankings are copied verbatim.

5. **Memory context from training data**: Stage 2's prompt is augmented with (answer_type, negation_pattern) → recommended boost_fields mappings learned from training set field confusion analysis. This is direct pattern matching guidance, not abstract statistics that the LLM would need to interpret.
