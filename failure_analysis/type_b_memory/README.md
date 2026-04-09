# Type B Memory: Negation Query Field Re-Routing for mFAR

## Problem

mFAR learns query-conditioned field weights via `score = sum(softmax(q @ W) * field_scores)`. For negation queries (~19% of test), these weights route to the **wrong field**. For example, "drugs NOT indicated for diabetes" causes mFAR to search the `indication` field when the answer is in `contraindication`.

**Impact**: Rerouted queries have 0.340 MRR vs 0.504 for non-negation queries on test.

## Pipeline Overview

```
                  ┌─────────────────────────────────────────────┐
  OFFLINE         │  Training Set (6,162 queries)               │
  (once)          │    ↓ Stage 1+2 (detect + classify)          │
                  │    ↓ extract rerouted + gold docs            │
                  │    ↓ group by (gold_entity_type, neg_pattern)│
                  │    ↓ count field population in gold docs     │
                  │    ↓                                         │
                  │  Memory v1 (text) / Memory KG (graph)       │
                  └─────────────────────────────────────────────┘
                                     │
                  ┌──────────────────▼──────────────────────────┐
  FEEDBACK        │  Memory v1 → rerank val → verify top-1     │
  (optional)      │    ↓ per-group pass rate                    │
                  │    ↓ rate < 70% → add WARNING tag           │
                  │  Memory v2 (text + confidence annotations)  │
                  └─────────────────────────────────────────────┘
                                     │
                  ┌──────────────────▼──────────────────────────┐
  ONLINE          │  Stage 1: Detect — needs rerouting? (yes/no)│
  (per query)     │    ~81% "no" → copy baseline, skip all      │
                  │  Stage 1.5: Classify — negation_pattern +   │
                  │    answer_type                               │
                  │  Stage 2: Route — match memory rule →       │
                  │    output boost_fields (JSON)                │
                  │  Stage 3: Rerank — score top-50 docs 0-10   │
                  │    doc shows ALL fields, boost tagged        │
                  │    [RELEVANT]; final = α·LLM + (1-α)·mFAR   │
                  └─────────────────────────────────────────────┘
```

## Memory Versions

All memory is generated from training data statistics (model-independent). All models share the same memory files.

| Version | File | Description |
|---------|------|-------------|
| **v1** | `memory_context_train.txt` | Per-group boost_fields from gold doc field population frequencies |
| **v2** | `memory_context_train_v2.txt` | v1 + `WARNING` tags on rules with <70% self-verification pass rate |
| **KG** | `memory_kg.json` | v1 structured as a graph: PatternNode → BOOSTS → FieldNode edges |

**v1 example:**
```
When answer_type=disease and query contains "not indicated" (109 training examples):
  Recommended boost_fields: ['contraindication', 'parent-child', 'associated with', ...]
  Gold doc field distribution: contraindication (19%), parent-child (18%), ...
```

**v2 example (same rule + feedback):**
```
When answer_type=disease and query contains "not indicated" (109 training examples):
  Recommended boost_fields: ['contraindication', 'parent-child', 'associated with', ...]
  WARNING: Low verification confidence: 0% (0/48). This rule may be unreliable.
  Consider reasoning from entity type field inventory instead.
```

**How memory is used**: Stage 2 LLM sees the matched rule → outputs `boost_fields` → Stage 3 tags those fields with `[RELEVANT]` in the document. The no-memory ablation shows the same document content without `[RELEVANT]` tags.

## Document Formatting ([RELEVANT] Tag)

Stage 3 shows **all** field content to the LLM. Memory-recommended fields are tagged `[RELEVANT]`:

```
Peptic Ulcer (disease);
[RELEVANT] contraindication: Aspirin, Ibuprofen;
indication: Metformin, Omeprazole;
associated with: H. pylori, stress;
phenotype present: abdominal pain, bleeding
```

No-memory ablation: same content, no `[RELEVANT]` tags. This isolates the memory contribution.

## Results (Best: Qwen3-32B, Memory v1, alpha=0.7)

| Group | N | MRR | Delta | Rel. % |
|-------|---|-----|-------|--------|
| Rerouted | 548 | 0.340 → 0.419 | +0.079 | +23.3% |
| Non-negation | 2253 | 0.504 → 0.504 | 0.000 | 0.0% |
| Overall | 2801 | 0.472 → 0.487 | +0.016 | +3.3% |

## Two Approaches Attempted

### 1. Logit Bias (`logit_bias/`) -- Failed

Added bias to softmax weight logits to boost/suppress specific fields.

**Why it failed**: Bias is per-query not per-document. 43-48% of suppressed fields are populated in gold docs.

### 2. LLM Re-Ranking (`rerank/`) -- Works

Post-hoc re-ranking of mFAR's top-50 using LLM for negation-aware scoring. Non-negation queries keep baseline rankings unchanged (zero regression).

## Code Structure

```
type_b_memory/
├── rerank/                                 # Main pipeline
│   ├── shared/                             #   Shared utilities
│   │   ├── qwen3_client.py                 #     Ollama API, Stage 1+2 prompts, cache I/O
│   │   └── memory_kg.py                    #     MemoryKG class (nodes, edges, query, format)
│   │
│   ├── train_memory/                       #   Step 1-2: Build memory from training data
│   │   ├── run_stage1_stage2.py            #     Stage 1+2 on train split (no memory)
│   │   ├── extract_rerouted.py             #     Extract rerouted queries + gold doc IDs
│   │   ├── build_memory_context.py         #     Field confusion → memory_context_train.txt (v1)
│   │   ├── build_memory_kg.py              #     v1 → memory_kg.json (KG version)
│   │   └── finetune_W.py                   #     (Experimental) Fine-tune W matrix with memory
│   │
│   ├── detect_route/                       #   Step 3: Detect + route on val/test
│   │   └── run_stage1_stage2.py            #     Stage 1+2 with memory context
│   │
│   ├── scoring/                            #   Step 4-6: Score, merge, evaluate
│   │   ├── rerank.py                       #     score / merge / pilot commands
│   │   ├── evaluate.py                     #     MRR/Hit@k/NDCG grouped evaluation
│   │   └── verify.py                       #     Self-verification → memory v2 (WARNING tags)
│   │
│   ├── analysis/                           #   Validation utilities
│   │   └── validate_boost_precision.py     #     Boost precision/recall vs gold docs
│   │
│   ├── run_full_pipeline.sh                #   End-to-end script (gemma4:31b)
│   └── run_full_pipeline_qwen35_27b.sh     #   End-to-end script (qwen3.5:27b)
│
├── logit_bias/                             # Failed approach
│   ├── negation_memory_module.py           #   Bias injection into LinearWeights
│   ├── eval.sh                             #   Evaluation script
│   └── dump_logit_range.py                 #   W matrix analysis
│
├── meta_harness/                           # (Legacy) Meta-learning harness experiments
│
├── rerank_results_standalone.tex           # Paper writeup
└── rerank_results.tex                      # Alternate writeup
```

## Output Structure

```
output/failure_analysis/type_b_memory/
├── analysis/                               # Memory files (shared across models)
│   ├── memory_context_train.txt            #   Memory v1
│   ├── memory_context_train_v2.txt         #   Memory v2 (v1 + WARNING tags)
│   ├── memory_kg.json                      #   Memory KG
│   ├── rerouted_train.json                 #   Rerouted training queries
│   ├── field_confusion_train.json          #   Per-group field population stats
│   └── verification_rates.json             #   Self-verification pass rates
│
├── cache/
│   ├── stage12/{model_tag}/                # Stage 1+2 detection/routing cache
│   │   ├── shared/                         #   Train split (no memory, shared)
│   │   │   └── qwen3_cache_train.jsonl
│   │   ├── memory_v1/                      #   Val/test with memory v1
│   │   │   ├── qwen3_cache_val.jsonl
│   │   │   └── qwen3_cache_test.jsonl
│   │   ├── memory_v2/                      #   Val/test with memory v2
│   │   └── memory_kg/                      #   Val/test with memory KG
│   │
│   ├── rerank/{model_tag}/                 # Stage 3 rerank score cache
│   │   ├── memory_v1/
│   │   │   └── rerank_cache_{split}.jsonl
│   │   ├── memory_v1_no_memory/            #   No-memory ablation
│   │   ├── memory_v2/
│   │   └── memory_kg/
│   │
│   └── verify/{model_tag}/                 # Self-verification cache
│       └── verify_cache_{split}.jsonl
│
├── runs/
│   └── rerank/{model_tag}/                 # Merged results (.qres) + evaluation
│       ├── memory_v1/
│       │   └── alpha_0.7_top50/
│       │       ├── final-all-0.qres
│       │       ├── final-additional-all-0.qres
│       │       └── memory_evaluation.json
│       ├── memory_v1_no_memory/
│       ├── memory_v2/
│       └── memory_kg/
│
└── backups/                                # Timestamped backups of previous runs
```

**Model tags**: `qwen3_8b`, `qwen3_32b`, `qwen3.5_27b`, `gemma4_e4b`, `gemma4_31b`

## How to Run

### Prerequisites

```bash
PY=/scratch/mcity_project_root/mcity_project/xxxchen/.conda/envs/mfar/bin/python
cd /scratch/mcity_project_root/mcity_project/xxxchen/CSE_585/multifield-adaptive-retrieval

# Start 8 Ollama instances (8x H100)
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=127.0.0.1:$((11434+i)) ollama serve &
  sleep 2
done
sleep 15
EP=http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436,http://127.0.0.1:11437,http://127.0.0.1:11438,http://127.0.0.1:11439,http://127.0.0.1:11440,http://127.0.0.1:11441
```

### Full Pipeline (6 Steps)

See `run_full_pipeline.sh` for the complete script. Summary:

```bash
MODEL=qwen3:8b  # or qwen3:32b, gemma4:e4b, qwen3.5:27b, etc.

# Step 1: Stage 1+2 on train (no memory) → extract rerouted → build memory v1
$PY .../train_memory/run_stage1_stage2.py --splits train --model $MODEL --endpoints $EP
$PY .../train_memory/extract_rerouted.py --splits train --detect_model $MODEL
$PY .../train_memory/build_memory_context.py --splits train

# Step 2: Build memory KG
$PY .../train_memory/build_memory_kg.py build

# Step 3: Stage 1+2 on val/test with memory v1
$PY .../detect_route/run_stage1_stage2.py --splits val test --model $MODEL --endpoints $EP --memory_version memory_v1

# Step 4: Rerank v1 on val → verify → build memory v2
$PY .../scoring/rerank.py score --splits val --model $MODEL --endpoints $EP --memory_version memory_v1
$PY .../scoring/rerank.py merge --splits val --model $MODEL --alpha 0.7 --memory_version memory_v1
$PY .../scoring/verify.py verify --splits val --model $MODEL --endpoints $EP --memory_version memory_v1
$PY .../scoring/verify.py update-memory --splits val --model $MODEL --memory_version memory_v1

# Step 5: Stage 1+2 on val/test with memory v2 + KG
$PY .../detect_route/run_stage1_stage2.py --splits val test --model $MODEL --endpoints $EP --memory_version memory_v2
$PY .../detect_route/run_stage1_stage2.py --splits val test --model $MODEL --endpoints $EP --memory_version memory_kg

# Step 6: Score + merge + evaluate all conditions
for MV in memory_v1 memory_v2 memory_kg; do
  $PY .../scoring/rerank.py score --splits test --model $MODEL --endpoints $EP --memory_version $MV
  $PY .../scoring/rerank.py merge --splits val test --model $MODEL --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version $MV
done
# No-memory ablation
$PY .../scoring/rerank.py score --splits test --model $MODEL --endpoints $EP --memory_version memory_v1 --no_memory
$PY .../scoring/rerank.py merge --splits val test --model $MODEL --alpha_sweep 0.3 0.5 0.7 1.0 --memory_version memory_v1 --no_memory
```

## Key Design Decisions

1. **[RELEVANT] tag, not field hiding**: All fields shown with content. Boost fields tagged `[RELEVANT]` for attention guidance. Previously tried `[has data]` hiding which hurt small models.

2. **Memory from gold doc statistics**: Memory rules come from counting which fields are populated in gold docs per (answer_type, negation_pattern) group. This is objective data, not LLM-generated.

3. **`_has_effective_reroute()` filter**: Queries with `needs_reroute=True` but empty `boost_fields` (Stage 2 produced nothing) are excluded from scoring/merging. They keep baseline rankings.

4. **Hybrid scoring**: `final = alpha * LLM + (1-alpha) * mFAR`. Pure LLM (alpha=1.0) underperforms alpha=0.7, showing mFAR's retrieval signal is complementary.

5. **Zero regression**: Non-flagged queries (~81%) bypass re-ranking entirely. Their baseline rankings are copied verbatim.

6. **Model-specific cache paths**: Each model gets its own subfolder under `cache/stage12/{model_tag}/` and `cache/rerank/{model_tag}/` to prevent cache pollution across models.

7. **DOC_FORMAT_VERSION**: Cache entries tagged with format version to prevent mixing old/new document formatting across runs.
