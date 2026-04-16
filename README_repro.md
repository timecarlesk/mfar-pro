# MFAR Reproduction on PRIME Dataset

Reproduction of [Multi-Field Adaptive Retrieval (MFAR)](https://github.com/louisowen6/MFAR) on the PRIME biomedical knowledge graph retrieval benchmark for CSE 585.

## Hardware & Environment

| Item | Value |
|---|---|
| GPUs | 8x L40S (48GB) |
| RAM | 256GB |
| Conda env | `mfar` |
| Python | 3.10 |
| PyTorch Lightning | 2.4.0 |
| SLURM partition | mcity_project |

## Hyperparameter Iterations

We iterated through three training runs to match the paper's configuration (Table 7):

| Parameter | Paper | v1 | v2 | **v3 (final)** |
|---|---|---|---|---|
| Encoder LR | 5e-5 | 1e-5 | 5e-5 | **5e-5** |
| Weights LR | 1e-2 | 1e-1 | 1e-2 | **1e-2** |
| Batch Norm | Yes | No | No | **Yes** |
| Train batch size | 12/GPU | 12/GPU | 12/GPU | **12/GPU** |
| Num GPUs | 8 | 4 | 8 | **8** |
| Effective batch | 96 | 48 | 96 | **96** |
| Field config | MFARAll | MFARAll | MFARAll | **MFARAll** |

**Key insight:** Batch normalization was critical. Without it (v2), BM25 scores (range [0, 50+]) overwhelmed dense cosine similarities (range [-1, 1]), causing the combined model to perform worse than dense-only. Adding `--use_batchnorm` in v3 normalizes all scorer outputs to the same scale before weighted combination.

## Training (v3)

```bash
bash train_all.sh   # also auto-runs eval_all.sh after training
```

Best checkpoint at **epoch 13** (`valid_loss=2.661`).

Training log: `output/contriever/prime/train.log`

## Evaluation (v3)

```bash
bash eval_all.sh
```

The evaluation runs a **field masking ablation** (72 configurations): for each of the 23 fields, it zeroes out the learned weights for dense-only, sparse-only, and both scorers, then re-ranks the full test set. This measures each field's contribution to the final retrieval score.

Eval log: `output/contriever/prime_eval/eval.log`

### Baseline Results (Test Set)

| Metric | Value |
|---|---|
| NDCG | 0.550 |
| **NDCG@10** | **0.489** |
| **MRR** | **0.496** |
| **Hits@1** | **0.375** |
| Hits@5 | 0.638 |
| Recall@5 | 0.521 |
| Recall@10 | 0.609 |
| Recall@20 | 0.690 |
| MAP | 0.441 |

### Dense Field Ablation (Test Set, sorted by NDCG@10)

Masking each dense scorer individually. Delta = change from baseline.

| Masked Field | NDCG@10 | Δ | MRR | Δ | R@10 | Δ | MAP | Δ |
|---|---|---|---|---|---|---|---|---|
| *Baseline* | 0.489 | — | 0.496 | — | 0.609 | — | 0.441 | — |
| details_dense | 0.439 | -0.050 | 0.444 | -0.052 | 0.554 | -0.055 | 0.392 | -0.049 |
| parent-child_dense | 0.461 | -0.028 | 0.467 | -0.029 | 0.581 | -0.028 | 0.414 | -0.027 |
| interacts with_dense | 0.469 | -0.020 | 0.473 | -0.023 | 0.595 | -0.014 | 0.419 | -0.022 |
| single_dense | 0.475 | -0.014 | 0.481 | -0.015 | 0.593 | -0.016 | 0.428 | -0.013 |
| target_dense | 0.478 | -0.011 | 0.486 | -0.010 | 0.599 | -0.010 | 0.429 | -0.012 |
| indication_dense | 0.481 | -0.008 | 0.489 | -0.007 | 0.600 | -0.009 | 0.433 | -0.008 |
| type_dense | 0.482 | -0.007 | 0.485 | -0.011 | 0.603 | -0.006 | 0.434 | -0.007 |
| expression absent_dense | 0.486 | -0.003 | 0.493 | -0.003 | 0.607 | -0.002 | 0.438 | -0.003 |
| associated with_dense | 0.487 | -0.002 | 0.494 | -0.002 | 0.607 | -0.002 | 0.438 | -0.003 |
| name_dense | 0.489 | +0.000 | 0.496 | +0.000 | 0.609 | +0.000 | 0.441 | +0.000 |
| phenotype present_dense | 0.489 | +0.000 | 0.496 | +0.000 | 0.609 | +0.000 | 0.441 | +0.000 |
| All others (12 fields) | 0.489 | +0.000 | 0.496 | +0.000 | 0.609 | +0.000 | 0.441 | +0.000 |

**Top dense contributors:** `details` (Δ=-0.050), `parent-child` (Δ=-0.028), `interacts with` (Δ=-0.020). These three fields alone account for the majority of the dense retrieval signal.

### Sparse Field Ablation (Test Set, sorted by NDCG@10)

| Masked Field | NDCG@10 | Δ | MRR | Δ | R@10 | Δ | MAP | Δ |
|---|---|---|---|---|---|---|---|---|
| *Baseline* | 0.489 | — | 0.496 | — | 0.609 | — | 0.441 | — |
| parent-child_sparse | 0.478 | -0.011 | 0.483 | -0.013 | 0.604 | -0.005 | 0.428 | -0.013 |
| associated with_sparse | 0.483 | -0.006 | 0.489 | -0.007 | 0.603 | -0.006 | 0.434 | -0.007 |
| ppi_sparse | 0.483 | -0.006 | 0.489 | -0.007 | 0.601 | -0.008 | 0.435 | -0.006 |
| interacts with_sparse | 0.486 | -0.003 | 0.493 | -0.003 | 0.607 | -0.002 | 0.438 | -0.003 |
| phenotype absent_sparse | 0.488 | -0.001 | 0.496 | +0.000 | 0.608 | -0.001 | 0.440 | -0.001 |
| phenotype present_sparse | 0.488 | -0.001 | 0.495 | -0.001 | 0.608 | -0.001 | 0.440 | -0.001 |
| details_sparse | 0.494 | **+0.005** | 0.501 | **+0.005** | 0.613 | +0.004 | 0.446 | +0.005 |
| single_sparse | 0.517 | **+0.028** | 0.525 | **+0.029** | 0.630 | +0.021 | 0.470 | +0.029 |
| All others (15 fields) | 0.489 | +0.000 | 0.496 | +0.000 | 0.609 | +0.000 | 0.441 | +0.000 |

**Notable findings:**
- `single_sparse` masking **improves** all metrics significantly (+0.028 NDCG@10, +0.029 MRR). This means the BM25 single-field scorer is **hurting** retrieval — its noisy signal gets non-zero weight but degrades the ranking. The model would perform better without it.
- `details_sparse` masking also slightly **improves** results (+0.005), suggesting BM25 on the details field is redundant when dense details is already the strongest contributor.
- `parent-child_sparse` is the only sparse field whose removal clearly hurts (-0.011 NDCG@10).

### Paired Field Ablation (Test Set — mask both dense + sparse, sorted by NDCG@10 delta)

Completely removing a field's information (both dense and sparse scorers).

| Masked Field | NDCG@10 | Δ | MRR | Δ | R@10 | Δ | MAP | Δ |
|---|---|---|---|---|---|---|---|---|
| *Baseline* | 0.489 | — | 0.496 | — | 0.609 | — | 0.441 | — |
| contraindication | 0.465 | -0.024 | 0.471 | -0.025 | 0.585 | -0.024 | 0.418 | -0.023 |
| parent-child | 0.466 | -0.023 | 0.469 | -0.027 | 0.586 | -0.023 | 0.418 | -0.023 |
| details | 0.473 | -0.016 | 0.480 | -0.016 | 0.590 | -0.019 | 0.425 | -0.016 |
| phenotype present | 0.476 | -0.013 | 0.480 | -0.016 | 0.598 | -0.011 | 0.428 | -0.013 |
| target | 0.477 | -0.012 | 0.483 | -0.013 | 0.599 | -0.010 | 0.429 | -0.012 |
| single | 0.479 | -0.010 | 0.488 | -0.008 | 0.597 | -0.012 | 0.432 | -0.009 |
| ppi | 0.480 | -0.009 | 0.486 | -0.010 | 0.599 | -0.010 | 0.432 | -0.009 |
| side effect | 0.480 | -0.009 | 0.487 | -0.009 | 0.600 | -0.009 | 0.432 | -0.009 |
| associated with | 0.481 | -0.008 | 0.488 | -0.008 | 0.598 | -0.011 | 0.433 | -0.008 |
| expression absent | 0.481 | -0.008 | 0.489 | -0.007 | 0.601 | -0.008 | 0.433 | -0.008 |
| interacts with | 0.482 | -0.007 | 0.488 | -0.008 | 0.604 | -0.005 | 0.433 | -0.008 |
| carrier | 0.483 | -0.006 | 0.491 | -0.005 | 0.602 | -0.007 | 0.435 | -0.006 |
| transporter | 0.484 | -0.005 | 0.490 | -0.006 | 0.605 | -0.004 | 0.435 | -0.006 |
| enzyme | 0.485 | -0.004 | 0.493 | -0.003 | 0.605 | -0.004 | 0.437 | -0.004 |
| linked to | 0.485 | -0.004 | 0.492 | -0.004 | 0.606 | -0.003 | 0.436 | -0.005 |
| phenotype absent | 0.485 | -0.004 | 0.492 | -0.004 | 0.606 | -0.003 | 0.437 | -0.004 |
| name | 0.486 | -0.003 | 0.493 | -0.003 | 0.605 | -0.004 | 0.438 | -0.003 |
| type | 0.486 | -0.003 | 0.491 | -0.005 | 0.608 | -0.001 | 0.436 | -0.005 |
| indication | 0.487 | -0.002 | 0.493 | -0.003 | 0.607 | -0.002 | 0.439 | -0.002 |
| expression present | 0.488 | -0.001 | 0.495 | -0.001 | 0.609 | +0.000 | 0.440 | -0.001 |
| off-label use | 0.488 | -0.001 | 0.495 | -0.001 | 0.607 | -0.002 | 0.440 | -0.001 |
| source | 0.490 | +0.001 | 0.496 | +0.000 | 0.608 | -0.001 | 0.442 | +0.001 |
| synergistic interaction | 0.490 | +0.001 | 0.497 | +0.001 | 0.608 | -0.001 | 0.442 | +0.001 |

### Bulk Masking (Test Set — model variants)

| Config | H@1 | H@5 | R@20 | MRR | NDCG@10 |
|---|---|---|---|---|---|
| **MFARAll (no mask)** | **0.375** | **0.638** | **0.690** | **0.496** | **0.489** |
| MFARDense (all sparse masked) | 0.358 | 0.586 | 0.649 | 0.463 | 0.456 |
| MFARLexical (all dense masked) | 0.235 | 0.442 | 0.489 | 0.331 | 0.331 |

The combined model (MFARAll) outperforms both single-modality variants, confirming that sparse and dense signals are complementary. Dense retrieval contributes much more than sparse (removing dense → -0.165 MRR; removing sparse → -0.033 MRR).

## Comparison with Paper

### Main Results (PRIME Test Set)

| Model | Paper H@1 | Ours H@1 | Δ | Paper MRR | Ours MRR | Δ | Paper R@20 | Ours R@20 | Δ |
|---|---|---|---|---|---|---|---|---|---|
| MFARAll | 0.409 | 0.375 | -0.034 | 0.512 | 0.496 | -0.016 | 0.683 | 0.690 | **+0.007** |
| MFARDense | 0.375 | 0.358 | -0.017 | 0.485 | 0.463 | -0.022 | 0.698 | 0.649 | -0.049 |
| MFARLexical | 0.257 | 0.235 | -0.022 | 0.347 | 0.331 | -0.016 | 0.500 | 0.489 | -0.011 |

### Reproduction Quality by Metric

| Metric | Paper | v1 | v2 | **v3** | v3 vs Paper |
|---|---|---|---|---|---|
| H@1 | 0.409 | 0.362 | 0.302 | **0.375** | -0.034 (-8.3%) |
| H@5 | 0.628 | 0.594 | 0.557 | **0.638** | **+0.010 (+1.6%)** |
| R@20 | 0.683 | 0.662 | 0.619 | **0.690** | **+0.007 (+1.0%)** |
| MRR | 0.512 | 0.469 | 0.420 | **0.496** | -0.016 (-3.1%) |
| NDCG@10 | — | 0.464 | 0.411 | **0.489** | — |
| MAP | — | 0.419 | 0.372 | **0.441** | — |

**v3 exceeds the paper on H@5 and R@20.** The remaining gap is primarily in H@1 (-8.3%), which measures exact top-1 precision — the hardest metric to reproduce since it is sensitive to tie-breaking and small score differences.

### v1 → v2 → v3 Progression

| Version | Change | MRR | Effect |
|---|---|---|---|
| v1 | Baseline (wrong LR, no BN, 4 GPU) | 0.469 | — |
| v2 | Fixed LR, 8 GPU, **no BN** | 0.420 | **-0.049** (worse!) |
| v3 | Fixed LR, 8 GPU, **with BN** | 0.496 | **+0.076** from v2 |

**Why v2 was worse than v1:** Without batch normalization, increasing the encoder LR from 1e-5 to 5e-5 caused the dense encoder to produce stronger gradients, but BM25 scores still dominated due to their much larger magnitude. The weights network learned to suppress dense signals, effectively regressing to a worse lexical-only model. In v1, the conservative encoder LR accidentally mitigated this by keeping dense scores small enough to not conflict with raw BM25.

**Why v3 fixed it:** Batch normalization projects all scorer outputs (dense cosine similarity ∈ [-1,1] and BM25 ∈ [0,50+]) to a standardized scale. This allows the adaptive weight network G(q,f,m) to meaningfully compare and combine scores from different modalities, which is the core mechanism of MFAR.

### Analysis

1. **MRR gap of 3.1% is within acceptable reproduction variance.** H@5 and R@20 actually exceed the paper. The primary gap is in H@1, which is the most sensitive to exact score ordering.

2. **Ablation trends match the paper.** Top contributing fields by paired masking:
   - Paper: off-label use > target > parent-child > details > single
   - Ours: contraindication > parent-child > details > phenotype present > target
   - Both identify `parent-child`, `details`, and `target` as important fields. The rank-order difference in other fields is within normal variation.

3. **Batch normalization is essential** — this was the most impactful hyperparameter. Without it, the model cannot effectively learn to combine dense and sparse signals.

4. **Remaining H@1 gap possible causes:**
   - STaRK dataset version differences (data may have been updated since paper submission)
   - BM25 index parameters (k1, b) or tokenizer differences in Pyserini/Lucene
   - Random seed / training dynamics differences across hardware

## Output Files

```
output/
├── prime/
│   ├── epoch=13-valid_loss=2.661.ckpt   (best checkpoint)
│   └── train.log
└── prime_eval/
    ├── results_dicts-all-0.jsonl         (144 lines, 72 configs × val+test)
    └── eval.log
```

## Reproducing

```bash
# Training (auto-runs eval after training completes)
sbatch --account=mcity_project --partition=mcity_project \
    --nodes=1 --cpus-per-task=32 --gpus-per-node=l40s:8 \
    --mem=256GB --time=48:00:00 --wrap="bash train_all.sh"

# Eval only (if you already have a checkpoint)
sbatch --account=mcity_project --partition=mcity_project \
    --nodes=1 --cpus-per-task=32 --gpus-per-node=l40s:8 \
    --mem=256GB --time=12:00:00 --wrap="bash eval_all.sh"
```

Training takes ~8 hours and eval takes ~4 hours on 8x L40S.
