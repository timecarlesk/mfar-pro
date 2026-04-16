# mFAR Failure Analysis

Systematic failure analysis of the mFAR model on the PRIME biomedical dataset.

## Structure

```
failure_analysis/                      # Scripts (code only)
├── README.md
├── utils.py                           # Shared data loading
├── general/
│   └── failure_analysis.py            # 7-section general analysis
├── multi-hop/
│   └── multi_hop_analysis.py          # Cross-type (multi-hop) analysis
└── negation/
    ├── negation_ablation.py           # Negation taxonomy ablation
    └── negation_examples.py           # Print negation failure examples

output/failure_analysis/               # All generated outputs
├── general/
│   ├── ANALYSIS_REPORT.md
│   └── *.png
├── multi-hop/
│   ├── REPORT.md
│   └── cross_type_comparison.png
└── negation/
    ├── REPORT_val.md
    ├── REPORT_test.md
    └── *.png
```

## Running

All scripts run from the **project root** (`multifield-adaptive-retrieval/`):

```bash
# General analysis
python failure_analysis/general/failure_analysis.py

# Multi-hop analysis
python failure_analysis/multi-hop/multi_hop_analysis.py
python failure_analysis/multi-hop/multi_hop_analysis.py val test

# Negation ablation
python failure_analysis/negation/negation_ablation.py
python failure_analysis/negation/negation_ablation.py val

# Negation examples
python failure_analysis/negation/negation_examples.py
```

## Shared Utilities (`utils.py`)

- `load_corpus(data_dir)` / `load_corpus_full(data_dir)` — corpus with field metadata
- `load_queries(data_dir, split)` — query text
- `load_qrels(data_dir, split)` — relevance judgments
- `load_retrieved(path)` — retrieval results (qres format)
- `dcg(gains, k)` — DCG computation

## Data Dependencies

| File | Path |
|------|------|
| Corpus | `data/prime/corpus` |
| Queries | `data/prime/{val,test}.queries` |
| QRels | `data/prime/{val,test}.qrels` |
| Retrieval results (val) | `output/contriever/prime_eval/final-all-0.qres` |
| Retrieval results (test) | `output/contriever/prime_eval/final-additional-all-0.qres` |
| Ablation metrics | `output/contriever/prime_eval/results_dicts-all-0.jsonl` |
| Checkpoint | `output/contriever/prime/best.txt` |

## Analysis Summary

| Analysis | Key Question | Key Finding |
|----------|-------------|-------------|
| General | Where and why does mFAR fail? | Field sparsity, competing entity types, score gaps |
| Multi-hop | Does cross-type (multi-hop) hurt? | Cross-type: MRR -16.4%, Miss 1.2x vs same-type |
| Negation | Is explicit NOT handling needed? | Run `negation_ablation.py`, see `output/failure_analysis/negation/` |
