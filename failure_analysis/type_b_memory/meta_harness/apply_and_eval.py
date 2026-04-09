"""
Step 7: Apply a HarnessConfig and evaluate MRR delta with bootstrap CI.

Applies the proposed config, runs rerank scoring + merging on train-dev,
computes MRR, and decides accept/reject based on the hard rule:
  reject if MRR delta < 0.5 × bootstrap CI width

Also runs val pilot (100 random rerouted queries) to monitor transfer.

Usage:
    python failure_analysis/type_b_memory/meta_harness/apply_and_eval.py \
        --config configs/round_1.json \
        --round 1 \
        --split train-dev \
        --model qwen3:8b \
        --endpoints http://127.0.0.1:11434
"""

import argparse
import json
import math
import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from failure_analysis.utils import load_queries, load_qrels, load_retrieved
from failure_analysis.type_b_memory.meta_harness.harness_config import load_config
from failure_analysis.type_b_memory.meta_harness.config_rerank import (
    batch_score_with_config, merge_with_config, load_qwen3_cache,
    BASELINE_QRES, BASE_RUNS_DIR,
)

DATA_DIR = "data/prime"


def compute_mrr_for_qids(qids, qrels, retrieved):
    """Compute MRR for a set of query IDs."""
    rr_list = []
    for qid in qids:
        gold = qrels.get(qid, set())
        if not gold:
            continue
        docs = retrieved.get(qid, [])[:100]
        rr = 0.0
        for rank, (docid, _) in enumerate(docs):
            if docid in gold:
                rr = 1.0 / (rank + 1)
                break
        rr_list.append(rr)
    return rr_list


def bootstrap_ci(rr_list, n_bootstrap=5000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for MRR."""
    if not rr_list:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(rr_list)
    means = []
    for _ in range(n_bootstrap):
        sample = [rr_list[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((1 - ci) / 2 * n_bootstrap)
    upper_idx = int((1 + ci) / 2 * n_bootstrap)
    mrr = sum(rr_list) / n
    return mrr, means[lower_idx], means[upper_idx]


def evaluate_round(split, round_num, baseline_round=0):
    """Evaluate a round's results vs baseline. Returns metrics dict."""
    qrels = load_qrels(DATA_DIR, split)
    qwen3_cache = load_qwen3_cache(split)
    rerouted_qids = [qid for qid, e in qwen3_cache.items() if e.get("needs_reroute")]

    # Load baseline results
    baseline_qres = os.path.join(
        BASE_RUNS_DIR, "meta_harness", f"round_{baseline_round}", f"reranked_{split}.qres")
    if not os.path.exists(baseline_qres):
        # Fall back to mFAR baseline
        baseline_qres = BASELINE_QRES[split]
    baseline_ret = load_retrieved(baseline_qres)

    # Load this round's results
    round_qres = os.path.join(
        BASE_RUNS_DIR, "meta_harness", f"round_{round_num}", f"reranked_{split}.qres")
    round_ret = load_retrieved(round_qres)

    # Compute MRR on rerouted queries
    baseline_rr = compute_mrr_for_qids(rerouted_qids, qrels, baseline_ret)
    round_rr = compute_mrr_for_qids(rerouted_qids, qrels, round_ret)

    base_mrr, base_lo, base_hi = bootstrap_ci(baseline_rr)
    round_mrr, round_lo, round_hi = bootstrap_ci(round_rr)

    delta = round_mrr - base_mrr
    ci_width = base_hi - base_lo  # baseline CI width as reference

    return {
        "split": split,
        "round": round_num,
        "baseline_round": baseline_round,
        "n_rerouted": len(rerouted_qids),
        "n_with_gold": len(baseline_rr),
        "baseline_mrr": round(base_mrr, 4),
        "baseline_ci": [round(base_lo, 4), round(base_hi, 4)],
        "round_mrr": round(round_mrr, 4),
        "round_ci": [round(round_lo, 4), round(round_hi, 4)],
        "delta": round(delta, 4),
        "ci_width": round(ci_width, 4),
        "threshold": round(0.5 * ci_width, 4),
        "accept": delta > 0.5 * ci_width,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to HarnessConfig JSON")
    parser.add_argument("--round", type=int, required=True, help="Round number")
    parser.add_argument("--split", default="train-dev")
    parser.add_argument("--baseline_round", type=int, default=0)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--endpoints", default="http://127.0.0.1:11434")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--score_only", action="store_true",
                        help="Only score, don't merge or evaluate")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate (assumes scoring and merging already done)")
    args = parser.parse_args()

    config = load_config(args.config)
    endpoints = [ep.strip() for ep in args.endpoints.split(",")]
    workers = args.workers or len(endpoints)

    if not args.eval_only:
        # Phase 1: Score
        print(f"\n{'='*60}")
        print(f"  Phase 1: Scoring round {args.round} on {args.split}")
        print(f"{'='*60}")
        rerank_scores, cache_path = batch_score_with_config(
            args.split, args.model, endpoints, workers, config, args.round)

        if args.score_only:
            print("  Score-only mode, stopping here.")
            return

        # Phase 2: Merge
        print(f"\n{'='*60}")
        print(f"  Phase 2: Merging round {args.round}")
        print(f"{'='*60}")
        qwen3_cache = load_qwen3_cache(args.split)
        merge_with_config(args.split, rerank_scores, qwen3_cache, config, args.round)

    # Phase 3: Evaluate
    print(f"\n{'='*60}")
    print(f"  Phase 3: Evaluating round {args.round} vs baseline (round {args.baseline_round})")
    print(f"{'='*60}")

    metrics = evaluate_round(args.split, args.round, args.baseline_round)

    print(f"\n  Results on {args.split} rerouted queries ({metrics['n_with_gold']} with gold):")
    print(f"    Baseline MRR: {metrics['baseline_mrr']:.4f} [{metrics['baseline_ci'][0]:.4f}, {metrics['baseline_ci'][1]:.4f}]")
    print(f"    Round {args.round} MRR: {metrics['round_mrr']:.4f} [{metrics['round_ci'][0]:.4f}, {metrics['round_ci'][1]:.4f}]")
    print(f"    Delta:        {metrics['delta']:+.4f}")
    print(f"    CI width:     {metrics['ci_width']:.4f}")
    print(f"    Threshold:    {metrics['threshold']:.4f} (0.5 × CI width)")
    print(f"    Decision:     {'ACCEPT ✓' if metrics['accept'] else 'REJECT ✗'}")

    # Save metrics
    metrics_dir = os.path.join(os.path.dirname(__file__), "configs")
    metrics_path = os.path.join(metrics_dir, f"metrics_round_{args.round}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    # Val pilot monitoring (only if main split is NOT val)
    if args.split not in ("val", "test") and not args.eval_only:
        print(f"\n  Val pilot monitoring...")
        try:
            val_metrics = evaluate_round("val", args.round, args.baseline_round)
            print(f"    Val rerouted MRR: {val_metrics['round_mrr']:.4f} (delta: {val_metrics['delta']:+.4f})")
            val_path = os.path.join(metrics_dir, f"val_pilot_round_{args.round}.json")
            with open(val_path, "w") as f:
                json.dump(val_metrics, f, indent=2)
        except Exception as e:
            print(f"    Val pilot skipped: {e}")

    return metrics


if __name__ == "__main__":
    main()
