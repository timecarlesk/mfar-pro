"""
Evaluate Negation Memory Impact.

Compares baseline mFAR retrieval results vs memory-augmented results,
broken down by negation type (A/B/C) and field pair subtype.

Run from project root:
  python failure_analysis/type_b_memory/evaluate_memory.py \
    --baseline_dir output/prime_eval \
    --memory_dir output/prime_eval_negmem \
    --cache_dir output/failure_analysis/type_b_memory
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from failure_analysis.utils import load_queries, load_qrels, load_retrieved, dcg

DATA_DIR = "data/prime"
CACHE_DIR = "output/failure_analysis/type_b_memory/cache"

SPLIT_QRES = {
    "val": "final-all-0.qres",
    "test": "final-additional-all-0.qres",
}


def load_qwen3_cache(split):
    """Load Qwen3 classification cache."""
    # Try model-specific path first, then fallback
    cache_path = os.path.join(CACHE_DIR, "qwen3_8b", f"qwen3_cache_{split}.jsonl")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(CACHE_DIR, f"qwen3_cache_{split}.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["qid"]] = entry
    return cache


def compute_metrics(qids, qrels, retrieved):
    """Compute MRR, Hit@1, Hit@5, Miss% for a set of query IDs."""
    rr_list = []
    hit1 = hit5 = miss = 0
    ndcg10_list = []

    for qid in qids:
        gold = qrels.get(qid, set())
        if not gold:
            continue
        docs = retrieved.get(qid, [])
        top100 = docs[:100]

        # Reciprocal rank
        rr = 0.0
        found = False
        for rank, (docid, _) in enumerate(top100):
            if docid in gold:
                rr = 1.0 / (rank + 1)
                if rank == 0:
                    hit1 += 1
                if rank < 5:
                    hit5 += 1
                found = True
                break
        if not found:
            miss += 1
        rr_list.append(rr)

        # NDCG@10
        gains = [1.0 if docid in gold else 0.0 for docid, _ in top100[:10]]
        ideal = sorted(gains, reverse=True)
        d = dcg(gains, 10)
        id_ = dcg(ideal, 10)
        ndcg10_list.append(d / id_ if id_ > 0 else 0.0)

    n = len(rr_list)
    if n == 0:
        return {"count": 0, "mrr": 0, "hit1": 0, "hit5": 0, "miss_pct": 0, "ndcg10": 0}

    return {
        "count": n,
        "mrr": round(sum(rr_list) / n, 4),
        "hit1": round(hit1 / n * 100, 1),
        "hit5": round(hit5 / n * 100, 1),
        "miss_pct": round(miss / n * 100, 1),
        "ndcg10": round(sum(ndcg10_list) / n, 4),
    }


def classify_queries(qids, qwen3_cache):
    """Classify queries into groups based on Qwen3 cache."""
    groups = defaultdict(list)
    for qid in qids:
        entry = qwen3_cache.get(qid, {})
        needs_reroute = entry.get("needs_reroute", False)
        boost = entry.get("boost_fields") or []
        suppress = entry.get("suppress_fields") or []
        has_reroute = needs_reroute and (boost or suppress)

        if has_reroute:
            label = ",".join(sorted(suppress)) if suppress else "boost_only"
            groups[f"reroute:{label}"].append(qid)
            groups["reroute:ALL"].append(qid)
        elif needs_reroute:
            groups["negation_no_reroute"].append(qid)
        else:
            groups["no_negation"].append(qid)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Evaluate negation memory impact")
    parser.add_argument("--baseline_dir", default="output/prime_eval",
                        help="Directory with baseline .qres files")
    parser.add_argument("--memory_dir", default="output/prime_eval_negmem",
                        help="Directory with memory-augmented .qres files")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    os.makedirs(args.memory_dir, exist_ok=True)

    results = {}
    for split in args.splits:
        print(f"\n{'='*70}")
        print(f"  Evaluating split: {split}")
        print(f"{'='*70}")

        queries = load_queries(DATA_DIR, split)
        qrels = load_qrels(DATA_DIR, split)
        qwen3_cache = load_qwen3_cache(split)

        qres_name = SPLIT_QRES.get(split)
        if not qres_name:
            print(f"  Unknown split: {split}")
            continue

        baseline_path = os.path.join(args.baseline_dir, qres_name)
        memory_path = os.path.join(args.memory_dir, qres_name)

        if not os.path.exists(baseline_path):
            print(f"  Baseline not found: {baseline_path}")
            continue
        if not os.path.exists(memory_path):
            print(f"  Memory-augmented not found: {memory_path}")
            continue

        baseline_ret = load_retrieved(baseline_path)
        memory_ret = load_retrieved(memory_path)

        # Classify queries
        groups = classify_queries(queries.keys(), qwen3_cache)

        # Compute metrics per group
        split_results = {}
        print(f"\n  {'Group':<35} {'N':>5} {'Base MRR':>10} {'Mem MRR':>10} {'Delta':>8} {'Base Miss%':>10} {'Mem Miss%':>10}")
        print(f"  {'-'*90}")

        for group_name in sorted(groups.keys()):
            qids = groups[group_name]
            base_m = compute_metrics(qids, qrels, baseline_ret)
            mem_m = compute_metrics(qids, qrels, memory_ret)
            delta = round(mem_m["mrr"] - base_m["mrr"], 4)
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"

            print(f"  {group_name:<35} {base_m['count']:>5} {base_m['mrr']:>10.4f} {mem_m['mrr']:>10.4f} {delta_str:>8} {base_m['miss_pct']:>9.1f}% {mem_m['miss_pct']:>9.1f}%")

            split_results[group_name] = {
                "baseline": base_m,
                "memory": mem_m,
                "delta_mrr": delta,
            }

        # Overall
        all_qids = list(queries.keys())
        base_all = compute_metrics(all_qids, qrels, baseline_ret)
        mem_all = compute_metrics(all_qids, qrels, memory_ret)
        delta_all = round(mem_all["mrr"] - base_all["mrr"], 4)
        print(f"\n  {'OVERALL':<35} {base_all['count']:>5} {base_all['mrr']:>10.4f} {mem_all['mrr']:>10.4f} {'+' if delta_all >= 0 else ''}{delta_all:.4f}")

        split_results["OVERALL"] = {
            "baseline": base_all,
            "memory": mem_all,
            "delta_mrr": delta_all,
        }

        results[split] = split_results

    # Save results
    out_path = os.path.join(args.memory_dir, "memory_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
