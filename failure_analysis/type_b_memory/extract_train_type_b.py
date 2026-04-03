"""
Extract queries with field re-routing from Qwen3 cache.

Filters queries where needs_reroute=True and boost_fields non-empty,
and enriches with gold doc IDs from qrels.

Run from project root:
  python failure_analysis/type_b_memory/extract_train_type_b.py
  python failure_analysis/type_b_memory/extract_train_type_b.py --splits val test
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from failure_analysis.utils import load_queries, load_qrels

DATA_DIR = "data/prime"
CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"


def load_qwen3_cache(split):
    """Load Qwen3 classification cache for a split."""
    cache_path = os.path.join(CACHE_DIR, f"qwen3_cache_{split}.jsonl")
    if not os.path.exists(cache_path):
        print(f"  ERROR: Cache not found at {cache_path}")
        print(f"  Run failure_analysis/type_b_memory/batch_qwen3_inference.py --splits {split} first")
        return {}
    cache = {}
    with open(cache_path) as f:
        for line in f:
            entry = json.loads(line)
            cache[entry["qid"]] = entry
    print(f"  Loaded {len(cache)} cached classifications for {split}")
    return cache


def extract_rerouted(cache, qrels):
    """Extract queries where Qwen3 recommends field re-routing."""
    rerouted = []
    for qid, entry in cache.items():
        if not entry.get("needs_reroute"):
            continue
        boost = entry.get("boost_fields") or []
        suppress = entry.get("suppress_fields") or []
        if not boost and not suppress:
            continue

        rerouted.append({
            "qid": qid,
            "query": entry.get("query", ""),
            "answer_type": entry.get("answer_type"),
            "boost_fields": boost,
            "suppress_fields": suppress,
            "gold_ids": sorted(qrels.get(qid, set())),
        })
    return rerouted


def main():
    parser = argparse.ArgumentParser(description="Extract rerouted queries from Qwen3 cache")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Splits to process (default: train)")
    args = parser.parse_args()

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"  Extracting rerouted queries for split: {split}")
        print(f"{'='*60}")

        cache = load_qwen3_cache(split)
        if not cache:
            continue

        qrels = load_qrels(DATA_DIR, split)
        rerouted = extract_rerouted(cache, qrels)

        # Summary
        total = len(cache)
        negation = sum(1 for e in cache.values() if e.get("needs_reroute"))
        boost_counts = Counter()
        suppress_counts = Counter()
        answer_types = Counter()
        for e in rerouted:
            for f in e["boost_fields"]:
                boost_counts[f] += 1
            for f in e["suppress_fields"]:
                suppress_counts[f] += 1
            answer_types[e.get("answer_type", "unknown")] += 1

        output = {
            "split": split,
            "total_queries": total,
            "negation_queries": negation,
            "rerouted_count": len(rerouted),
            "boost_field_counts": dict(boost_counts.most_common()),
            "suppress_field_counts": dict(suppress_counts.most_common()),
            "answer_type_counts": dict(answer_types.most_common()),
            "rerouted_queries": rerouted,
        }

        out_path = os.path.join(ANALYSIS_DIR, f"rerouted_{split}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved {len(rerouted)} rerouted queries to {out_path}")

        print(f"\n  Summary:")
        print(f"    Total queries:    {total}")
        print(f"    Negation (regex): {negation}")
        print(f"    Rerouted:         {len(rerouted)}")
        print(f"    Top boost fields:")
        for f, c in boost_counts.most_common(5):
            print(f"      {f}: {c}")
        print(f"    Top suppress fields:")
        for f, c in suppress_counts.most_common(5):
            print(f"      {f}: {c}")


if __name__ == "__main__":
    main()
