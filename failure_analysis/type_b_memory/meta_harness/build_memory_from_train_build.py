"""
Step 2: Build memory from train-build split.

Reuses existing Qwen3 stage12 cache (already split by split_cache.py),
runs extract_rerouted + build_memory_context restricted to train-build.

Usage:
    python failure_analysis/type_b_memory/meta_harness/build_memory_from_train_build.py
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
from failure_analysis.utils import load_qrels, RELATION_FIELDS
from failure_analysis.type_b_memory.rerank.train_memory.build_memory_context import (
    build_confusion, generate_memory_context, load_corpus_raw, classify_negation_pattern,
)

DATA_DIR = "data/prime"
CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"

SPLIT = "train-build"


def load_qwen3_cache_stage12(split):
    """Load Qwen3 cache from stage12/shared/ (where split_cache.py writes)."""
    cache_path = os.path.join(CACHE_DIR, "stage12", "shared", f"qwen3_cache_{split}.jsonl")
    if not os.path.exists(cache_path):
        print(f"  ERROR: {cache_path} not found. Run split_cache.py first.")
        sys.exit(1)
    cache = {}
    with open(cache_path) as f:
        for line in f:
            entry = json.loads(line)
            cache[entry["qid"]] = entry
    print(f"  Loaded {len(cache)} entries from {cache_path}")
    return cache


def extract_rerouted(cache, qrels):
    """Extract rerouted queries (needs_reroute=True + has boost/suppress fields)."""
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
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # 1. Load cache and qrels for train-build
    cache = load_qwen3_cache_stage12(SPLIT)
    qrels = load_qrels(DATA_DIR, SPLIT)

    # 2. Extract rerouted queries
    rerouted = extract_rerouted(cache, qrels)
    print(f"\n  Rerouted queries: {len(rerouted)} / {len(cache)}")

    # Save rerouted
    from collections import Counter
    output = {
        "split": SPLIT,
        "total_queries": len(cache),
        "negation_queries": sum(1 for e in cache.values() if e.get("needs_reroute")),
        "rerouted_count": len(rerouted),
        "rerouted_queries": rerouted,
    }
    rerouted_path = os.path.join(ANALYSIS_DIR, f"rerouted_{SPLIT}.json")
    with open(rerouted_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved rerouted → {rerouted_path}")

    # 3. Build field confusion
    print("\n  Loading corpus...")
    corpus_raw = load_corpus_raw(DATA_DIR)

    group_field_counts, group_doc_types, group_query_count, details = \
        build_confusion(rerouted, corpus_raw)

    # Save field confusion
    report = {
        "split": SPLIT,
        "rerouted_count": len(rerouted),
        "groups": {
            k: {
                "query_count": len(group_query_count[k]),
                "query_ids": sorted(group_query_count[k]),
                "gold_field_distribution": dict(sorted(
                    group_field_counts.get(k, {}).items(), key=lambda x: -x[1])),
                "gold_entity_types": dict(group_doc_types.get(k, {}).most_common()),
            }
            for k in sorted(group_query_count.keys(),
                            key=lambda k: -len(group_query_count[k]))
        },
        "per_query_details": details,
    }
    confusion_path = os.path.join(ANALYSIS_DIR, f"field_confusion_{SPLIT}.json")
    with open(confusion_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved field confusion → {confusion_path}")

    # 4. Generate memory context
    memory_ctx = generate_memory_context(
        group_field_counts, group_doc_types, group_query_count)
    ctx_path = os.path.join(ANALYSIS_DIR, f"memory_context_{SPLIT}.txt")
    with open(ctx_path, "w") as f:
        f.write(memory_ctx)
    print(f"  Saved memory context → {ctx_path}")

    # Summary
    print(f"\n  {'Group (answer_type|neg_pattern)':<45} {'N':>4} {'Top Gold Fields'}")
    print(f"  {'-'*90}")
    for group_key in sorted(group_query_count.keys(),
                            key=lambda k: -len(group_query_count[k])):
        count = len(group_query_count[group_key])
        if count < 3:
            continue
        fields = group_field_counts.get(group_key, {})
        total = sum(fields.values())
        top = sorted(fields.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{f}({100*c//total}%)" for f, c in top) if total else "-"
        print(f"  {group_key:<45} {count:>4} {top_str}")

    print(f"\n  Memory context preview:")
    for line in memory_ctx.split("\n")[:15]:
        print(f"    {line}")


if __name__ == "__main__":
    main()
