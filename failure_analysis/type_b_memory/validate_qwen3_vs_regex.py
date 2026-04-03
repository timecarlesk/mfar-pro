"""
Validate Qwen3 field routing against gold doc populated fields.

For queries where Qwen3 recommends re-routing (score > 0, boost_fields non-empty):
- Precision: what fraction of Qwen3's boost_fields are actually populated in gold docs?
- Recall: what fraction of gold doc populated fields are in Qwen3's boost_fields?

Also compares Qwen3 negation detection against regex baseline for backwards compat.

Run from project root:
  python failure_analysis/type_b_memory/validate_qwen3_vs_regex.py
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from failure_analysis.utils import load_queries, load_qrels, RELATION_FIELDS
from failure_analysis.negation.negation_ablation import NEGATION_PATTERN, classify_negation

DATA_DIR = "data/prime"
CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"


def load_qwen3_cache(split):
    cache_path = os.path.join(CACHE_DIR, f"qwen3_cache_{split}.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["qid"]] = entry
    return cache


def load_corpus_raw(data_dir):
    """Load corpus with full JSON for field population checks."""
    corpus = {}
    with open(f"{data_dir}/corpus") as f:
        for line in f:
            idx, json_str = line.strip().split("\t", 1)
            doc = json.loads(json_str)
            corpus[idx] = doc
    print(f"  Loaded {len(corpus):,} corpus documents")
    return corpus


def get_populated_fields(doc):
    """Get the set of relation fields that are non-empty in a doc."""
    populated = set()
    for field in RELATION_FIELDS:
        val = doc.get(field)
        if val is None:
            continue
        if isinstance(val, (dict, list)) and len(val) > 0:
            populated.add(field)
        elif isinstance(val, str) and val.strip():
            populated.add(field)
    return populated


def main():
    parser = argparse.ArgumentParser(description="Validate Qwen3 field routing")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum boost precision to pass (default: 0.5)")
    args = parser.parse_args()

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("Loading corpus...")
    corpus_raw = load_corpus_raw(DATA_DIR)

    for split in args.splits:
        print(f"\n{'='*70}")
        print(f"  Validating Qwen3 for split: {split}")
        print(f"{'='*70}")

        queries = load_queries(DATA_DIR, split)
        qrels = load_qrels(DATA_DIR, split)
        qwen3_cache = load_qwen3_cache(split)

        if not qwen3_cache:
            print(f"  ERROR: No Qwen3 cache for {split}. Run batch_qwen3_inference.py first.")
            continue

        # ── Part 1: Boost/suppress precision & recall vs gold doc fields ──

        precision_list = []
        recall_list = []
        reroute_count = 0
        examples = []

        for qid in sorted(queries.keys()):
            entry = qwen3_cache.get(qid, {})
            boost = entry.get("boost_fields") or []
            suppress = entry.get("suppress_fields") or []

            if not entry.get("needs_reroute") or not boost:
                continue  # no re-routing recommended

            reroute_count += 1
            gold_ids = qrels.get(qid, set())

            # Union of populated fields across all gold docs
            gold_populated = set()
            for doc_id in gold_ids:
                if doc_id in corpus_raw:
                    gold_populated |= get_populated_fields(corpus_raw[doc_id])

            if not gold_populated:
                continue

            # Precision: how many boost fields are in gold populated?
            boost_set = set(boost)
            tp = len(boost_set & gold_populated)
            prec = tp / len(boost_set) if boost_set else 0
            precision_list.append(prec)

            # Recall: how many gold populated fields are in boost?
            rec = tp / len(gold_populated) if gold_populated else 0
            recall_list.append(rec)

            # Collect low-precision examples for debugging
            if prec < 0.5:
                examples.append({
                    "qid": qid,
                    "query": queries[qid][:100],
                    "boost": boost,
                    "suppress": suppress,
                    "gold_populated": sorted(gold_populated),
                    "precision": round(prec, 2),
                })

        n = len(precision_list)
        avg_prec = sum(precision_list) / n if n else 0
        avg_rec = sum(recall_list) / n if n else 0

        print(f"\n  Boost Field Validation (queries with re-routing):")
        print(f"    Queries with re-routing:  {reroute_count}")
        print(f"    Evaluated (have gold):    {n}")
        print(f"    Avg boost precision:      {avg_prec:.3f}")
        print(f"    Avg boost recall:         {avg_rec:.3f}")

        if examples:
            print(f"\n  Low-precision examples (top 5):")
            for ex in examples[:5]:
                print(f"    [{ex['qid']}] prec={ex['precision']}")
                print(f"      query: {ex['query']}")
                print(f"      boost: {ex['boost']}")
                print(f"      gold:  {ex['gold_populated'][:8]}...")

        # ── Part 2: Negation detection agreement with regex ──

        neg_agree = 0
        total = 0
        for qid in queries:
            if qid not in qwen3_cache:
                continue
            total += 1
            has_neg_regex = bool(NEGATION_PATTERN.search(queries[qid]))
            # In new format: "has re-routing" ≈ "has negation" for comparison
            qwen3_entry = qwen3_cache[qid]
            has_neg_qwen3 = qwen3_entry.get("needs_reroute", False)
            if has_neg_regex == has_neg_qwen3:
                neg_agree += 1

        neg_rate = neg_agree / total if total else 0
        print(f"\n  Regex negation vs Qwen3 re-routing overlap:")
        print(f"    Agreement: {neg_agree}/{total} = {neg_rate:.1%}")
        print(f"    (Note: Qwen3 may reroute non-negation queries too — this is expected)")

        # ── Part 3: Suppress field check ──

        suppress_in_gold = 0
        suppress_total = 0
        for qid in queries:
            entry = qwen3_cache.get(qid, {})
            suppress = entry.get("suppress_fields") or []
            if not suppress or not entry.get("needs_reroute"):
                continue
            gold_ids = qrels.get(qid, set())
            gold_populated = set()
            for doc_id in gold_ids:
                if doc_id in corpus_raw:
                    gold_populated |= get_populated_fields(corpus_raw[doc_id])
            suppress_set = set(suppress)
            suppress_total += 1
            if suppress_set & gold_populated:
                suppress_in_gold += 1

        if suppress_total:
            suppress_rate = suppress_in_gold / suppress_total
            print(f"\n  Suppress field sanity check:")
            print(f"    Queries where suppress field IS in gold: {suppress_in_gold}/{suppress_total} = {suppress_rate:.1%}")
            print(f"    (Lower is better — we want to suppress fields NOT in gold)")

        # Save report
        report = {
            "split": split,
            "reroute_count": reroute_count,
            "evaluated": n,
            "avg_boost_precision": round(avg_prec, 4),
            "avg_boost_recall": round(avg_rec, 4),
            "regex_agreement": round(neg_rate, 4),
            "suppress_in_gold_rate": round(suppress_in_gold / suppress_total, 4) if suppress_total else None,
        }
        out_path = os.path.join(ANALYSIS_DIR, f"qwen3_validation_{split}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
