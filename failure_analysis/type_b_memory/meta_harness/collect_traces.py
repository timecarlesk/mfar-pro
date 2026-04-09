"""
Step 5: Collect execution traces for Meta-Harness proposer.

For each rerouted query in train-dev, records detailed information about
what happened during reranking: ranks before/after, LLM scores, document
formatting, gold doc field populations.

Usage:
    python failure_analysis/type_b_memory/meta_harness/collect_traces.py \
        --round 0 --split train-dev
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from failure_analysis.utils import (
    load_queries, load_qrels, load_retrieved, RELATION_FIELDS,
)
from failure_analysis.type_b_memory.meta_harness.config_rerank import (
    load_qwen3_cache, load_rerank_cache_full, load_corpus_docs,
    format_doc_with_config, BASELINE_QRES, BASE_CACHE_DIR, BASE_RUNS_DIR,
)
from failure_analysis.type_b_memory.meta_harness.harness_config import load_config

DATA_DIR = "data/prime"


def find_rank(docs, target_docids):
    """Find rank of first doc in target_docids. Returns (rank, docid, score) or None."""
    for rank, (docid, score) in enumerate(docs):
        if docid in target_docids:
            return rank, docid, score
    return None


def collect_traces(split, round_num, config_path=None):
    """Collect execution traces for all rerouted queries."""
    # Load data
    queries = load_queries(DATA_DIR, split)
    qrels = load_qrels(DATA_DIR, split)
    baseline_retrieved = load_retrieved(BASELINE_QRES[split])
    qwen3_cache = load_qwen3_cache(split)

    # Load reranked results
    reranked_qres = os.path.join(
        BASE_RUNS_DIR, "meta_harness", f"round_{round_num}", f"reranked_{split}.qres")
    if not os.path.exists(reranked_qres):
        print(f"  ERROR: {reranked_qres} not found. Run config_rerank first.")
        return []
    reranked_retrieved = load_retrieved(reranked_qres)

    # Load rerank scores (full entries with doc_text)
    tag = "qwen3_8b"  # default model tag
    cache_dir = os.path.join(BASE_CACHE_DIR, "meta_harness", tag, f"round_{round_num}")
    cache_path = os.path.join(cache_dir, f"rerank_cache_{split}.jsonl")
    rerank_full = load_rerank_cache_full(cache_path) if os.path.exists(cache_path) else {}

    # Load config
    if config_path:
        from failure_analysis.type_b_memory.meta_harness.harness_config import load_config
        config = load_config(config_path)
    else:
        from failure_analysis.type_b_memory.meta_harness.harness_config import baseline_config
        config = baseline_config()

    # Collect doc IDs we need from corpus
    rerouted_qids = {qid for qid, e in qwen3_cache.items()
                     if e.get("needs_reroute") and qid in queries}
    docid_set = set()
    for qid in rerouted_qids:
        gold = qrels.get(qid, set())
        docid_set.update(gold)
        docs = baseline_retrieved.get(qid, [])[:5]
        for docid, _ in docs:
            docid_set.add(docid)
    corpus = load_corpus_docs(DATA_DIR, docid_set)

    traces = []
    for qid in sorted(rerouted_qids):
        gold_ids = qrels.get(qid, set())
        if not gold_ids:
            continue

        qentry = qwen3_cache.get(qid, {})
        baseline_docs = baseline_retrieved.get(qid, [])[:100]
        reranked_docs = reranked_retrieved.get(qid, [])[:100]

        # Find gold rank before/after
        before = find_rank(baseline_docs, gold_ids)
        after = find_rank(reranked_docs, gold_ids)

        gold_rank_before = before[0] if before else -1
        gold_rank_after = after[0] if after else -1
        gold_docid = before[1] if before else (after[1] if after else None)

        if gold_docid is None:
            # Gold not in top-100 at all
            gold_docid = sorted(gold_ids)[0] if gold_ids else None

        # LLM scores
        llm_score_gold = None
        format_doc_gold = None
        if gold_docid:
            entry = rerank_full.get((qid, gold_docid))
            if entry:
                llm_score_gold = entry.get("llm_score")
                format_doc_gold = entry.get("doc_text")

        # Top-1 after reranking
        top1_docid = reranked_docs[0][0] if reranked_docs else None
        llm_score_top1 = None
        format_doc_top1 = None
        if top1_docid:
            entry = rerank_full.get((qid, top1_docid))
            if entry:
                llm_score_top1 = entry.get("llm_score")
                format_doc_top1 = entry.get("doc_text")

        # mFAR scores
        mfar_score_gold = None
        mfar_score_top1 = None
        if before:
            mfar_score_gold = before[2]
        if baseline_docs:
            mfar_score_top1 = baseline_docs[0][1]

        # Gold doc field populations
        gold_fields_populated = []
        if gold_docid and gold_docid in corpus:
            doc = corpus[gold_docid]
            for f in RELATION_FIELDS:
                val = doc.get(f)
                if val and isinstance(val, dict) and any(val.values()):
                    gold_fields_populated.append(f)

        trace = {
            "qid": qid,
            "query": queries.get(qid, ""),
            "negation_pattern": qentry.get("negation_pattern"),
            "answer_type": qentry.get("answer_type"),
            "boost_fields": qentry.get("boost_fields", []),
            "suppress_fields": qentry.get("suppress_fields", []),
            "gold_docid": gold_docid,
            "gold_rank_before": gold_rank_before,
            "gold_rank_after": gold_rank_after,
            "improved": gold_rank_after < gold_rank_before if (gold_rank_before >= 0 and gold_rank_after >= 0) else False,
            "llm_score_gold": llm_score_gold,
            "llm_score_top1": llm_score_top1,
            "mfar_score_gold": mfar_score_gold,
            "mfar_score_top1": mfar_score_top1,
            "format_doc_gold": format_doc_gold,
            "format_doc_top1": format_doc_top1,
            "gold_doc_fields_populated": gold_fields_populated,
        }
        traces.append(trace)

    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--split", default="train-dev")
    parser.add_argument("--config", default=None, help="Path to HarnessConfig JSON")
    args = parser.parse_args()

    traces = collect_traces(args.split, args.round, args.config)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"traces_round_{args.round}.json")
    with open(out_path, "w") as f:
        json.dump(traces, f, indent=2)

    # Summary
    improved = sum(1 for t in traces if t["improved"])
    worsened = sum(1 for t in traces
                   if t["gold_rank_before"] >= 0 and t["gold_rank_after"] >= 0
                   and t["gold_rank_after"] > t["gold_rank_before"])
    gold_missing = sum(1 for t in traces if t["gold_rank_before"] < 0)

    print(f"\n  Traces: {len(traces)} rerouted queries")
    print(f"  Improved:     {improved}")
    print(f"  Worsened:     {worsened}")
    print(f"  Unchanged:    {len(traces) - improved - worsened - gold_missing}")
    print(f"  Gold missing: {gold_missing} (not in top-100)")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
