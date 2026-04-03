"""
LLM Re-Ranking for Negation Query Failures.

Takes mFAR's baseline top-K results, uses Qwen3 to score each
(query, document) pair for negation-aware relevance, re-ranks.

Only applies to queries flagged by Stage 1 (needs_reroute=True).
Other queries keep baseline rankings unchanged.

Two phases:
  Phase 1: Score all (rerouted_query, top_k_doc) pairs via Qwen3 → cache
  Phase 2: Merge LLM scores with mFAR scores at various α → write .qres

Run from project root:
  # Phase 1: score (expensive, ~26 min with 6 endpoints)
  python failure_analysis/type_b_memory/rerank_negation.py score \
    --splits val --endpoints http://127.0.0.1:11434,...

  # Phase 2: merge + evaluate (instant, sweep α)
  python failure_analysis/type_b_memory/rerank_negation.py merge \
    --splits val --alpha_sweep 0.0 0.3 0.5 0.7 1.0

  # Pilot: score + check 50 queries interactively
  python failure_analysis/type_b_memory/rerank_negation.py pilot --splits val
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from failure_analysis.utils import load_queries, load_qrels, load_retrieved, RELATION_FIELDS

# Reuse Ollama caller from batch_qwen3_inference
from failure_analysis.type_b_memory.batch_qwen3_inference import call_ollama

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = "data/prime"
BASE_CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
BASE_RUNS_DIR = "output/failure_analysis/type_b_memory/runs"


def _model_tag(model_name):
    """Convert model name to folder-safe tag: 'qwen3:8b' → 'qwen3_8b'."""
    return model_name.replace(":", "_").replace("/", "_")

BASELINE_QRES = {
    "val": "output/prime_eval/final-all-0.qres",
    "test": "output/prime_eval/final-additional-all-0.qres",
}

RERANKED_QRES = {
    "val": "final-all-0.qres",
    "test": "final-additional-all-0.qres",
}


# ── Corpus Loading ───────────────────────────────────────────────────────────

def load_corpus_docs(data_dir, docid_set):
    """Load full JSON for a set of doc IDs."""
    corpus = {}
    with open(f"{data_dir}/corpus") as f:
        for line in f:
            idx, json_str = line.strip().split("\t", 1)
            if idx in docid_set:
                corpus[idx] = json.loads(json_str)
    print(f"  Loaded {len(corpus)}/{len(docid_set)} corpus documents")
    return corpus


# ── Document Formatting ──────────────────────────────────────────────────────

def format_doc(doc_json, boost_fields=None, max_items=5):
    """Create compact doc string for LLM prompt.

    Prioritizes boost_fields (from Stage 2 memory) for focused evidence.
    """
    name = doc_json.get("name", "?")
    dtype = doc_json.get("type", "?")
    parts = [f"{name} ({dtype})"]

    shown = set()
    if boost_fields:
        for field in boost_fields:
            val = doc_json.get(field)
            if val and isinstance(val, dict):
                items = []
                for subtype, lst in val.items():
                    if isinstance(lst, list):
                        items.extend(lst[:max_items])
                if items:
                    parts.append(f"{field}: {', '.join(str(x) for x in items[:max_items])}")
                    shown.add(field)

    for field in RELATION_FIELDS:
        if field in shown:
            continue
        val = doc_json.get(field)
        if val and isinstance(val, dict) and any(val.values()):
            parts.append(f"{field}: [has data]")

    return "; ".join(parts)


# ── Re-Ranking Prompt ────────────────────────────────────────────────────────

RERANK_PROMPT = """\
Query: "{query}"
Document: {doc_text}

This query contains negation or constraints. Score how well this document satisfies the query INCLUDING any negation constraints (e.g., "not indicated" means the document should have contraindication data, not indication data).

Score 0-10 where 0=completely irrelevant, 10=perfect match.
Output ONLY the number."""


def parse_rerank_score(raw_output):
    """Parse LLM output to a float score 0-10."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Extract first number
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if match:
        score = float(match.group(1))
        return max(0.0, min(10.0, score))
    return 5.0  # neutral fallback


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_one(qid, docid, query, doc_json, boost_fields, model, endpoint):
    """Score a single (query, doc) pair via Qwen3."""
    doc_text = format_doc(doc_json, boost_fields)
    prompt = RERANK_PROMPT.replace("{query}", query).replace("{doc_text}", doc_text)
    try:
        raw = call_ollama(prompt, model, endpoint, max_tokens=5)
        score = parse_rerank_score(raw)
        return {"qid": qid, "docid": docid, "llm_score": score, "raw": raw}
    except Exception as e:
        return {"qid": qid, "docid": docid, "llm_score": 5.0, "raw": str(e)}


def load_rerank_cache(cache_path):
    """Load existing rerank cache. Returns {(qid, docid): score}."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[(entry["qid"], entry["docid"])] = entry["llm_score"]
    return cache


def _load_qwen3_cache(split, detect_model="qwen3_8b"):
    """Load Qwen3 Stage 1+2 cache."""
    qwen3_cache = {}
    qwen3_path = os.path.join(BASE_CACHE_DIR, detect_model, f"qwen3_cache_{split}.jsonl")
    if not os.path.exists(qwen3_path):
        # Fallback to old path
        qwen3_path = os.path.join(BASE_CACHE_DIR, f"qwen3_cache_{split}.jsonl")
    with open(qwen3_path) as f:
        for line in f:
            e = json.loads(line)
            qwen3_cache[e["qid"]] = e
    return qwen3_cache


def batch_score(split, model, endpoints, workers, top_k=50):
    """Score all (rerouted_query, top_k_doc) pairs via Qwen3. Cache as JSONL."""
    tag = _model_tag(model)
    queries = load_queries(DATA_DIR, split)
    baseline_path = BASELINE_QRES[split]
    retrieved = load_retrieved(baseline_path)

    # Load Qwen3 Stage 1+2 cache for needs_reroute and boost_fields
    qwen3_cache = _load_qwen3_cache(split)

    # Collect (qid, docid) pairs to score
    rerouted_qids = {qid for qid, e in qwen3_cache.items() if e.get("needs_reroute")}
    pairs = []
    docid_set = set()
    for qid in rerouted_qids:
        docs = retrieved.get(qid, [])[:top_k]
        for docid, _ in docs:
            pairs.append((qid, docid))
            docid_set.add(docid)

    print(f"  Model: {model} (tag: {tag})")
    print(f"  Rerouted queries: {len(rerouted_qids)}")
    print(f"  Pairs to score: {len(pairs)}")
    print(f"  Unique docs: {len(docid_set)}")

    # Load corpus for needed docs
    print("  Loading corpus...")
    corpus = load_corpus_docs(DATA_DIR, docid_set)

    # Load existing cache (model-specific)
    rerank_cache_dir = os.path.join(BASE_CACHE_DIR, "rerank", tag)
    os.makedirs(rerank_cache_dir, exist_ok=True)
    cache_path = os.path.join(rerank_cache_dir, f"rerank_cache_{split}.jsonl")
    existing = load_rerank_cache(cache_path)
    remaining = [(qid, did) for qid, did in pairs if (qid, did) not in existing]
    print(f"  Cache has {len(existing)} entries, {len(remaining)} remaining")

    if not remaining:
        print("  All pairs already scored")
        return existing

    # Score with thread pool
    write_lock = threading.Lock()
    done = [0]

    with open(cache_path, "a") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, (qid, docid) in enumerate(remaining):
                ep = endpoints[i % len(endpoints)]
                qentry = qwen3_cache.get(qid, {})
                boost = qentry.get("boost_fields", [])
                suppress = qentry.get("suppress_fields", [])
                doc_json = corpus.get(docid, {"name": "?", "type": "?"})
                fut = pool.submit(score_one, qid, docid, queries[qid], doc_json, boost, model, ep)
                futures[fut] = (qid, docid)

            for future in as_completed(futures):
                entry = future.result()
                with write_lock:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    existing[(entry["qid"], entry["docid"])] = entry["llm_score"]
                    done[0] += 1
                    if done[0] % 500 == 0:
                        print(f"  Scored {done[0]}/{len(remaining)}")

    print(f"  Done: {len(existing)} total scored")
    return existing


# ── Score Merging ────────────────────────────────────────────────────────────

def merge_and_write_qres(split, rerank_scores, qwen3_cache, alpha=0.5, top_k=50, output_dir=None):
    """Merge LLM scores with mFAR scores, write new .qres file.

    For rerouted queries: final = α * norm(llm) + (1-α) * norm(mfar) within top-K
    For non-rerouted queries: copy baseline unchanged.
    """
    baseline_path = BASELINE_QRES[split]
    retrieved = load_retrieved(baseline_path)

    rerouted_qids = {qid for qid, e in qwen3_cache.items() if e.get("needs_reroute")}

    if output_dir is None:
        output_dir = os.path.join(RUNS_DIR, f"rerank_alpha_{alpha}_top{top_k}")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, RERANKED_QRES[split])
    reranked_count = 0
    copied_count = 0

    with open(output_path, "w") as f:
        for qid in sorted(retrieved.keys()):
            docs = retrieved[qid]  # [(docid, score), ...]

            if qid not in rerouted_qids:
                # Copy baseline unchanged
                for rank, (docid, score) in enumerate(docs[:100]):
                    f.write(f"{qid}\t0\t{docid}\t{rank}\t{score}\t0\n")
                copied_count += 1
                continue

            # Re-rank top-K
            top_docs = docs[:top_k]
            rest_docs = docs[top_k:100]

            # Collect mFAR and LLM scores for top-K
            mfar_scores = [score for _, score in top_docs]
            llm_scores = [rerank_scores.get((qid, docid), 5.0) for docid, _ in top_docs]

            # Min-max normalize within query
            mfar_norm = _minmax_normalize(mfar_scores)
            llm_norm = _minmax_normalize(llm_scores)

            # Hybrid score
            final_scores = [alpha * l + (1 - alpha) * m for l, m in zip(llm_norm, mfar_norm)]

            # Sort top-K by final score
            indexed = list(zip(range(len(top_docs)), final_scores))
            indexed.sort(key=lambda x: -x[1])

            # Write re-ranked top-K
            min_reranked_score = min(final_scores) if final_scores else 0
            for rank, (orig_idx, fscore) in enumerate(indexed):
                docid = top_docs[orig_idx][0]
                # Scale final score to be above rest_docs scores
                f.write(f"{qid}\t0\t{docid}\t{rank}\t{fscore + 100}\t0\n")

            # Write remaining docs below re-ranked set
            for rank_offset, (docid, score) in enumerate(rest_docs):
                rank = top_k + rank_offset
                f.write(f"{qid}\t0\t{docid}\t{rank}\t{score}\t0\n")

            reranked_count += 1

    print(f"  Wrote {output_path}: {reranked_count} reranked, {copied_count} copied")
    return output_path


def _minmax_normalize(scores):
    """Normalize scores to [0, 1] via min-max."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-8:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


# ── Pilot ────────────────────────────────────────────────────────────────────

def run_pilot(split, model, endpoint, n_queries=50):
    """Score a small set of queries and print detailed results for manual inspection."""
    queries = load_queries(DATA_DIR, split)
    qrels = load_qrels(DATA_DIR, split)
    retrieved = load_retrieved(BASELINE_QRES[split])

    qwen3_cache = _load_qwen3_cache(split)

    # Find rerouted queries where gold is in top-50 but NOT rank 0
    # These are the hard cases where reranking can actually help
    candidates = []
    for qid, entry in qwen3_cache.items():
        if not entry.get("needs_reroute"):
            continue
        gold = qrels.get(qid, set())
        docs = retrieved.get(qid, [])[:50]
        gold_rank = None
        for rank, (docid, _) in enumerate(docs):
            if docid in gold:
                gold_rank = rank
                break
        if gold_rank is not None and gold_rank >= 1:  # skip already-correct rank 0
            candidates.append((qid, gold_rank))
    candidates.sort(key=lambda x: x[1])  # sort by gold rank (hardest first = lowest non-zero)

    selected = candidates[:n_queries]
    print(f"  Pilot: {len(selected)} queries (gold in top-50)")

    # Load needed docs
    docid_set = set()
    for qid, _ in selected:
        for docid, _ in retrieved[qid][:10]:
            docid_set.add(docid)
        for docid in qrels.get(qid, set()):
            docid_set.add(docid)
    corpus = load_corpus_docs(DATA_DIR, docid_set)

    # Score and print
    correct_boost = 0
    total = 0
    for qid, gold_rank in selected:
        gold = qrels.get(qid, set())
        docs = retrieved[qid][:10]
        boost = qwen3_cache[qid].get("boost_fields", [])
        suppress = qwen3_cache[qid].get("suppress_fields", [])
        gold_docid = None
        for docid in gold:
            if docid in [d for d, _ in retrieved[qid][:50]]:
                gold_docid = docid
                break

        print(f"\n{'='*60}")
        print(f"  Query [{qid}] gold at rank {gold_rank}")
        print(f"  Q: {queries[qid][:120]}")
        print(f"  boost: {boost}  suppress: {suppress}")

        # Score top-3 and gold doc
        test_docs = []
        for rank, (docid, mfar_score) in enumerate(docs[:3]):
            test_docs.append((rank, docid, mfar_score))
        if gold_docid and gold_rank >= 3:
            test_docs.append((gold_rank, gold_docid, retrieved[qid][gold_rank][1]))

        for rank, docid, mfar_score in test_docs:
            doc_json = corpus.get(docid, {"name": "?", "type": "?"})
            is_gold = docid in gold
            result = score_one(qid, docid, queries[qid], doc_json, boost, model, endpoint)
            llm_score = result["llm_score"]
            marker = " <<< GOLD" if is_gold else ""
            print(f"    rank {rank}: doc={docid} mfar={mfar_score:.4f} llm={llm_score:.0f} "
                  f"name={doc_json.get('name', '?')[:30]}{marker}")

            if is_gold:
                total += 1
                # Check if LLM score is higher than top-1 non-gold
                top1_docid = docs[0][0]
                if top1_docid not in gold:
                    top1_result = score_one(qid, top1_docid, queries[qid],
                                           corpus.get(top1_docid, {"name": "?"}),
                                           boost, model, endpoint)
                    if llm_score > top1_result["llm_score"]:
                        correct_boost += 1

    print(f"\n{'='*60}")
    print(f"  Pilot summary: {correct_boost}/{total} gold docs scored higher than top-1 distractor")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Re-Ranking for negation queries")
    subparsers = parser.add_subparsers(dest="command")

    # Score command
    score_parser = subparsers.add_parser("score", help="Score (query, doc) pairs via Qwen3")
    score_parser.add_argument("--splits", nargs="+", default=["val"])
    score_parser.add_argument("--model", default="qwen3:8b")
    score_parser.add_argument("--endpoints", default="http://127.0.0.1:11434")
    score_parser.add_argument("--workers", type=int, default=None)
    score_parser.add_argument("--top_k", type=int, default=50)

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge LLM + mFAR scores and evaluate")
    merge_parser.add_argument("--splits", nargs="+", default=["val"])
    merge_parser.add_argument("--model", default="qwen3:8b",
                              help="Model whose rerank cache to use")
    merge_parser.add_argument("--top_k", type=int, default=50)
    merge_parser.add_argument("--alpha", type=float, default=0.5)
    merge_parser.add_argument("--alpha_sweep", nargs="+", type=float, default=None)

    # Pilot command
    pilot_parser = subparsers.add_parser("pilot", help="Score 50 queries for manual inspection")
    pilot_parser.add_argument("--splits", nargs="+", default=["val"])
    pilot_parser.add_argument("--model", default="qwen3:8b")
    pilot_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    pilot_parser.add_argument("--n_queries", type=int, default=50)

    args = parser.parse_args()

    if args.command == "score":
        endpoints = [ep.strip() for ep in args.endpoints.split(",")]
        workers = args.workers if args.workers else len(endpoints)
        for split in args.splits:
            print(f"\n{'='*60}")
            print(f"  Scoring split: {split} (model: {args.model})")
            print(f"{'='*60}")
            batch_score(split, args.model, endpoints, workers, args.top_k)

    elif args.command == "merge":
        tag = _model_tag(args.model)
        alphas = args.alpha_sweep if args.alpha_sweep else [args.alpha]

        # Merge all splits first, then evaluate per alpha with all splits at once
        for alpha in alphas:
            out_dir = os.path.join(BASE_RUNS_DIR, "rerank", tag, f"alpha_{alpha}_top{args.top_k}")

            for split in args.splits:
                print(f"\n{'='*60}")
                print(f"  Merging split: {split}, α={alpha} (model: {args.model})")
                print(f"{'='*60}")

                cache_path = os.path.join(BASE_CACHE_DIR, "rerank", tag, f"rerank_cache_{split}.jsonl")
                rerank_scores = load_rerank_cache(cache_path)
                print(f"  Loaded {len(rerank_scores)} rerank scores from {cache_path}")

                qwen3_cache = _load_qwen3_cache(split)
                merge_and_write_qres(split, rerank_scores, qwen3_cache,
                                     alpha=alpha, top_k=args.top_k, output_dir=out_dir)

            # Evaluate all splits together
            print(f"\n  Evaluating α={alpha} → {out_dir}")
            from failure_analysis.type_b_memory.evaluate_memory import main as eval_main
            sys.argv = ["evaluate_memory.py",
                         "--baseline_dir", "output/prime_eval",
                         "--memory_dir", out_dir,
                         "--splits"] + args.splits
            try:
                eval_main()
            except SystemExit:
                pass

    elif args.command == "pilot":
        for split in args.splits:
            print(f"\n{'='*60}")
            print(f"  Pilot for split: {split} (model: {args.model})")
            print(f"{'='*60}")
            run_pilot(split, args.model, args.endpoint, args.n_queries)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
