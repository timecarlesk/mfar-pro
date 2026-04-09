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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from failure_analysis.utils import load_queries, load_qrels, load_retrieved, RELATION_FIELDS

# Reuse Ollama caller from batch_qwen3_inference
from failure_analysis.type_b_memory.rerank.shared.qwen3_client import call_ollama

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = "data/prime"
BASE_CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
BASE_RUNS_DIR = "output/failure_analysis/type_b_memory/runs"
DOC_FORMAT_VERSION = "relevant_tag_all_fields_v1"


def _model_tag(model_name):
    """Convert model name to folder-safe tag: 'qwen3:8b' → 'qwen3_8b'."""
    return model_name.replace(":", "_").replace("/", "_")


def _memory_label(memory_version=None, no_memory=False):
    """Build folder/file label for a memory variant."""
    base = memory_version or "default"
    if no_memory:
        return f"{base}_no_memory"
    return base

BASELINE_QRES = {
    "train": "output/prime_train_eval/final-all-0.qres",
    "val": "output/prime_eval/final-all-0.qres",
    "test": "output/prime_eval/final-additional-all-0.qres",
}

RERANKED_QRES = {
    "train": "final-all-0.qres",
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

def format_doc(doc_json, boost_fields=None, max_items=5, show_all=False, max_chars=800):
    """Create compact doc string for LLM prompt.

    Memory mode (default): all fields shown with content, boost_fields tagged [RELEVANT]
    No-memory mode (show_all=True): all fields shown equally, no tags, capped at max_chars
    """
    name = doc_json.get("name", "?")
    dtype = doc_json.get("type", "?")
    parts = [f"{name} ({dtype})"]

    boost_set = set(boost_fields) if boost_fields else set()

    for field in RELATION_FIELDS:
        val = doc_json.get(field)
        if val and isinstance(val, dict):
            items = []
            for subtype, lst in val.items():
                if isinstance(lst, list):
                    items.extend(lst[:max_items])
            if items:
                content = ', '.join(str(x) for x in items[:max_items])
                if not show_all and field in boost_set:
                    parts.append(f"[RELEVANT] {field}: {content}")
                else:
                    parts.append(f"{field}: {content}")
                if sum(len(p) for p in parts) > max_chars:
                    break

    return "; ".join(parts)


def _has_effective_reroute(entry):
    """A query is rerankable only if Stage 1 flagged it and Stage 2 produced fields."""
    if not entry.get("needs_reroute"):
        return False
    boost = entry.get("boost_fields") or []
    suppress = entry.get("suppress_fields") or []
    return bool(boost or suppress)


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

def score_one(qid, docid, query, doc_json, boost_fields, model, endpoint, show_all=False):
    """Score a single (query, doc) pair via Qwen3."""
    doc_text = format_doc(doc_json, boost_fields, show_all=show_all)
    prompt = RERANK_PROMPT.replace("{query}", query).replace("{doc_text}", doc_text)
    try:
        raw = call_ollama(prompt, model, endpoint, max_tokens=5)
        score = parse_rerank_score(raw)
        return {
            "qid": qid,
            "docid": docid,
            "llm_score": score,
            "raw": raw,
            "doc_format_version": DOC_FORMAT_VERSION,
            "no_memory": show_all,
        }
    except Exception as e:
        return {
            "qid": qid,
            "docid": docid,
            "llm_score": 5.0,
            "raw": str(e),
            "doc_format_version": DOC_FORMAT_VERSION,
            "no_memory": show_all,
        }


def load_rerank_cache(cache_path, no_memory=False):
    """Load existing rerank cache. Returns {(qid, docid): score}."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("doc_format_version") != DOC_FORMAT_VERSION:
                    continue
                if bool(entry.get("no_memory", False)) != bool(no_memory):
                    continue
                cache[(entry["qid"], entry["docid"])] = entry["llm_score"]
    return cache


def _load_qwen3_cache(split, memory_version=None, detect_model=None):
    """Load Stage 1+2 cache.

    Searches: stage12/{detect_model}/{memory_version or shared}/ first,
    then falls back through older path patterns.
    """
    search_paths = []
    # Model-specific paths (new structure)
    model_tags = [detect_model] if detect_model else []
    model_tags.extend(["qwen3_8b", "gemma4"])  # common fallbacks
    for mtag in model_tags:
        if split == "train":
            search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", mtag, "shared", f"qwen3_cache_{split}.jsonl"))
        else:
            if memory_version:
                base_version = memory_version.replace("_no_memory", "")
                search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", mtag, base_version, f"qwen3_cache_{split}.jsonl"))
            search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", mtag, "memory_v1", f"qwen3_cache_{split}.jsonl"))
    # Old flat structure fallbacks
    if split == "train":
        search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", "shared", f"qwen3_cache_{split}.jsonl"))
    else:
        if memory_version:
            search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", memory_version.replace("_no_memory", ""), f"qwen3_cache_{split}.jsonl"))
        search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", "memory_v1", f"qwen3_cache_{split}.jsonl"))
    search_paths.append(os.path.join(BASE_CACHE_DIR, f"qwen3_cache_{split}.jsonl"))

    qwen3_cache = {}
    for qwen3_path in search_paths:
        if os.path.exists(qwen3_path):
            print(f"  Loading qwen3 cache from {qwen3_path}")
            with open(qwen3_path) as f:
                for line in f:
                    e = json.loads(line)
                    qwen3_cache[e["qid"]] = e
            return qwen3_cache

    raise FileNotFoundError(f"No qwen3 cache found for split={split}, memory_version={memory_version}")


def batch_score(split, model, endpoints, workers, top_k=50, memory_version=None, no_memory=False, detect_model=None):
    """Score all (rerouted_query, top_k_doc) pairs via Qwen3. Cache as JSONL."""
    tag = _model_tag(model)
    detect_tag = _model_tag(detect_model) if detect_model else tag
    memory_label = _memory_label(memory_version, no_memory=no_memory)
    queries = load_queries(DATA_DIR, split)
    baseline_path = BASELINE_QRES[split]
    retrieved = load_retrieved(baseline_path)

    # Load Stage 1+2 cache (may be from a different model than the reranker)
    qwen3_cache = _load_qwen3_cache(split, memory_version=memory_version, detect_model=detect_tag)

    # Collect (qid, docid) pairs to score
    stage1_flagged_qids = {qid for qid, e in qwen3_cache.items() if e.get("needs_reroute")}
    rerouted_qids = {qid for qid, e in qwen3_cache.items() if _has_effective_reroute(e)}
    pairs = []
    docid_set = set()
    for qid in rerouted_qids:
        docs = retrieved.get(qid, [])[:top_k]
        for docid, _ in docs:
            pairs.append((qid, docid))
            docid_set.add(docid)

    print(f"  Model: {model} (tag: {tag})")
    print(f"  Stage 1 flagged queries: {len(stage1_flagged_qids)}")
    print(f"  Rerouted queries: {len(rerouted_qids)}")
    print(f"  Pairs to score: {len(pairs)}")
    print(f"  Unique docs: {len(docid_set)}")

    # Load corpus for needed docs
    print("  Loading corpus...")
    corpus = load_corpus_docs(DATA_DIR, docid_set)

    # Load existing cache (model + memory_version specific)
    if memory_version or no_memory:
        rerank_cache_dir = os.path.join(BASE_CACHE_DIR, "rerank", tag, memory_label)
    else:
        rerank_cache_dir = os.path.join(BASE_CACHE_DIR, "rerank", tag)
    os.makedirs(rerank_cache_dir, exist_ok=True)
    cache_path = os.path.join(rerank_cache_dir, f"rerank_cache_{split}.jsonl")
    existing = {
        key: score
        for key, score in load_rerank_cache(cache_path, no_memory=no_memory).items()
        if key[0] in rerouted_qids
    }
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
                boost = [] if no_memory else qentry.get("boost_fields", [])
                doc_json = corpus.get(docid, {"name": "?", "type": "?"})
                fut = pool.submit(score_one, qid, docid, queries[qid], doc_json, boost, model, ep, show_all=no_memory)
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

    rerouted_qids = {qid for qid, e in qwen3_cache.items() if _has_effective_reroute(e)}

    if output_dir is None:
        output_dir = os.path.join(BASE_RUNS_DIR, f"rerank_alpha_{alpha}_top{top_k}")
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

def run_pilot(split, model, endpoint, n_queries=50, detect_model=None):
    """Score a small set of queries and print detailed results for manual inspection."""
    queries = load_queries(DATA_DIR, split)
    qrels = load_qrels(DATA_DIR, split)
    retrieved = load_retrieved(BASELINE_QRES[split])

    qwen3_cache = _load_qwen3_cache(split, detect_model=_model_tag(detect_model or model))

    # Find rerouted queries where gold is in top-50 but NOT rank 0
    # These are the hard cases where reranking can actually help
    candidates = []
    for qid, entry in qwen3_cache.items():
        if not _has_effective_reroute(entry):
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
    score_parser.add_argument("--memory_version", default=None,
                              help="Memory version for cache paths (e.g. 'memory_v2')")
    score_parser.add_argument("--no_memory", action="store_true",
                              help="Ablation: ignore boost_fields, show all fields equally to reranker")
    score_parser.add_argument("--detect_model", default=None,
                              help="Model used for Stage 1+2 detection (for loading correct cache). Defaults to --model.")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge LLM + mFAR scores and evaluate")
    merge_parser.add_argument("--splits", nargs="+", default=["val"])
    merge_parser.add_argument("--model", default="qwen3:8b",
                              help="Model whose rerank cache to use")
    merge_parser.add_argument("--top_k", type=int, default=50)
    merge_parser.add_argument("--alpha", type=float, default=0.5)
    merge_parser.add_argument("--alpha_sweep", nargs="+", type=float, default=None)
    merge_parser.add_argument("--memory_version", default=None,
                              help="Subfolder for results, e.g. 'memory_v2' → runs/rerank/model/memory_v2/alpha_...")
    merge_parser.add_argument("--no_memory", action="store_true",
                              help="Use no-memory ablation rerank cache and write to *_no_memory outputs")
    merge_parser.add_argument("--detect_model", default=None,
                              help="Model used for Stage 1+2 detection (for loading correct cache). Defaults to --model.")

    # Pilot command
    pilot_parser = subparsers.add_parser("pilot", help="Score 50 queries for manual inspection")
    pilot_parser.add_argument("--splits", nargs="+", default=["val"])
    pilot_parser.add_argument("--model", default="qwen3:8b")
    pilot_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    pilot_parser.add_argument("--n_queries", type=int, default=50)
    pilot_parser.add_argument("--detect_model", default=None,
                              help="Model used for Stage 1+2 detection (for loading correct cache). Defaults to --model.")

    args = parser.parse_args()

    if args.command == "score":
        endpoints = [ep.strip() for ep in args.endpoints.split(",")]
        workers = args.workers if args.workers else len(endpoints)
        for split in args.splits:
            print(f"\n{'='*60}")
            print(f"  Scoring split: {split} (model: {args.model})")
            print(f"{'='*60}")
            batch_score(split, args.model, endpoints, workers, args.top_k, args.memory_version,
                       getattr(args, 'no_memory', False),
                       detect_model=getattr(args, 'detect_model', None) or args.model)

    elif args.command == "merge":
        tag = _model_tag(args.model)
        alphas = args.alpha_sweep if args.alpha_sweep else [args.alpha]
        memory_label = _memory_label(args.memory_version, no_memory=getattr(args, "no_memory", False))

        # Merge all splits first, then evaluate per alpha with all splits at once
        for alpha in alphas:
            if args.memory_version or getattr(args, "no_memory", False):
                out_dir = os.path.join(BASE_RUNS_DIR, "rerank", tag, memory_label, f"alpha_{alpha}_top{args.top_k}")
            else:
                out_dir = os.path.join(BASE_RUNS_DIR, "rerank", tag, f"alpha_{alpha}_top{args.top_k}")

            for split in args.splits:
                print(f"\n{'='*60}")
                print(f"  Merging split: {split}, α={alpha} (model: {args.model})")
                print(f"{'='*60}")

                if args.memory_version or getattr(args, "no_memory", False):
                    cache_path = os.path.join(BASE_CACHE_DIR, "rerank", tag, memory_label, f"rerank_cache_{split}.jsonl")
                else:
                    cache_path = os.path.join(BASE_CACHE_DIR, "rerank", tag, f"rerank_cache_{split}.jsonl")
                rerank_scores = load_rerank_cache(cache_path, no_memory=getattr(args, "no_memory", False))
                print(f"  Loaded {len(rerank_scores)} rerank scores from {cache_path}")

                detect_tag = _model_tag(getattr(args, 'detect_model', None) or args.model)
                qwen3_cache = _load_qwen3_cache(split, memory_version=args.memory_version, detect_model=detect_tag)
                merge_and_write_qres(split, rerank_scores, qwen3_cache,
                                     alpha=alpha, top_k=args.top_k, output_dir=out_dir)

            # Evaluate all splits together
            print(f"\n  Evaluating α={alpha} → {out_dir}")
            from failure_analysis.type_b_memory.rerank.scoring.evaluate import main as eval_main
            eval_args = ["evaluate_memory.py",
                         "--baseline_dir", "output/prime_eval",
                         "--memory_dir", out_dir,
                         "--splits"] + args.splits
            if args.memory_version:
                eval_args += ["--memory_version", args.memory_version]
            dm = getattr(args, 'detect_model', None) or args.model
            eval_args += ["--detect_model", dm.replace(":", "_").replace("/", "_")]
            sys.argv = eval_args
            try:
                eval_main()
            except SystemExit:
                pass

    elif args.command == "pilot":
        for split in args.splits:
            print(f"\n{'='*60}")
            print(f"  Pilot for split: {split} (model: {args.model})")
            print(f"{'='*60}")
            run_pilot(split, args.model, args.endpoint, args.n_queries,
                      detect_model=getattr(args, "detect_model", None) or args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
