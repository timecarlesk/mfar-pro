"""
Config-aware rerank scoring and merging for Meta-Harness.

Wraps the existing rerank pipeline but reads format_doc / prompt / alpha
from a HarnessConfig instead of hardcoded values.

Each round gets its own rerank cache (keyed by round number) so we can
compare across rounds without invalidating previous caches.
"""

import json
import os
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from failure_analysis.utils import (
    load_queries, load_qrels, load_retrieved, RELATION_FIELDS,
)
from failure_analysis.type_b_memory.rerank.shared.qwen3_client import call_ollama
from failure_analysis.type_b_memory.meta_harness.harness_config import HarnessConfig

DATA_DIR = "data/prime"
BASE_CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
BASE_RUNS_DIR = "output/failure_analysis/type_b_memory/runs"

BASELINE_QRES = {
    "val": "output/contriever/prime_eval/final-all-0.qres",
    "test": "output/contriever/prime_eval/final-additional-all-0.qres",
    "train-dev": "output/contriever/prime_eval/final-train-dev-all-0.qres",
}


def load_corpus_docs(data_dir, docid_set):
    """Load full JSON for a set of doc IDs."""
    corpus = {}
    with open(f"{data_dir}/corpus") as f:
        for line in f:
            idx, json_str = line.strip().split("\t", 1)
            if idx in docid_set:
                corpus[idx] = json.loads(json_str)
    return corpus


# ── Config-Aware format_doc ─────────────────────────────────────────────────

def format_doc_with_config(doc_json, boost_fields, config: HarnessConfig,
                           negation_pattern=None):
    """Format a document for LLM prompt, driven by HarnessConfig.

    Priority logic:
    1. If config.field_priority has an entry for this negation_pattern, use that
       as the ordered field list (overrides Stage 2 boost_fields).
    2. Otherwise, use the Stage 2 boost_fields as-is.
    3. show_suppressed_as controls how non-boosted fields appear.
    """
    name = doc_json.get("name", "?")
    dtype = doc_json.get("type", "?")
    parts = [f"{name} ({dtype})"]

    # Determine which fields to show with content
    priority = config.get_field_priority(negation_pattern)
    if priority:
        show_fields = priority  # config overrides Stage 2
    else:
        show_fields = boost_fields or []  # fall back to Stage 2

    max_items = config.max_items_per_field
    suppressed_label = config.show_suppressed_as

    # Show priority/boost fields with content
    shown = set()
    for field in show_fields:
        val = doc_json.get(field)
        if val and isinstance(val, dict):
            items = []
            for subtype, lst in val.items():
                if isinstance(lst, list):
                    items.extend(lst[:max_items])
            if items:
                parts.append(f"{field}: {', '.join(str(x) for x in items[:max_items])}")
                shown.add(field)

    # Show remaining fields
    for field in RELATION_FIELDS:
        if field in shown:
            continue
        val = doc_json.get(field)
        if val and isinstance(val, dict) and any(val.values()):
            if suppressed_label:
                parts.append(f"{field}: {suppressed_label}")
            # else: hide entirely

    result = "; ".join(parts)
    if len(result) > config.max_chars:
        result = result[:config.max_chars] + "..."
    return result


# ── Config-Aware scoring ────────────────────────────────────────────────────

def parse_rerank_score(raw_output):
    """Parse LLM output to a float score 0-10."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip()
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if match:
        score = float(match.group(1))
        return max(0.0, min(10.0, score))
    return 5.0


def score_one_with_config(qid, docid, query, doc_json, boost_fields,
                          model, endpoint, config: HarnessConfig,
                          negation_pattern=None, max_retries=3):
    """Score a single (query, doc) pair using config-driven format_doc and prompt."""
    doc_text = format_doc_with_config(doc_json, boost_fields, config, negation_pattern)
    prompt_template = config.get_full_prompt()
    prompt = prompt_template.replace("{query}", query).replace("{doc_text}", doc_text)
    for attempt in range(max_retries):
        try:
            raw = call_ollama(prompt, model, endpoint, max_tokens=5)
            score = parse_rerank_score(raw)
            return {"qid": qid, "docid": docid, "llm_score": score, "raw": raw,
                    "doc_text": doc_text}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"qid": qid, "docid": docid, "llm_score": 5.0, "raw": str(e),
                    "doc_text": doc_text}


def load_rerank_cache(cache_path):
    """Load existing rerank cache → {(qid, docid): score}."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[(entry["qid"], entry["docid"])] = entry["llm_score"]
    return cache


def load_rerank_cache_full(cache_path):
    """Load full cache entries (including doc_text) → {(qid, docid): entry}."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[(entry["qid"], entry["docid"])] = entry
    return cache


def load_qwen3_cache(split):
    """Load Qwen3 Stage 1+2 cache with multi-path fallback."""
    search_paths = [
        os.path.join(BASE_CACHE_DIR, "stage12", "shared", f"qwen3_cache_{split}.jsonl"),
        os.path.join(BASE_CACHE_DIR, "stage12", "memory_v2", f"qwen3_cache_{split}.jsonl"),
        os.path.join(BASE_CACHE_DIR, "stage12", "memory_v1", f"qwen3_cache_{split}.jsonl"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            cache = {}
            with open(p) as f:
                for line in f:
                    e = json.loads(line)
                    cache[e["qid"]] = e
            print(f"  Loaded {len(cache)} qwen3 entries from {p}")
            return cache
    raise FileNotFoundError(f"No qwen3 cache for split={split}")


# ── Batch scoring ───────────────────────────────────────────────────────────

def _try_reuse_existing_cache(split, tag, round_num, cache_path):
    """For Round 0 baseline, try to reuse existing memory_v1 rerank scores."""
    if round_num != 0:
        return False
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return False  # already has data

    existing_cache = os.path.join(
        BASE_CACHE_DIR, "rerank", tag, "memory_v1", f"rerank_cache_{split}.jsonl")
    if not os.path.exists(existing_cache):
        return False

    import shutil
    shutil.copy2(existing_cache, cache_path)
    print(f"  Reused existing scores from {existing_cache}")
    return True


def batch_score_with_config(split, model, endpoints, workers,
                            config: HarnessConfig, round_num=0):
    """Score all rerouted (query, doc) pairs using config. Cache per round."""
    tag = model.replace(":", "_").replace("/", "_")
    queries = load_queries(DATA_DIR, split)
    retrieved = load_retrieved(BASELINE_QRES[split])
    qwen3_cache = load_qwen3_cache(split)

    rerouted_qids = {qid for qid, e in qwen3_cache.items() if e.get("needs_reroute")}
    top_k = config.top_k

    pairs = []
    docid_set = set()
    for qid in rerouted_qids:
        if qid not in retrieved:
            continue
        docs = retrieved[qid][:top_k]
        for docid, _ in docs:
            pairs.append((qid, docid))
            docid_set.add(docid)

    print(f"  Rerouted queries: {len(rerouted_qids)}, pairs: {len(pairs)}")

    # Cache path: per round to avoid mixing configs
    cache_dir = os.path.join(BASE_CACHE_DIR, "meta_harness", tag, f"round_{round_num}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"rerank_cache_{split}.jsonl")

    # Try reuse existing memory_v1 scores for Round 0
    _try_reuse_existing_cache(split, tag, round_num, cache_path)

    existing = load_rerank_cache(cache_path)
    remaining = [(qid, did) for qid, did in pairs if (qid, did) not in existing]
    print(f"  Cache: {len(existing)} existing, {len(remaining)} remaining")

    if not remaining:
        return existing, cache_path

    # Health check: verify at least one endpoint is alive
    alive = 0
    for ep in endpoints:
        try:
            call_ollama("hi", model, ep, max_tokens=1)
            alive += 1
        except:
            print(f"  WARNING: endpoint {ep} not responding")
    if alive == 0:
        raise RuntimeError("All Ollama endpoints are down. Start Ollama first.")
    print(f"  Ollama health: {alive}/{len(endpoints)} endpoints alive")

    corpus = load_corpus_docs(DATA_DIR, docid_set)

    if not remaining:
        return existing, cache_path

    write_lock = threading.Lock()
    done = [0]
    total = len(remaining)
    start_time = time.time()

    def print_progress():
        elapsed = time.time() - start_time
        rate = done[0] / elapsed if elapsed > 0 else 0
        eta = (total - done[0]) / rate if rate > 0 else 0
        pct = 100 * done[0] / total
        bar_len = 30
        filled = int(bar_len * done[0] / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {done[0]}/{total} ({pct:.1f}%) | {rate:.1f}/s | ETA {int(eta)}s", end="", flush=True)

    with open(cache_path, "a") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, (qid, docid) in enumerate(remaining):
                ep = endpoints[i % len(endpoints)]
                qentry = qwen3_cache.get(qid, {})
                boost = qentry.get("boost_fields", [])
                neg_pattern = qentry.get("negation_pattern")
                doc_json = corpus.get(docid, {"name": "?", "type": "?"})
                fut = pool.submit(
                    score_one_with_config, qid, docid, queries[qid],
                    doc_json, boost, model, ep, config, neg_pattern)
                futures[fut] = (qid, docid)

            for future in as_completed(futures):
                entry = future.result()
                with write_lock:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    existing[(entry["qid"], entry["docid"])] = entry["llm_score"]
                    done[0] += 1
                    if done[0] % 50 == 0 or done[0] == total:
                        print_progress()

    print(f"\n  Done: {len(existing)} total scored ({time.time() - start_time:.0f}s)")
    return existing, cache_path


# ── Config-Aware merging ────────────────────────────────────────────────────

def _minmax_normalize(scores):
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-8:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def merge_with_config(split, rerank_scores, qwen3_cache, config: HarnessConfig,
                      round_num=0):
    """Merge LLM + mFAR scores using config's alpha_by_type. Write .qres."""
    retrieved = load_retrieved(BASELINE_QRES[split])
    rerouted_qids = {qid for qid, e in qwen3_cache.items() if e.get("needs_reroute")}
    top_k = config.top_k

    out_dir = os.path.join(BASE_RUNS_DIR, "meta_harness", f"round_{round_num}")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"reranked_{split}.qres")

    reranked_count = 0
    copied_count = 0

    with open(output_path, "w") as f:
        for qid in sorted(retrieved.keys()):
            docs = retrieved[qid]

            if qid not in rerouted_qids:
                for rank, (docid, score) in enumerate(docs[:100]):
                    f.write(f"{qid}\t0\t{docid}\t{rank}\t{score}\t0\n")
                copied_count += 1
                continue

            top_docs = docs[:top_k]
            rest_docs = docs[top_k:100]

            # Get alpha based on answer_type
            answer_type = qwen3_cache.get(qid, {}).get("answer_type")
            alpha = config.get_alpha(answer_type)

            mfar_scores = [score for _, score in top_docs]
            llm_scores = [rerank_scores.get((qid, docid), 5.0)
                          for docid, _ in top_docs]

            mfar_norm = _minmax_normalize(mfar_scores)
            llm_norm = _minmax_normalize(llm_scores)

            final_scores = [alpha * l + (1 - alpha) * m
                            for l, m in zip(llm_norm, mfar_norm)]

            indexed = list(zip(range(len(top_docs)), final_scores))
            indexed.sort(key=lambda x: -x[1])

            for rank, (orig_idx, fscore) in enumerate(indexed):
                docid = top_docs[orig_idx][0]
                f.write(f"{qid}\t0\t{docid}\t{rank}\t{fscore + 100}\t0\n")

            for rank_offset, (docid, score) in enumerate(rest_docs):
                rank = top_k + rank_offset
                f.write(f"{qid}\t0\t{docid}\t{rank}\t{score}\t0\n")

            reranked_count += 1

    print(f"  Wrote {output_path}: {reranked_count} reranked, {copied_count} copied")
    return output_path
