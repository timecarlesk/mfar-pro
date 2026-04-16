"""
Self-Verification Feedback Loop for Memory Evolution.

After reranking, Qwen3 checks if each rerouted query's top-1 result
actually satisfies the negation constraint. No gold answer needed.

Per-(answer_type, negation_pattern) group pass rates are computed and
injected back into memory_context as confidence signals.

Commands:
  verify        — LLM-verify top-1 results, cache as JSONL
  update-memory — compute pass rates, write memory_context_v2

Run from project root:
  # Step 0: need validation rerank results first
  $PY rerank/scoring/rerank.py score --splits val --model qwen3:8b --endpoints $EP
  $PY rerank/scoring/rerank.py merge --splits val --model qwen3:8b --alpha 0.7

  # Step 1: verify validation top-1
  $PY rerank/scoring/verify.py verify --splits val --model qwen3:8b --endpoints $EP

  # Step 2: update memory
  $PY rerank/scoring/verify.py update-memory
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from failure_analysis.utils import load_queries, load_retrieved
from failure_analysis.type_b_memory.rerank.shared.qwen3_client import call_ollama
from failure_analysis.type_b_memory.rerank.scoring.rerank import (
    DOC_FORMAT_VERSION,
    format_doc as format_rerank_doc,
)

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = "data/prime"
BASE_CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
BASE_RUNS_DIR = "output/failure_analysis/type_b_memory/runs"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"

SPLIT_QRES = {
    "val": "final-all-0.qres",
    "test": "final-additional-all-0.qres",
    "train": "final-all-0.qres",
}


def _model_tag(model_name):
    return model_name.replace(":", "_").replace("/", "_")


def _stage12_required_path(split, memory_version=None, detect_model=None):
    model_tag = _model_tag(detect_model) if detect_model else "qwen3_8b"
    if split == "train":
        return os.path.join(BASE_CACHE_DIR, "stage12", model_tag, "shared", f"qwen3_cache_{split}.jsonl")
    version = (memory_version or "memory_v1").replace("_no_memory", "")
    return os.path.join(BASE_CACHE_DIR, "stage12", model_tag, version, f"qwen3_cache_{split}.jsonl")


# ── Corpus + Doc Formatting ──────────────────────────────────────────────────

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


def format_doc(doc_json, boost_fields=None, max_items=5):
    """Mirror rerank document formatting for verification."""
    return format_rerank_doc(doc_json, boost_fields=boost_fields, max_items=max_items, show_all=False)


def _has_effective_reroute(entry):
    """A query should be verified only if it actually triggered field reranking."""
    if not entry.get("needs_reroute"):
        return False
    boost = entry.get("boost_fields") or []
    suppress = entry.get("suppress_fields") or []
    unmapped_boost = entry.get("unmapped_boost_fields") or []
    unmapped_suppress = entry.get("unmapped_suppress_fields") or []
    return bool(boost or suppress or unmapped_boost or unmapped_suppress)


# ── Qwen3 Cache Loading ─────────────────────────────────────────────────────

def _load_qwen3_cache(split, memory_version=None, detect_model=None):
    """Load Qwen3 Stage 1+2 cache."""
    search_paths = []
    base_version = memory_version.replace("_no_memory", "") if memory_version else None
    required_path = _stage12_required_path(split, memory_version=base_version, detect_model=detect_model)

    if detect_model:
        model_tags = [_model_tag(detect_model)]
    else:
        model_tags = ["qwen3_8b", "gemma4"]

    for mtag in model_tags:
        if split == "train":
            search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", mtag, "shared", f"qwen3_cache_{split}.jsonl"))
        else:
            if base_version:
                search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", mtag, base_version, f"qwen3_cache_{split}.jsonl"))
            search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", mtag, "memory_v1", f"qwen3_cache_{split}.jsonl"))
    # Fallbacks
    if split == "train":
        search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", "shared", f"qwen3_cache_{split}.jsonl"))
    else:
        if base_version:
            search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", base_version, f"qwen3_cache_{split}.jsonl"))
        search_paths.append(os.path.join(BASE_CACHE_DIR, "stage12", "memory_v1", f"qwen3_cache_{split}.jsonl"))
    search_paths.append(os.path.join(BASE_CACHE_DIR, "qwen3_8b", f"qwen3_cache_{split}.jsonl"))
    search_paths.append(os.path.join(BASE_CACHE_DIR, f"qwen3_cache_{split}.jsonl"))

    for path in search_paths:
        if os.path.exists(path):
            if memory_version and path != required_path:
                raise FileNotFoundError(
                    f"Expected qwen3 cache for split={split}, memory_version={memory_version} at {required_path}, "
                    f"but verifier would have fallen back to {path}"
                )
            cache = {}
            with open(path) as f:
                for line in f:
                    e = json.loads(line)
                    cache[e["qid"]] = e
            print(f"  Loaded qwen3 cache from {path}")
            return cache

    raise FileNotFoundError(
        f"No qwen3 cache for split={split}, memory_version={memory_version}, detect_model={detect_model}"
    )


# ── Verification Prompt ──────────────────────────────────────────────────────

VERIFY_PROMPT = """\
Query: "{query}"
Top retrieval result: {doc_text}

Does this document satisfy ALL constraints in the query, including any negation?
For example:
- "drugs NOT indicated for diabetes" → the document should show this drug is contraindicated/not used for diabetes
- "genes not expressed in liver" → the document should show this gene is absent from liver expression
- "diseases lacking treatment" → the document should be a disease with no known drug treatments

Does the document above correctly answer the query when the negation is properly interpreted?
Answer ONLY "yes" or "no"."""


def parse_verify_output(raw_output):
    """Parse verification yes/no output."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip().lower()
    return "yes" in cleaned


_VERIFY_ERROR_MARKERS = (
    "http error",
    "internal server error",
    "timed out",
    "timeout",
    "connection refused",
    "connection reset",
    "remote end closed connection",
    "service unavailable",
    "temporary failure",
    "name or service not known",
    "nodename nor servname",
    "[errno",
)


def _looks_like_verify_error(raw_text):
    lowered = str(raw_text).strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _VERIFY_ERROR_MARKERS)


def _verify_entry_status(entry):
    status = entry.get("status")
    if status in {"ok", "error"}:
        return status
    if _looks_like_verify_error(entry.get("raw", "")):
        return "error"
    return "ok"


# ── Verification ─────────────────────────────────────────────────────────────

def verify_one(qid, docid, query, doc_json, boost_fields, model, endpoint,
               memory_version="memory_v1", alpha=0.7, top_k=50):
    """Verify a single (query, top-1 doc) pair."""
    doc_text = format_doc(doc_json, boost_fields)
    prompt = VERIFY_PROMPT.replace("{query}", query).replace("{doc_text}", doc_text)
    try:
        raw = call_ollama(prompt, model, endpoint, max_tokens=5)
        verified = parse_verify_output(raw)
        return {
            "qid": qid,
            "docid": docid,
            "verified": verified,
            "raw": raw,
            "status": "ok",
            "memory_version": memory_version,
            "alpha": alpha,
            "top_k": top_k,
            "doc_format_version": DOC_FORMAT_VERSION,
        }
    except Exception as e:
        return {
            "qid": qid,
            "docid": docid,
            "verified": False,
            "raw": str(e),
            "status": "error",
            "memory_version": memory_version,
            "alpha": alpha,
            "top_k": top_k,
            "doc_format_version": DOC_FORMAT_VERSION,
        }


def load_verify_cache(cache_path, memory_version="memory_v1", alpha=0.7, top_k=50):
    """Load existing verification cache."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("doc_format_version") != DOC_FORMAT_VERSION:
                    continue
                if entry.get("memory_version", "memory_v1") != memory_version:
                    continue
                if float(entry.get("alpha", alpha)) != float(alpha):
                    continue
                if int(entry.get("top_k", top_k)) != int(top_k):
                    continue
                if _verify_entry_status(entry) != "ok":
                    continue
                cache[entry["qid"]] = entry
    return cache


def batch_verify(split, model, endpoints, workers, alpha=0.7, top_k=50, memory_version=None, detect_model=None):
    """Verify all rerouted queries' top-1 results."""
    tag = _model_tag(model)
    queries = load_queries(DATA_DIR, split)
    detect_tag = _model_tag(detect_model) if detect_model else tag
    qwen3_cache = _load_qwen3_cache(split, memory_version=memory_version, detect_model=detect_tag)

    # Load reranked results to get top-1 per query
    qres_name = SPLIT_QRES.get(split)
    if memory_version:
        reranked_dir = os.path.join(BASE_RUNS_DIR, "rerank", tag, memory_version, f"alpha_{alpha}_top{top_k}")
    else:
        reranked_dir = os.path.join(BASE_RUNS_DIR, "rerank", tag, f"alpha_{alpha}_top{top_k}")
    reranked_path = os.path.join(reranked_dir, qres_name)

    if not os.path.exists(reranked_path):
        print(f"  ERROR: Reranked results not found: {reranked_path}")
        print(f"  Run: rerank.py merge --splits {split} --model {model} --alpha {alpha}")
        return {}

    reranked = load_retrieved(reranked_path)
    stage1_flagged_qids = {qid for qid, e in qwen3_cache.items() if e.get("needs_reroute")}
    rerouted_qids = {qid for qid, e in qwen3_cache.items() if _has_effective_reroute(e)}
    print(f"  Stage 1 flagged queries: {len(stage1_flagged_qids)}")
    print(f"  Rerouted queries: {len(rerouted_qids)}")

    # Collect top-1 doc IDs
    top1_pairs = {}
    docid_set = set()
    for qid in rerouted_qids:
        docs = reranked.get(qid, [])
        if docs:
            top1_docid = docs[0][0]
            top1_pairs[qid] = top1_docid
            docid_set.add(top1_docid)

    print(f"  Top-1 pairs to verify: {len(top1_pairs)}")

    # Load corpus
    print("  Loading corpus...")
    corpus = load_corpus_docs(DATA_DIR, docid_set)

    # Load existing cache
    verify_cache_dir = os.path.join(BASE_CACHE_DIR, "verify", tag)
    os.makedirs(verify_cache_dir, exist_ok=True)
    cache_path = os.path.join(verify_cache_dir, f"verify_cache_{split}.jsonl")
    existing = {
        qid: entry
        for qid, entry in load_verify_cache(cache_path, memory_version=memory_version, alpha=alpha, top_k=top_k).items()
        if qid in rerouted_qids
    }
    remaining = {qid: docid for qid, docid in top1_pairs.items() if qid not in existing}
    print(f"  Cache has {len(existing)} entries, {len(remaining)} remaining")

    if not remaining:
        print("  All pairs already verified")
        return existing

    # Verify with thread pool
    write_lock = threading.Lock()
    done = [0]
    error_count = [0]

    with open(cache_path, "a") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, (qid, docid) in enumerate(remaining.items()):
                ep = endpoints[i % len(endpoints)]
                boost = qwen3_cache.get(qid, {}).get("boost_fields", [])
                doc_json = corpus.get(docid, {"name": "?", "type": "?"})
                fut = pool.submit(
                    verify_one, qid, docid, queries[qid], doc_json, boost, model, ep,
                    memory_version, alpha, top_k
                )
                futures[fut] = qid

            for future in as_completed(futures):
                entry = future.result()
                with write_lock:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    if entry.get("status") == "ok":
                        existing[entry["qid"]] = entry
                    else:
                        error_count[0] += 1
                    done[0] += 1
                    if done[0] % 100 == 0:
                        print(f"  Verified {done[0]}/{len(remaining)} "
                              f"({error_count[0]} errors)")

    # Summary
    verified_yes = sum(1 for e in existing.values() if e.get("verified"))
    pass_rate = 100 * verified_yes / len(existing) if existing else 0
    print(f"  Done: {len(existing)} verified, {verified_yes} passed ({pass_rate:.0f}%), "
          f"{error_count[0]} errors")
    return existing


# ── Pass Rate Computation ────────────────────────────────────────────────────

def compute_pass_rates(split, model, memory_version="memory_v1", alpha=0.7, top_k=50, detect_model=None):
    """Compute per-(answer_type, negation_pattern) verification pass rates."""
    tag = _model_tag(model)
    detect_tag = _model_tag(detect_model) if detect_model else tag
    qwen3_cache = _load_qwen3_cache(split, memory_version=memory_version, detect_model=detect_tag)

    # Load verification cache
    cache_path = os.path.join(BASE_CACHE_DIR, "verify", tag, f"verify_cache_{split}.jsonl")
    verify_cache = load_verify_cache(cache_path, memory_version=memory_version, alpha=alpha, top_k=top_k)

    # Group by (answer_type, negation_pattern) and compute pass rates
    group_stats = defaultdict(lambda: {"total": 0, "passed": 0})

    for qid, entry in verify_cache.items():
        qwen3_entry = qwen3_cache.get(qid, {})
        if not _has_effective_reroute(qwen3_entry):
            continue
        answer_type = qwen3_entry.get("answer_type", "unknown")
        neg_pattern = qwen3_entry.get("negation_pattern", "other")
        group_key = f"{answer_type}|{neg_pattern}"

        group_stats[group_key]["total"] += 1
        if entry.get("verified"):
            group_stats[group_key]["passed"] += 1

    # Compute rates
    rates = {}
    for group_key, stats in sorted(group_stats.items(), key=lambda x: -x[1]["total"]):
        rate = stats["passed"] / stats["total"] if stats["total"] else 0
        rates[group_key] = {
            "total": stats["total"],
            "passed": stats["passed"],
            "rate": round(rate, 3),
        }

    return rates


# ── Memory Update ────────────────────────────────────────────────────────────

def update_memory_context(pass_rates, memory_path, output_path):
    """Inject verification confidence into memory context rules."""
    with open(memory_path) as f:
        memory = f.read()

    lines = memory.split("\n")
    new_lines = []

    for line in lines:
        new_lines.append(line)

        # After each "Recommended boost_fields:" line, inject confidence
        if line.strip().startswith("Recommended boost_fields:"):
            # Find the group key from the preceding "When" line
            # Look back for the "When answer_type=X and query contains Y" line
            for prev_line in reversed(new_lines[:-1]):
                if prev_line.strip().startswith("When answer_type="):
                    # Extract answer_type and pattern
                    match = re.search(
                        r'answer_type=(\S+)\s+and\s+query\s+contains\s+"([^"]+)"',
                        prev_line
                    )
                    if match:
                        at = match.group(1)
                        pat = match.group(2).replace(" ", "_")
                        group_key = f"{at}|{pat}"

                        rate_info = pass_rates.get(group_key)
                        if rate_info:
                            rate = rate_info["rate"]
                            total = rate_info["total"]
                            passed = rate_info["passed"]
                            if rate >= 0.7:
                                new_lines.append(
                                    f"  Verification confidence: {rate:.0%} "
                                    f"({passed}/{total} top-1 results verified correct)")
                            else:
                                new_lines.append(
                                    f"  WARNING: Low verification confidence: {rate:.0%} "
                                    f"({passed}/{total}). This rule may be unreliable.")
                                new_lines.append(
                                    "  Consider reasoning from entity type field inventory instead.")
                    break

    with open(output_path, "w") as f:
        f.write("\n".join(new_lines))

    print(f"  Memory v2 written to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-verification feedback loop")
    subparsers = parser.add_subparsers(dest="command")

    # Verify command
    v_parser = subparsers.add_parser("verify", help="Verify top-1 results via LLM")
    v_parser.add_argument("--splits", nargs="+", default=["val"])
    v_parser.add_argument("--model", default="qwen3:8b")
    v_parser.add_argument("--endpoints", default="http://127.0.0.1:11434")
    v_parser.add_argument("--workers", type=int, default=None)
    v_parser.add_argument("--alpha", type=float, default=0.7)
    v_parser.add_argument("--top_k", type=int, default=50)
    v_parser.add_argument("--memory_version", default="memory_v1",
                          help="Which memory version's reranked results to verify")
    v_parser.add_argument("--detect_model", default=None,
                          help="Model used for Stage 1+2 detection. Defaults to --model.")

    # Update-memory command
    u_parser = subparsers.add_parser("update-memory", help="Compute pass rates and update memory")
    u_parser.add_argument("--splits", nargs="+", default=["val"])
    u_parser.add_argument("--model", default="qwen3:8b")
    u_parser.add_argument("--memory_version", default="memory_v1")
    u_parser.add_argument("--alpha", type=float, default=0.7)
    u_parser.add_argument("--top_k", type=int, default=50)
    u_parser.add_argument("--detect_model", default=None,
                          help="Model used for Stage 1+2 detection. Defaults to --model.")

    args = parser.parse_args()

    if args.command == "verify":
        endpoints = [ep.strip() for ep in args.endpoints.split(",")]
        workers = args.workers if args.workers else len(endpoints)

        for split in args.splits:
            print(f"\n{'='*60}")
            print(f"  Verifying split: {split} (model: {args.model})")
            print(f"{'='*60}")
            batch_verify(split, args.model, endpoints, workers, args.alpha, args.top_k,
                         args.memory_version, detect_model=getattr(args, "detect_model", None) or args.model)

    elif args.command == "update-memory":
        # Compute pass rates from all specified splits
        all_rates = defaultdict(lambda: {"total": 0, "passed": 0})
        for split in args.splits:
            print(f"\n  Computing pass rates for {split}...")
            rates = compute_pass_rates(
                split, args.model,
                memory_version=args.memory_version,
                alpha=args.alpha,
                top_k=args.top_k,
                detect_model=getattr(args, "detect_model", None) or args.model,
            )
            for group_key, stats in rates.items():
                all_rates[group_key]["total"] += stats["total"]
                all_rates[group_key]["passed"] += stats["passed"]

        # Finalize rates
        final_rates = {}
        print(f"\n  {'Group':<45} {'Total':>5} {'Pass':>5} {'Rate':>6}")
        print(f"  {'-'*65}")
        for group_key in sorted(all_rates, key=lambda k: -all_rates[k]["total"]):
            stats = all_rates[group_key]
            rate = stats["passed"] / stats["total"] if stats["total"] else 0
            final_rates[group_key] = {
                "total": stats["total"],
                "passed": stats["passed"],
                "rate": round(rate, 3),
            }
            status = "OK" if rate >= 0.7 else "LOW"
            print(f"  {group_key:<45} {stats['total']:>5} {stats['passed']:>5} {rate:>5.0%}  {status}")

        # Save rates
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        rates_path = os.path.join(ANALYSIS_DIR, "verification_rates.json")
        with open(rates_path, "w") as f:
            json.dump(final_rates, f, indent=2)
        print(f"\n  Rates saved to {rates_path}")

        # Update memory context
        memory_path = os.path.join(ANALYSIS_DIR, "memory_context_train.txt")
        output_path = os.path.join(ANALYSIS_DIR, "memory_context_train_v2.txt")

        if os.path.exists(memory_path):
            update_memory_context(final_rates, memory_path, output_path)
        else:
            print(f"  ERROR: {memory_path} not found")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
