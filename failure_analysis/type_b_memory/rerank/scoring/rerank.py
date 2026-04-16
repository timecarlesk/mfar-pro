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
import hashlib
import json
import math
import os
import random
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
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"
DOC_FORMAT_VERSION = "relevant_tag_all_fields_v1"
DISCRIMINATION_RULES_PATH = os.path.join(ANALYSIS_DIR, "discrimination_rules.json")


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
    "train": "output/contriever/prime_train_eval/final-all-0.qres",
    "val": "output/contriever/prime_eval/final-all-0.qres",
    "test": "output/contriever/prime_eval/final-additional-all-0.qres",
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

def _ordered_doc_fields(boost_fields=None, show_all=False):
    """Show boosted fields first so `[RELEVANT]` tags survive truncation."""
    if show_all or not boost_fields:
        return list(RELATION_FIELDS)
    boost_set = set(boost_fields)
    boosted = [field for field in RELATION_FIELDS if field in boost_set]
    others = [field for field in RELATION_FIELDS if field not in boost_set]
    return boosted + others


def _truncate_doc_part(part, remaining_chars):
    """Fit one `field: content` segment into the remaining char budget."""
    if remaining_chars <= 0:
        return None
    if len(part) <= remaining_chars:
        return part
    if remaining_chars < 12:
        return None

    if ": " in part:
        prefix, content = part.split(": ", 1)
        min_needed = len(prefix) + len(": ...")
        if remaining_chars < min_needed:
            return None
        keep = remaining_chars - len(prefix) - len(": ...")
        clipped = content[:keep].rstrip(" ,;")
        if not clipped:
            return None
        return f"{prefix}: {clipped}..."

    clipped = part[:remaining_chars - 3].rstrip(" ,;")
    if not clipped:
        return None
    return f"{clipped}..."


def format_doc(doc_json, boost_fields=None, max_items=5, show_all=False, max_chars=800):
    """Create compact doc string for LLM prompt.

    Memory mode (default): boosted fields are shown first and tagged `[RELEVANT]`.
    No-memory mode (show_all=True): all fields shown equally, no tags.
    """
    name = doc_json.get("name", "?")
    dtype = doc_json.get("type", "?")
    parts = [f"{name} ({dtype})"]

    boost_set = set(boost_fields) if boost_fields else set()

    for field in _ordered_doc_fields(boost_fields, show_all=show_all):
        val = doc_json.get(field)
        if not (val and isinstance(val, dict)):
            continue

        items = []
        for subtype, lst in val.items():
            if isinstance(lst, list):
                items.extend(lst[:max_items])
        if not items:
            continue

        content = ', '.join(str(x) for x in items[:max_items])
        label = f"[RELEVANT] {field}" if not show_all and field in boost_set else field
        candidate = f"{label}: {content}"
        current_len = len("; ".join(parts))
        remaining_chars = max_chars - current_len - 2
        fitted = _truncate_doc_part(candidate, remaining_chars)
        if fitted is None:
            break
        parts.append(fitted)
        if fitted.endswith("..."):
            break

    return "; ".join(parts)


def _has_effective_reroute(entry):
    """A query is rerankable only if Stage 1 flagged it and Stage 2 produced fields."""
    if not entry.get("needs_reroute"):
        return False
    boost = entry.get("boost_fields") or []
    suppress = entry.get("suppress_fields") or []
    unmapped_boost = entry.get("unmapped_boost_fields") or []
    unmapped_suppress = entry.get("unmapped_suppress_fields") or []
    return bool(boost or suppress or unmapped_boost or unmapped_suppress)


# ── Re-Ranking Prompt ────────────────────────────────────────────────────────

RERANK_PROMPT = """\
Query: "{query}"
Document: {doc_text}

This query contains negation or constraints. Score how well this document satisfies the query INCLUDING any negation constraints (e.g., "not indicated" means the document should have contraindication data, not indication data).

Score 0-10 where 0=completely irrelevant, 10=perfect match.
Output ONLY the number."""


DISC_PROMPT = """\
Query: "{query}"

Discrimination rule for this query type:
{discrimination_block}

Document: {doc_text}

Apply the discrimination rule above. Score 0-10 where 0=completely irrelevant, 10=perfect match including the negation constraint.
Output ONLY the number."""


# ── Tournament Rerank (Session Memory v2) ────────────────────────────────────

TOURNAMENT_PROMPT = """\
Query: "{query}"

Below are candidate documents for this query. The current leaders survived earlier rounds; the new candidates are being evaluated now.

Current leaders:
{leaders_block}

New candidates:
{new_block}

Task: pick the best {K} documents for this query, considering any negation constraint. Output ONLY {K} unique bracketed IDs, best first, separated by ` > `.
Example:
{example_format}

Each ID must be one of L1..L{K_leaders} (leaders) or N1..N{B} (new candidates). No other text."""

TOURNAMENT_RETRY_HINT = (
    "\n\nStrict format: output exactly {K} unique bracketed IDs separated by ` > `, "
    "for example {example_format}. Use only the IDs listed above. No other text."
)


# ── Discrimination Memory (v3) ───────────────────────────────────────────────

_DISC_CACHE = None


def load_discrimination_rules(path=DISCRIMINATION_RULES_PATH):
    """Lazy-load discrimination_rules.json. Returns {group_key: rule_dict}."""
    global _DISC_CACHE
    if _DISC_CACHE is not None:
        return _DISC_CACHE
    if not os.path.exists(path):
        print(f"  WARNING: no discrimination rules at {path}")
        _DISC_CACHE = {}
        return _DISC_CACHE
    with open(path) as f:
        data = json.load(f)
    _DISC_CACHE = data.get("rules", {})
    print(f"  Loaded {len(_DISC_CACHE)} discrimination rules from {path}")
    return _DISC_CACHE


def _format_discrimination_block(rule):
    """Render a discrimination rule as a concise prompt block."""
    parts = []
    if rule.get("pattern"):
        parts.append(f"Pattern: {rule['pattern']}")
    parts.append(f"Rule: {rule['discrimination_rule']}")
    if rule.get("common_trap"):
        parts.append(f"Common trap: {rule['common_trap']}")
    ex = rule.get("example") or {}
    if ex.get("gold") and ex.get("distractor"):
        parts.append(f"Example gold: {ex['gold']}")
        parts.append(f"Example distractor: {ex['distractor']}")
    return "\n".join(parts)


def get_discrimination_block(answer_type, negation_pattern, rules=None):
    """Return the prompt block for this (answer_type, negation_pattern) or None if no rule."""
    if rules is None:
        rules = load_discrimination_rules()
    if not rules or not answer_type or not negation_pattern:
        return None
    group_key = f"{answer_type.lower()}|{negation_pattern}"
    rule = rules.get(group_key)
    if rule is None:
        return None
    return _format_discrimination_block(rule)


_RERANK_SCORE_PATTERNS = [
    re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:/10)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*score\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:/10)?(?:\b|$)", re.IGNORECASE),
    re.compile(r"\bscore(?:\s+is)?\s*(\d+(?:\.\d+)?)\s*(?:/10)?(?:\b|$)", re.IGNORECASE),
]

_CALL_ERROR_MARKERS = (
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


def _parse_score_candidate(text):
    """Return a parsed score if this text looks like a score-bearing line."""
    for pattern in _RERANK_SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
    return None


def _looks_like_call_error(raw_text):
    """Best-effort legacy error detection for cache rows without explicit status."""
    lowered = str(raw_text).strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _CALL_ERROR_MARKERS)


def _rerank_entry_status(entry):
    """Return `ok` or `error`, inferring legacy rows when needed."""
    status = entry.get("status")
    if status in {"ok", "error"}:
        return status
    if _looks_like_call_error(entry.get("raw", "")):
        return "error"
    return "ok"


def parse_rerank_score(raw_output):
    """Parse LLM output to a float score 0-10.

    Be conservative: prefer a standalone score line or an explicit `Score: X`
    label, and avoid grabbing unrelated numbers like years.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip()

    candidates = [cleaned]
    candidates.extend(line.strip() for line in cleaned.splitlines() if line.strip())
    for candidate in candidates:
        score = _parse_score_candidate(candidate)
        if score is not None:
            return score
    return 5.0  # neutral fallback


def _stable_qid_seed(qid):
    """Deterministic seed for per-query tournament shuffling."""
    digest = hashlib.md5(str(qid).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_one(qid, docid, query, doc_json, boost_fields, model, endpoint, show_all=False, discrimination_block=None):
    """Score a single (query, doc) pair via Qwen3.

    If `discrimination_block` is provided, uses DISC_PROMPT (memory v3).
    Else uses RERANK_PROMPT with `[RELEVANT]` tags driven by boost_fields.
    """
    doc_text = format_doc(doc_json, boost_fields, show_all=show_all)
    if discrimination_block:
        prompt = (DISC_PROMPT
                  .replace("{query}", query)
                  .replace("{discrimination_block}", discrimination_block)
                  .replace("{doc_text}", doc_text))
    else:
        prompt = RERANK_PROMPT.replace("{query}", query).replace("{doc_text}", doc_text)
    try:
        raw = call_ollama(prompt, model, endpoint, max_tokens=5)
        score = parse_rerank_score(raw)
        return {
            "qid": qid,
            "docid": docid,
            "llm_score": score,
            "raw": raw,
            "status": "ok",
            "doc_format_version": DOC_FORMAT_VERSION,
            "no_memory": show_all,
            "memory_mode": "discrimination" if discrimination_block else ("none" if show_all else "relevant_tag"),
        }
    except Exception as e:
        return {
            "qid": qid,
            "docid": docid,
            "llm_score": 5.0,
            "raw": str(e),
            "status": "error",
            "doc_format_version": DOC_FORMAT_VERSION,
            "no_memory": show_all,
            "memory_mode": "discrimination" if discrimination_block else ("none" if show_all else "relevant_tag"),
        }


def _parse_tournament_output(raw, valid_ids):
    """Extract ordered list of IDs from LLM output. Returns list of valid IDs, deduped, in order."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    matches = re.findall(r"\[?(L\d+|N\d+)\]?", cleaned)
    seen = set()
    out = []
    for m in matches:
        if m in valid_ids and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def tournament_rerank_one_query(qid, query, docs, model, endpoint, K=5, B=10):
    """Rolling top-K tournament rerank for a single query.

    docs: list of (docid, doc_json, mfar_score) in mFAR rank order (rank 0 first).
    Returns (result_dict, parse_failures).
    result_dict: {docid: {"tournament_rank": int|None, "llm_score": float}}.
    """
    rng = random.Random(_stable_qid_seed(qid))

    # Seed: mFAR top-K as initial leaders; no LLM call for round 0.
    running_topk = list(docs[:K])
    remaining = list(docs[K:])
    rng.shuffle(remaining)

    parse_failures = 0
    rounds = max(0, (len(remaining) + B - 1) // B)

    for round_idx in range(rounds):
        batch = remaining[round_idx * B : (round_idx + 1) * B]
        if not batch:
            break
        rng.shuffle(batch)

        K_leaders = len(running_topk)
        leaders_lines = [f"[L{i+1}] {format_doc(d[1], show_all=True)}"
                         for i, d in enumerate(running_topk)]
        new_lines = [f"[N{i+1}] {format_doc(d[1], show_all=True)}"
                     for i, d in enumerate(batch)]
        ordered_ids = (
            [f"L{i+1}" for i in range(K_leaders)] +
            [f"N{i+1}" for i in range(len(batch))]
        )
        example_format = " > ".join(f"[{pid}]" for pid in ordered_ids[:K])

        prompt = (TOURNAMENT_PROMPT
                  .replace("{query}", query)
                  .replace("{leaders_block}", "\n".join(leaders_lines))
                  .replace("{new_block}", "\n".join(new_lines))
                  .replace("{K_leaders}", str(K_leaders))
                  .replace("{K}", str(K))
                  .replace("{B}", str(len(batch)))
                  .replace("{example_format}", example_format))

        valid_ids = ({f"L{i+1}" for i in range(K_leaders)} |
                     {f"N{i+1}" for i in range(len(batch))})

        parsed = []
        try:
            raw = call_ollama(prompt, model, endpoint, max_tokens=80)
            parsed = _parse_tournament_output(raw, valid_ids)
            if len(parsed) < K:
                retry_hint = (TOURNAMENT_RETRY_HINT
                              .replace("{K}", str(K))
                              .replace("{example_format}", example_format))
                raw2 = call_ollama(prompt + retry_hint, model, endpoint, max_tokens=80)
                parsed2 = _parse_tournament_output(raw2, valid_ids)
                if len(parsed2) >= len(parsed):
                    parsed = parsed2
        except Exception:
            pass

        if len(parsed) < K:
            parse_failures += 1
            continue  # keep running_topk unchanged

        id_to_doc = {}
        for i, d in enumerate(running_topk):
            id_to_doc[f"L{i+1}"] = d
        for i, d in enumerate(batch):
            id_to_doc[f"N{i+1}"] = d
        running_topk = [id_to_doc[pid] for pid in parsed[:K]]

    # Assign banded scores
    topk_ids = {d[0] for d in running_topk}
    result = {}
    # Top-K: rank 1..K → score 10..(10-K+1). With K=5: {10, 9, 8, 7, 6}. Band [6,10].
    for rank_idx, (docid, _, _) in enumerate(running_topk):
        tournament_rank = rank_idx + 1
        llm_score = 10.0 - (tournament_rank - 1)
        result[docid] = {"tournament_rank": tournament_rank, "llm_score": llm_score}

    # Eliminated: rescale their mFAR scores into [0, 4]
    eliminated = [(d[0], d[2]) for d in docs if d[0] not in topk_ids]
    if eliminated:
        mfar_vals = [m for _, m in eliminated]
        mn, mx = min(mfar_vals), max(mfar_vals)
        for docid, mfar in eliminated:
            if mx - mn < 1e-8:
                score = 2.0
            else:
                score = (mfar - mn) / (mx - mn) * 4.0
            result[docid] = {"tournament_rank": None, "llm_score": score}

    return result, parse_failures


def load_rerank_cache(cache_path, no_memory=False, memory_mode=None):
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
                if memory_mode is not None and entry.get("memory_mode") != memory_mode:
                    continue
                if _rerank_entry_status(entry) != "ok":
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

    # Discrimination memory (v3): load rules once per batch
    use_discrimination = (memory_version or "").startswith("discrimination") and not no_memory
    disc_rules = load_discrimination_rules() if use_discrimination else {}

    # Session-memory tournament rerank: uses memory_v1 Stage 1+2 cache for routing gate
    use_session = (memory_version or "").startswith("session") and not no_memory

    # Load Stage 1+2 cache (may be from a different model than the reranker)
    # For "discrimination" / "session" memory_versions, the Stage 1+2 cache is the same as memory_v1
    # (we reuse existing Stage 2 outputs; only Stage 3 scoring differs).
    if use_discrimination or use_session:
        stage12_version = "memory_v1"
    else:
        stage12_version = memory_version
    qwen3_cache = _load_qwen3_cache(split, memory_version=stage12_version, detect_model=detect_tag)

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
    expected_cache_mode = "tournament" if use_session else None
    existing = {
        key: score
        for key, score in load_rerank_cache(
            cache_path,
            no_memory=no_memory,
            memory_mode=expected_cache_mode,
        ).items()
        if key[0] in rerouted_qids
    }
    remaining = [(qid, did) for qid, did in pairs if (qid, did) not in existing]
    print(f"  Cache has {len(existing)} entries, {len(remaining)} remaining")

    if use_session:
        # Per-query tournament: a qid is rerun if ANY of its top-K docs is missing from cache.
        qids_all = sorted(rerouted_qids)
        qids_to_run = []
        for qid in qids_all:
            docids = [d for d, _ in retrieved.get(qid, [])[:top_k]]
            if not all((qid, did) in existing for did in docids):
                qids_to_run.append(qid)
        print(f"  Session tournament: {len(qids_all) - len(qids_to_run)} cached, {len(qids_to_run)} to run")
        if not qids_to_run:
            print("  All queries already tournament-scored")
            return existing

        write_lock = threading.Lock()
        done = [0]
        parse_fail_total = [0]

        with open(cache_path, "a") as f:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                for i, qid in enumerate(qids_to_run):
                    ep = endpoints[i % len(endpoints)]
                    docs_list = []
                    for did, mfar in retrieved.get(qid, [])[:top_k]:
                        doc_json = corpus.get(did, {"name": "?", "type": "?"})
                        docs_list.append((did, doc_json, float(mfar)))
                    fut = pool.submit(tournament_rerank_one_query,
                                      qid, queries[qid], docs_list, model, ep)
                    futures[fut] = qid

                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result, pf = future.result()
                    except Exception as e:
                        with write_lock:
                            print(f"  Tournament failed for qid={qid}: {e}")
                        continue
                    with write_lock:
                        for docid, info in result.items():
                            entry = {
                                "qid": qid,
                                "docid": docid,
                                "llm_score": info["llm_score"],
                                "tournament_rank": info["tournament_rank"],
                                "raw": "",
                                "doc_format_version": DOC_FORMAT_VERSION,
                                "no_memory": False,
                                "memory_mode": "tournament",
                            }
                            f.write(json.dumps(entry) + "\n")
                            existing[(qid, docid)] = info["llm_score"]
                        f.flush()
                        parse_fail_total[0] += pf
                        done[0] += 1
                        if done[0] % 50 == 0:
                            print(f"  Tournament: {done[0]}/{len(qids_to_run)} queries  "
                                  f"(parse fails so far: {parse_fail_total[0]})")

        print(f"  Done: {done[0]} tournaments, total parse failures: {parse_fail_total[0]}")
        return existing

    if not remaining:
        print("  All pairs already scored")
        return existing

    # Score with thread pool
    write_lock = threading.Lock()
    done = [0]
    error_count = [0]

    with open(cache_path, "a") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, (qid, docid) in enumerate(remaining):
                ep = endpoints[i % len(endpoints)]
                qentry = qwen3_cache.get(qid, {})
                boost = [] if no_memory else qentry.get("boost_fields", [])
                doc_json = corpus.get(docid, {"name": "?", "type": "?"})
                disc_block = None
                if use_discrimination:
                    disc_block = get_discrimination_block(
                        qentry.get("answer_type"), qentry.get("negation_pattern"),
                        rules=disc_rules)
                fut = pool.submit(score_one, qid, docid, queries[qid], doc_json, boost, model, ep,
                                  show_all=no_memory, discrimination_block=disc_block)
                futures[fut] = (qid, docid)

            for future in as_completed(futures):
                entry = future.result()
                with write_lock:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    if entry.get("status") == "ok":
                        existing[(entry["qid"], entry["docid"])] = entry["llm_score"]
                    else:
                        error_count[0] += 1
                    done[0] += 1
                    if done[0] % 500 == 0:
                        print(f"  Scored {done[0]}/{len(remaining)} "
                              f"({error_count[0]} errors)")

    print(f"  Done: {len(existing)} cached scores, {error_count[0]} errors")
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
                cache_mode = None
                mv = (args.memory_version or "")
                if mv.startswith("session") and not getattr(args, "no_memory", False):
                    cache_mode = "tournament"
                rerank_scores = load_rerank_cache(
                    cache_path,
                    no_memory=getattr(args, "no_memory", False),
                    memory_mode=cache_mode,
                )
                print(f"  Loaded {len(rerank_scores)} rerank scores from {cache_path}")

                detect_tag = _model_tag(getattr(args, 'detect_model', None) or args.model)
                # Discrimination / session modes reuse memory_v1 Stage 1+2 cache
                stage12_ver = args.memory_version
                if (mv.startswith("discrimination") or mv.startswith("session")) \
                        and not getattr(args, "no_memory", False):
                    stage12_ver = "memory_v1"
                qwen3_cache = _load_qwen3_cache(split, memory_version=stage12_ver, detect_model=detect_tag)
                merge_and_write_qres(split, rerank_scores, qwen3_cache,
                                     alpha=alpha, top_k=args.top_k, output_dir=out_dir)

            # Evaluate all splits together
            print(f"\n  Evaluating α={alpha} → {out_dir}")
            from failure_analysis.type_b_memory.rerank.scoring.evaluate import main as eval_main
            eval_args = ["evaluate_memory.py",
                         "--baseline_dir", "output/contriever/prime_eval",
                         "--memory_dir", out_dir,
                         "--splits"] + args.splits
            if args.memory_version:
                # For discrimination / session modes, evaluate's grouping uses memory_v1 Stage 1+2 cache
                eval_mem_ver = args.memory_version
                mv = args.memory_version
                if (mv.startswith("discrimination") or mv.startswith("session")) \
                        and not getattr(args, "no_memory", False):
                    eval_mem_ver = "memory_v1"
                eval_args += ["--memory_version", eval_mem_ver]
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
