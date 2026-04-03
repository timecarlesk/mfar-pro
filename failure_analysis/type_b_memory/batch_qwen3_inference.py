"""
Batch Qwen3 inference for field re-routing.

Two-stage LLM pipeline:
  Stage 1 (Detect): Qwen3 decides if query needs field re-routing (yes/no)
           Short prompt, runs on ALL queries. ~400ms each.
  Stage 2 (Route):  For flagged queries, Qwen3 determines boost/suppress fields.
           Detailed prompt with known failure patterns.

Results are cached as JSONL for resume support.

Run from project root:
  python failure_analysis/type_b_memory/batch_qwen3_inference.py --splits train
  python failure_analysis/type_b_memory/batch_qwen3_inference.py --splits val test
"""

import argparse
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from failure_analysis.utils import load_queries
from failure_analysis.negation.negation_ablation import NEGATION_PATTERN

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = "data/prime"
CACHE_DIR = "output/failure_analysis/type_b_memory/cache"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"

VALID_FIELDS = [
    "name", "type", "details", "indication", "contraindication",
    "side effect", "target", "associated with", "ppi", "interacts with",
    "parent-child", "phenotype present", "phenotype absent",
    "expression present", "expression absent", "linked to", "carrier",
    "enzyme", "transporter", "synergistic interaction", "off-label use", "source",
]

# Qwen3 may output underscores instead of spaces — normalize both ways
_FIELD_ALIAS = {}
for f in VALID_FIELDS:
    _FIELD_ALIAS[f] = f
    _FIELD_ALIAS[f.replace(" ", "_")] = f
    _FIELD_ALIAS[f.replace("-", "_")] = f

# ── Stage 1 Prompt: Detection (short, yes/no) ───────────────────────────────

DETECT_PROMPT = """\
Does this biomedical knowledge graph query contain negation or semantic constraints that would cause a field-based retrieval system to search the WRONG field?

Examples that NEED re-routing:
- "drugs NOT indicated for diabetes"
- "genes not expressed in liver"
- "diseases lacking approved treatment"
- "conditions that should not be treated with aspirin"
- "drugs to avoid for hypertension"
- "proteins inappropriate for targeting"

Examples that do NOT need re-routing:
- "drugs indicated for diabetes"
- "genes expressed in liver"
- "non-small cell lung cancer" (part of entity name)
- "proteins involved in non-homologous end joining" (technical term)

Query: "{query}"

Answer ONLY "yes" or "no"."""

# ── Stage 2 Prompt: Field routing (detailed) ────────────────────────────────

# Default fallback if no memory_context file exists yet (first run on train)
_DEFAULT_MEMORY_CONTEXT = """\
Known failure patterns:
- "not indicated for X" / "should not treat X" → system incorrectly searches "indication", but answer is in "contraindication" (if answer is a drug) or "associated with" (if answer is a disease)
- "not expressed in X" / "lacking expression" → system searches "expression present", but answer is in "expression absent"
- "phenotype not observed" → system searches "phenotype present", but answer is in "phenotype absent"

Important: Not all entity types have all fields. For example:
- Diseases do NOT have indication/contraindication/side effect/target fields
- Genes/proteins do NOT have indication/contraindication fields
- Only drugs have indication, contraindication, side effect, target, carrier, enzyme, transporter
Consider what fields the ANSWER entity type actually has before suggesting boost/suppress."""

def load_memory_context():
    """Load training-data-derived memory context, or fall back to default."""
    ctx_path = os.path.join(ANALYSIS_DIR, "memory_context_train.txt")
    if os.path.exists(ctx_path):
        with open(ctx_path) as f:
            ctx = f.read().strip()
        print(f"  Loaded memory context from {ctx_path} ({len(ctx)} chars)")
        return ctx
    print(f"  No memory context found at {ctx_path}, using default")
    return _DEFAULT_MEMORY_CONTEXT


def build_route_prompt(query, memory_context):
    """Build Stage 2 prompt with dynamic memory context."""
    return f"""\
This query was flagged because it contains negation or constraints that cause the retrieval system to search the WRONG field:
"{query}"

The retrieval system uses field-level scoring with learned weights.
When a query contains negation, the system often routes to the WRONG field.

{memory_context}

Available fields:
indication, contraindication, side effect, target, associated with, ppi, interacts with, parent-child, phenotype present, phenotype absent, expression present, expression absent, details, name, type, linked to, carrier, enzyme, transporter, synergistic interaction, off-label use, source

Tasks:
1. What type of entity is the ANSWER likely to be? (drug/disease/gene/protein/effect/phenotype)
2. Given the negation, which fields is the system INCORRECTLY searching? (suppress these)
3. Which fields SHOULD it search instead? (boost these)

Rules:
- boost and suppress must NOT overlap
- Only list fields where the negation causes a clear mismatch

Output ONLY a JSON object:
{{"answer_type": "...", "boost_fields": [...], "suppress_fields": [...]}}"""


# ── Ollama API ───────────────────────────────────────────────────────────────

def call_ollama(prompt, model, endpoint, max_tokens=150):
    """Call Ollama API and return raw text response."""
    url = f"{endpoint}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_detect_output(raw_output):
    """Parse Stage 1 yes/no output."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip().lower()
    # Look for yes/no in the response
    if "yes" in cleaned:
        return True
    return False


def parse_route_output(raw_output):
    """Parse Stage 2 JSON output with boost/suppress fields."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(cleaned[start:end+1])

            answer_type = result.get("answer_type")
            boost_fields = result.get("boost_fields")
            suppress_fields = result.get("suppress_fields")

            if isinstance(boost_fields, list):
                boost_fields = [_FIELD_ALIAS[f] for f in boost_fields if f in _FIELD_ALIAS]
            else:
                boost_fields = []

            if isinstance(suppress_fields, list):
                suppress_fields = [_FIELD_ALIAS[f] for f in suppress_fields if f in _FIELD_ALIAS]
            else:
                suppress_fields = []

            # Remove overlaps — keep in boost, remove from suppress
            boost_set = set(boost_fields)
            suppress_fields = [f for f in suppress_fields if f not in boost_set]

            return {
                "answer_type": answer_type,
                "boost_fields": boost_fields,
                "suppress_fields": suppress_fields,
            }
        except json.JSONDecodeError:
            pass

    return {"answer_type": None, "boost_fields": [], "suppress_fields": []}


# ── Batch Processing ─────────────────────────────────────────────────────────

def load_cache(cache_path):
    """Load existing JSONL cache for resume support."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["qid"]] = entry
    return cache


def process_one(qid, query_text, model, endpoint, memory_context, detect_mode="llm"):
    """Two-stage processing for a single query.

    detect_mode: "llm" = Qwen3 Stage 1, "regex" = regex NEGATION_PATTERN
    """
    # Stage 1: detect
    if detect_mode == "regex":
        needs_reroute = bool(NEGATION_PATTERN.search(query_text))
        detect_raw = f"regex:{needs_reroute}"
    else:
        try:
            detect_raw = call_ollama(
                DETECT_PROMPT.replace("{query}", query_text), model, endpoint, max_tokens=10
            )
            needs_reroute = parse_detect_output(detect_raw)
        except Exception as e:
            return {"qid": qid, "query": query_text, "needs_reroute": False,
                    "detect_raw": str(e), "raw_output": "",
                    "answer_type": None, "boost_fields": [], "suppress_fields": []}

    if not needs_reroute:
        return {"qid": qid, "query": query_text, "needs_reroute": False,
                "detect_raw": detect_raw, "raw_output": "",
                "answer_type": None, "boost_fields": [], "suppress_fields": []}

    # Stage 2: route (with memory context from training data)
    try:
        route_raw = call_ollama(
            build_route_prompt(query_text, memory_context), model, endpoint, max_tokens=150
        )
        parsed = parse_route_output(route_raw)
        return {"qid": qid, "query": query_text, "needs_reroute": True,
                "detect_raw": detect_raw, "raw_output": route_raw, **parsed}
    except Exception as e:
        return {"qid": qid, "query": query_text, "needs_reroute": True,
                "detect_raw": detect_raw, "raw_output": str(e),
                "answer_type": None, "boost_fields": [], "suppress_fields": []}


def batch_classify_queries(queries, cache_path, model, endpoints, workers=1, detect_mode="llm"):
    """
    Two-stage pipeline:
      Stage 1: detect (llm or regex)
      Stage 2: Qwen3 field routing (only flagged queries)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    memory_context = load_memory_context()

    cache = load_cache(cache_path)
    print(f"  Cache has {len(cache)} existing entries")

    remaining = [(qid, q) for qid, q in queries.items() if qid not in cache]
    print(f"  {len(remaining)} queries remaining")
    print(f"  Stage 1: {detect_mode}, {len(endpoints)} endpoint(s), {workers} worker(s)")

    if not remaining:
        return cache

    write_lock = threading.Lock()
    done = [0]
    stage2_count = [0]

    with open(cache_path, "a") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, (qid, q) in enumerate(remaining):
                ep = endpoints[i % len(endpoints)]
                fut = pool.submit(process_one, qid, q, model, ep, memory_context, detect_mode)
                futures[fut] = qid

            for future in as_completed(futures):
                entry = future.result()
                with write_lock:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    cache[entry["qid"]] = entry
                    done[0] += 1
                    if entry.get("needs_reroute"):
                        stage2_count[0] += 1
                    if done[0] % 100 == 0:
                        print(f"  Processed {done[0]}/{len(remaining)} "
                              f"({stage2_count[0]} needed routing)")

    print(f"  Done: {len(cache)} total, {stage2_count[0]} went to Stage 2")
    return cache


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Qwen3 field re-routing")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Splits to classify (default: train)")
    parser.add_argument("--model", default="qwen3:8b",
                        help="Ollama model name (default: qwen3:8b)")
    parser.add_argument("--endpoints", default="http://127.0.0.1:11434",
                        help="Ollama API endpoints, comma-separated for parallel instances")
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent workers (default: number of endpoints)")
    parser.add_argument("--detect", choices=["llm", "regex"], default="llm",
                        help="Stage 1 detection mode: llm (Qwen3) or regex (default: llm)")
    args = parser.parse_args()

    endpoints = [ep.strip() for ep in args.endpoints.split(",")]
    workers = args.workers if args.workers else len(endpoints)

    os.makedirs(CACHE_DIR, exist_ok=True)

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"  Processing split: {split}")
        print(f"{'='*60}")

        queries = load_queries(DATA_DIR, split)
        cache_path = os.path.join(CACHE_DIR, f"qwen3_cache_{split}.jsonl")
        cache = batch_classify_queries(queries, cache_path, args.model, endpoints, workers, args.detect)

        # Summary statistics
        from collections import Counter
        flagged = [e for e in cache.values() if e.get("needs_reroute")]
        has_boost = [e for e in flagged if e.get("boost_fields")]
        not_flagged = [e for e in cache.values() if not e.get("needs_reroute")]

        print(f"\n  Summary for {split}:")
        print(f"    Total queries:        {len(cache)}")
        print(f"    Stage 1 flagged:      {len(flagged)} ({100*len(flagged)/len(cache):.1f}%)")
        print(f"    Stage 1 skipped:      {len(not_flagged)} ({100*len(not_flagged)/len(cache):.1f}%)")
        print(f"    With boost fields:    {len(has_boost)}")

        if has_boost:
            boost_counts = Counter()
            suppress_counts = Counter()
            answer_types = Counter()
            for e in has_boost:
                for f in e.get("boost_fields", []):
                    boost_counts[f] += 1
                for f in e.get("suppress_fields", []):
                    suppress_counts[f] += 1
                answer_types[e.get("answer_type", "unknown")] += 1
            print(f"    Top boost fields:")
            for f, c in boost_counts.most_common(5):
                print(f"      {f}: {c}")
            print(f"    Top suppress fields:")
            for f, c in suppress_counts.most_common(5):
                print(f"      {f}: {c}")
            print(f"    Answer types:")
            for t, c in answer_types.most_common():
                print(f"      {t}: {c}")


if __name__ == "__main__":
    main()
