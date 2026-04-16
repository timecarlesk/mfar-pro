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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
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

_EXTRA_FIELD_ALIASES = {
    "contraindications": "contraindication",
    "indications": "indication",
    "side effects": "side effect",
    "adverse effect": "side effect",
    "adverse effects": "side effect",
    "association": "associated with",
    "associations": "associated with",
    "interaction": "interacts with",
    "interactions": "interacts with",
    "off label use": "off-label use",
}
for alias, target in _EXTRA_FIELD_ALIASES.items():
    _FIELD_ALIAS[alias] = target
    _FIELD_ALIAS[alias.replace(" ", "_")] = target
for alias, target in list(_FIELD_ALIAS.items()):
    _FIELD_ALIAS[alias.lower()] = target

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

# ── Stage 1.5 Prompt: Negation pattern classification ───────────────────────

CLASSIFY_PROMPT = """\
This query contains negation that affects retrieval:
"{query}"

What type of negation is this? Choose the BEST match:
- not_indicated: drug/treatment should not be used, not recommended, not suitable
- not_expressed: gene/protein not expressed, lacking expression, unexpressed in tissue
- lacking_treatment: no approved drugs/treatments exist for a condition
- avoid_contraindicated: contraindicated, should avoid, preclude use of
- not_associated: not associated/related/linked to
- phenotype_absent: phenotype/symptom not observed, absent
- no_side_effect: without side effects, no adverse effects
- other: none of the above

ALSO: What type of entity is the query ASKING YOU TO FIND?
- answer_type is what you're searching for, NOT what's mentioned
- "Which anatomical structures lack expression of gene X?" → anatomy
- "Which drugs should not treat disease X?" → drug
- "Which diseases should not be treated with drug X?" → disease

Output ONLY a JSON object:
{{"negation_pattern": "...", "answer_type": "..."}}"""

# ── Stage 2 Prompt: Field routing (detailed) ────────────────────────────────

# Default fallback if no memory_context file exists yet (first run on train)
_DEFAULT_MEMORY_CONTEXT = """\
Retrieval re-routing rules:

When answer_type=drug and query contains "not indicated" or "should not treat":
  Recommended boost_fields: ["contraindication", "synergistic interaction"]

When answer_type=disease and query contains "not indicated" or "should not treat":
  Recommended boost_fields: ["associated with", "parent-child", "contraindication"]

When answer_type=disease and query contains "lacking treatment":
  Recommended boost_fields: ["associated with", "parent-child", "phenotype present"]

When answer_type=gene/protein and query contains "not expressed":
  Recommended boost_fields: ["expression absent"]

When answer_type=anatomy and query contains "not expressed" or "lack expression":
  Recommended boost_fields: ["expression absent", "expression present", "parent-child"]

Important: Not all entity types have all fields.
- drug entities have: indication, contraindication, side effect, target, carrier, enzyme, transporter, synergistic interaction
- disease entities have: associated with, phenotype present/absent, parent-child, indication, contraindication
- gene/protein entities have: ppi, interacts with, expression present/absent, associated with, target
- anatomy entities have: expression present/absent, parent-child"""

_memory_kg = None  # Global cached KG instance

def load_memory_kg():
    """Load Memory KG if available, return MemoryKG instance or None."""
    global _memory_kg
    if _memory_kg is not None:
        return _memory_kg

    kg_path = os.path.join(ANALYSIS_DIR, "memory_kg.json")
    if os.path.exists(kg_path):
        from failure_analysis.type_b_memory.rerank.shared.memory_kg import MemoryKG
        _memory_kg = MemoryKG.from_json(kg_path)
        print(f"  Loaded Memory KG from {kg_path} ({_memory_kg.summary()})")
        return _memory_kg
    return None


def _load_text_memory_context(path, label):
    """Load a text memory file and log its source."""
    with open(path) as f:
        ctx = f.read().strip()
    print(f"  Loaded memory context {label} from {path} ({len(ctx)} chars)")
    return ctx


def load_memory_context(memory_version=None):
    """Load the selected memory context.

    Returns (memory_context, memory_kg). When memory_version is None, keep the
    historical auto fallback order: KG → v2 text → v1 text → default.
    """
    v2_path = os.path.join(ANALYSIS_DIR, "memory_context_train_v2.txt")
    v1_path = os.path.join(ANALYSIS_DIR, "memory_context_train.txt")

    if memory_version == "memory_kg":
        kg = load_memory_kg()
        if kg is None:
            raise FileNotFoundError(
                f"Requested --memory_version=memory_kg but no KG exists at "
                f"{os.path.join(ANALYSIS_DIR, 'memory_kg.json')}"
            )
        ctx = kg.format_full_context()
        print(f"  Using Memory KG as context ({len(ctx)} chars)")
        return ctx, kg

    if memory_version == "memory_v2":
        if not os.path.exists(v2_path):
            raise FileNotFoundError(
                f"Requested --memory_version=memory_v2 but file is missing: {v2_path}"
            )
        return _load_text_memory_context(v2_path, "v2"), None

    if memory_version == "memory_v1":
        if not os.path.exists(v1_path):
            raise FileNotFoundError(
                f"Requested --memory_version=memory_v1 but file is missing: {v1_path}"
            )
        return _load_text_memory_context(v1_path, "v1"), None

    if memory_version is not None:
        raise ValueError(
            f"Unknown memory_version={memory_version!r}; expected one of "
            "'memory_v1', 'memory_v2', 'memory_kg', or omitted for auto selection."
        )

    # Historical auto-selection for legacy callers.
    kg = load_memory_kg()
    if kg is not None:
        ctx = kg.format_full_context()
        print(f"  Using Memory KG as context ({len(ctx)} chars)")
        return ctx, kg

    if os.path.exists(v2_path):
        return _load_text_memory_context(v2_path, "v2"), None

    if os.path.exists(v1_path):
        return _load_text_memory_context(v1_path, "v1"), None

    print(f"  No memory context found, using default")
    return _DEFAULT_MEMORY_CONTEXT, None


def _find_matching_rule(memory_context, answer_type, negation_pattern):
    """Extract the matching rule from memory context for this answer_type + pattern."""
    if not memory_context:
        return None
    # Look for line starting with "When answer_type=X and query contains Y"
    best_match = None
    for line in memory_context.split("\n"):
        line_lower = line.lower().strip()
        if not line_lower.startswith("when answer_type="):
            continue
        # Check if this rule matches
        if answer_type and answer_type.lower() in line_lower:
            if negation_pattern and negation_pattern.replace("_", " ") in line_lower:
                # Exact match on both — collect this rule + next lines
                idx = memory_context.find(line)
                end = memory_context.find("\nWhen ", idx + 1)
                if end == -1:
                    end = memory_context.find("\nImportant:", idx + 1)
                if end == -1:
                    end = idx + 500
                best_match = memory_context[idx:end].strip()
                break
            elif best_match is None:
                # Partial match on answer_type only
                idx = memory_context.find(line)
                end = memory_context.find("\nWhen ", idx + 1)
                if end == -1:
                    end = idx + 500
                best_match = memory_context[idx:end].strip()
    return best_match


def build_route_prompt(query, memory_context, answer_type=None, negation_pattern=None,
                       prompt_format="natural", memory_kg=None):
    """Build Stage 2 prompt with matched rule from memory context.

    prompt_format: "natural" = verbose NL description, "structured" = compact KV format
    Uses KG query only when the selected memory source is KG.
    """
    if memory_kg is not None and answer_type and negation_pattern:
        if prompt_format == "structured":
            matched_rule = memory_kg.format_structured_for_prompt(answer_type, negation_pattern)
        else:
            matched_rule = memory_kg.format_for_prompt(answer_type, negation_pattern)
    else:
        matched_rule = _find_matching_rule(memory_context, answer_type, negation_pattern)

    if matched_rule:
        rule_text = f"Matched rule from training data:\n{matched_rule}"
    else:
        rule_text = f"No exact rule matched for answer_type={answer_type}, pattern={negation_pattern}.\n"
        rule_text += "Use the full rule set:\n" + memory_context

    return f"""\
This query contains negation that may cause the retrieval system to search the wrong field:
"{query}"

The system has classified:
- answer_type: {answer_type} (the entity type the query is searching for)
- negation_pattern: {negation_pattern}

{rule_text}

Based on the rule above, which fields should be boosted for this query?
If the rule doesn't fit, reason about which fields the answer entity ({answer_type}) would actually have populated.

Output ONLY a JSON object:
{{"answer_type": "{answer_type}", "boost_fields": [...]}}"""


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
    if "yes" in cleaned:
        return True
    return False


VALID_NEG_PATTERNS = [
    "not_indicated", "not_expressed", "lacking_treatment",
    "avoid_contraindicated", "not_associated", "phenotype_absent",
    "no_side_effect", "other",
]

VALID_ANSWER_TYPES = [
    "drug", "disease", "gene/protein", "anatomy", "phenotype", "pathway",
]

def _extract_first_json_object(text):
    """Return the first valid JSON object embedded in model output."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _normalize_field_name(field):
    """Map model field variants onto the PRIME field whitelist."""
    if not isinstance(field, str):
        return None
    raw = field.strip()
    if not raw:
        return None
    candidates = [
        raw,
        raw.lower(),
        raw.replace("-", "_"),
        raw.lower().replace("-", "_"),
        raw.replace("-", " "),
        raw.lower().replace("-", " "),
        raw.replace("_", " "),
        raw.lower().replace("_", " "),
    ]
    for candidate in candidates:
        mapped = _FIELD_ALIAS.get(candidate)
        if mapped is not None:
            return mapped
    return None


def _normalize_field_list(fields):
    """Normalize a list of fields, preserving unknown ones for diagnostics."""
    normalized = []
    unmapped = []
    seen = set()
    seen_unmapped = set()
    if not isinstance(fields, list):
        return normalized, unmapped

    for field in fields:
        mapped = _normalize_field_name(field)
        if mapped is not None:
            if mapped not in seen:
                normalized.append(mapped)
                seen.add(mapped)
            continue
        if isinstance(field, str):
            cleaned = field.strip()
            if cleaned and cleaned not in seen_unmapped:
                unmapped.append(cleaned)
                seen_unmapped.add(cleaned)
    return normalized, unmapped


def parse_classify_output(raw_output):
    """Parse Stage 1.5 JSON output with negation_pattern and answer_type."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip()

    result = _extract_first_json_object(cleaned)
    if isinstance(result, dict):
        neg_pat = result.get("negation_pattern", "other")
        answer_type = result.get("answer_type", "unknown")

        if neg_pat not in VALID_NEG_PATTERNS:
            neg_pat = "other"

        # Normalize answer_type
        at_lower = (answer_type or "").lower().strip()
        if at_lower in ("gene", "protein", "gene/protein"):
            answer_type = "gene/protein"
        elif at_lower in ("disease", "condition"):
            answer_type = "disease"
        elif at_lower in ("anatomy", "anatomical structure", "anatomical part",
                          "body structure", "tissue", "cellular structure"):
            answer_type = "anatomy"
        elif at_lower in ("drug", "medication"):
            answer_type = "drug"
        elif at_lower in ("phenotype", "effect/phenotype"):
            answer_type = "phenotype"
        else:
            answer_type = at_lower or "unknown"

        return neg_pat, answer_type

    return "other", "unknown"


def parse_route_output(raw_output):
    """Parse Stage 2 JSON output with boost/suppress fields."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip()

    result = _extract_first_json_object(cleaned)
    if isinstance(result, dict):
        answer_type = result.get("answer_type")
        boost_fields, unmapped_boost_fields = _normalize_field_list(result.get("boost_fields"))
        suppress_fields, unmapped_suppress_fields = _normalize_field_list(result.get("suppress_fields"))

        # Remove overlaps — keep in boost, remove from suppress
        boost_set = set(boost_fields)
        suppress_fields = [f for f in suppress_fields if f not in boost_set]

        return {
            "answer_type": answer_type,
            "boost_fields": boost_fields,
            "suppress_fields": suppress_fields,
            "unmapped_boost_fields": unmapped_boost_fields,
            "unmapped_suppress_fields": unmapped_suppress_fields,
        }

    return {
        "answer_type": None,
        "boost_fields": [],
        "suppress_fields": [],
        "unmapped_boost_fields": [],
        "unmapped_suppress_fields": [],
    }


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


def process_one(qid, query_text, model, endpoint, memory_context, memory_kg=None,
                detect_mode="llm", prompt_format="natural"):
    """Three-stage processing for a single query.

    Stage 1: Detect — does this query need re-routing? (yes/no)
    Stage 1.5: Classify — what negation pattern + answer type? (JSON)
    Stage 2: Route — which fields to boost? (JSON, uses matched memory rule)
    prompt_format: "natural" or "structured" — how memory rule is presented to Stage 2
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
                    "detect_raw": str(e), "classify_raw": "", "raw_output": "",
                    "negation_pattern": None, "answer_type": None,
                    "boost_fields": [], "suppress_fields": []}

    if not needs_reroute:
        return {"qid": qid, "query": query_text, "needs_reroute": False,
                "detect_raw": detect_raw, "classify_raw": "", "raw_output": "",
                "negation_pattern": None, "answer_type": None,
                "boost_fields": [], "suppress_fields": []}

    # Stage 1.5: classify negation pattern + answer type
    try:
        classify_raw = call_ollama(
            CLASSIFY_PROMPT.replace("{query}", query_text), model, endpoint, max_tokens=30
        )
        neg_pattern, answer_type = parse_classify_output(classify_raw)
    except Exception as e:
        classify_raw = str(e)
        neg_pattern, answer_type = "other", "unknown"

    # Stage 2: route (with matched memory rule)
    try:
        route_raw = call_ollama(
            build_route_prompt(
                query_text,
                memory_context,
                answer_type,
                neg_pattern,
                prompt_format=prompt_format,
                memory_kg=memory_kg,
            ),
            model, endpoint, max_tokens=150
        )
        parsed = parse_route_output(route_raw)
        return {"qid": qid, "query": query_text, "needs_reroute": True,
                "detect_raw": detect_raw, "classify_raw": classify_raw,
                "negation_pattern": neg_pattern,
                "raw_output": route_raw, **parsed}
    except Exception as e:
        return {"qid": qid, "query": query_text, "needs_reroute": True,
                "detect_raw": detect_raw, "classify_raw": classify_raw,
                "negation_pattern": neg_pattern,
                "raw_output": str(e),
                "answer_type": answer_type, "boost_fields": [], "suppress_fields": []}


def batch_classify_queries(queries, cache_path, model, endpoints, workers=1, detect_mode="llm",
                           prompt_format="natural", memory_version=None):
    """
    Three-stage pipeline:
      Stage 1: detect (llm or regex)
      Stage 1.5: classify negation pattern + answer type
      Stage 2: field routing (only flagged queries)
    prompt_format: "natural" or "structured"
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    memory_context, memory_kg = load_memory_context(memory_version)

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
                fut = pool.submit(
                    process_one,
                    qid,
                    q,
                    model,
                    ep,
                    memory_context,
                    memory_kg,
                    detect_mode,
                    prompt_format,
                )
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
    parser.add_argument("--memory_version", default=None,
                        help="Memory version subfolder (e.g. 'memory_v2'). Train always writes to shared/.")
    parser.add_argument("--prompt_format", choices=["natural", "structured"], default="natural",
                        help="Stage 2 memory format: natural (verbose NL) or structured (compact KV)")
    args = parser.parse_args()

    endpoints = [ep.strip() for ep in args.endpoints.split(",")]
    workers = args.workers if args.workers else len(endpoints)

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"  Processing split: {split}")
        print(f"{'='*60}")

        queries = load_queries(DATA_DIR, split)

        # Path: stage12/{model_tag}/shared/ (train) or stage12/{model_tag}/{memory_version}/ (val/test)
        model_tag = args.model.replace(":", "_").replace("/", "_")
        if split == "train":
            cache_dir = os.path.join(CACHE_DIR, "stage12", model_tag, "shared")
        elif args.memory_version:
            cache_dir = os.path.join(CACHE_DIR, "stage12", model_tag, args.memory_version)
        else:
            cache_dir = os.path.join(CACHE_DIR, "stage12", model_tag)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"qwen3_cache_{split}.jsonl")
        cache = batch_classify_queries(
            queries,
            cache_path,
            args.model,
            endpoints,
            workers,
            args.detect,
            args.prompt_format,
            args.memory_version,
        )

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
