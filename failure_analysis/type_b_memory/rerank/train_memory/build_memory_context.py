"""
Build Field Confusion Matrix and Memory Context for Stage 2.

Groups rerouted queries by (answer_type, negation_pattern), then for each
group computes which fields are actually populated in gold docs. This tells
us the "correct boost" for each query pattern.

Output:
  - field_confusion_{split}.json: detailed per-query data
  - memory_context_{split}.txt: direct boost recommendations for Stage 2 prompt

Run from project root:
  python failure_analysis/type_b_memory/build_field_confusion.py
  python failure_analysis/type_b_memory/build_field_confusion.py --splits val test
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from failure_analysis.utils import RELATION_FIELDS, load_corpus

DATA_DIR = "data/prime"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"


# ── Negation Pattern Classification ─────────────────────────────────────────

NEGATION_PATTERNS = [
    ("not_indicated", re.compile(
        r"not\s+(?:indicated|treat|recommended|advised|suitable|safe|managed|prescribed)"
        r"|should\s+not\s+(?:be\s+)?(?:treat|manage|prescri|administer|use)"
        r"|renders?\s+.*(?:unsuitable|contraindicated)"
        r"|preclude\s+(?:the\s+)?(?:use|administration)"
        r"|won'?t\s+(?:exacerbat|worsen)",
        re.IGNORECASE)),
    ("not_expressed", re.compile(
        r"not\s+expressed|lack(?:s|ing)?\s+(?:the\s+)?(?:genetic\s+|proteomic\s+)?expression"
        r"|unexpressed|not\s+(?:found|detected|present)\s+in"
        r"|lack\s+(?:the\s+)?(?:gene|protein)\s+(?:or\s+protein\s+)?expression",
        re.IGNORECASE)),
    ("lacking_treatment", re.compile(
        r"lack(?:s|ing)?\s+(?:any\s+)?(?:approved\s+)?(?:treat|drug|medic|therap|pharmacolog)"
        r"|no\s+drugs?\s+(?:indicated|available|approved)"
        r"|(?:have|has)\s+no\s+(?:drugs?|medications?)\s+(?:indicated|for)",
        re.IGNORECASE)),
    ("not_associated", re.compile(
        r"not\s+(?:associated|related|linked|connected|involved)"
        r"|(?:un)(?:associated|related|linked)",
        re.IGNORECASE)),
    ("no_side_effect", re.compile(
        r"(?:no|without|lacking)\s+(?:significant\s+)?(?:side|adverse)",
        re.IGNORECASE)),
    ("avoid_contraindicated", re.compile(
        r"(?:avoid|contraindic)"
        r"|disqualifying\s+factor"
        r"|should\s+not\s+be\s+(?:given|combined)",
        re.IGNORECASE)),
    ("phenotype_absent", re.compile(
        r"(?:absent|not\s+observed|phenotype.*not|lack.*phenotype)",
        re.IGNORECASE)),
]


def classify_negation_pattern(query_text):
    """Classify query into a negation pattern bucket."""
    for name, pattern in NEGATION_PATTERNS:
        if pattern.search(query_text):
            return name
    return "other"


def normalize_answer_type(answer_type):
    """Normalize answer types to a small set of categories."""
    if not answer_type:
        return "unknown"
    at = answer_type.lower().strip()
    if at in ("gene", "protein", "gene/protein"):
        return "gene/protein"
    if at in ("disease", "condition", "medical condition"):
        return "disease"
    if at in ("drug", "medication"):
        return "drug"
    if at in ("anatomy", "anatomical structure", "anatomical part",
              "anatomical region", "body part", "body structure",
              "tissue/organ", "tissue/organs", "cellular structure",
              "structure"):
        return "anatomy"
    if at in ("phenotype", "effect/phenotype"):
        return "phenotype"
    if at in ("pathway", "biological process"):
        return "pathway"
    return at


# ── Corpus Loading ───────────────────────────────────────────────────────────

def load_corpus_raw(data_dir):
    """Load corpus with full JSON for field inspection."""
    corpus = {}
    with open(f"{data_dir}/corpus") as f:
        for line in f:
            idx, json_str = line.strip().split("\t", 1)
            doc = json.loads(json_str)
            corpus[idx] = doc
    print(f"  Loaded {len(corpus):,} raw corpus documents")
    return corpus


def check_field_populated(doc, field_name):
    """Check if a field is non-empty in the doc."""
    val = doc.get(field_name)
    if val is None:
        return False
    if isinstance(val, (dict, list)):
        return len(val) > 0
    if isinstance(val, str):
        return len(val.strip()) > 0
    return bool(val)


# ── Analysis ─────────────────────────────────────────────────────────────────

def build_confusion(rerouted_queries, corpus_raw):
    """Build field confusion grouped by (gold_entity_type, negation_pattern).

    Uses gold doc's actual entity type (ground truth) instead of Qwen3's
    predicted answer_type, which has 30-70% error rate on some groups.
    """
    group_field_counts = defaultdict(lambda: defaultdict(int))
    group_doc_types = defaultdict(lambda: Counter())
    group_query_count = defaultdict(set)  # count unique queries per group
    per_query_details = []

    for entry in rerouted_queries:
        qwen3_answer_type = normalize_answer_type(entry.get("answer_type"))
        neg_pattern = entry.get("negation_pattern") or classify_negation_pattern(entry.get("query", ""))
        gold_ids = entry.get("gold_ids", [])

        query_detail = {
            "qid": entry["qid"],
            "query": entry["query"],
            "qwen3_answer_type": qwen3_answer_type,
            "negation_pattern": neg_pattern,
            "boost_fields": entry.get("boost_fields", []),
            "gold_docs": [],
        }

        for doc_id in gold_ids:
            if doc_id not in corpus_raw:
                continue
            doc = corpus_raw[doc_id]

            # Use gold doc's actual entity type for grouping (ground truth)
            gold_type = normalize_answer_type(doc.get("type", "unknown"))
            group_key = f"{gold_type}|{neg_pattern}"
            group_query_count[group_key].add(entry["qid"])

            populated = {}
            for field in RELATION_FIELDS:
                populated[field] = check_field_populated(doc, field)
                if populated[field]:
                    group_field_counts[group_key][field] += 1

            group_doc_types[group_key][doc.get("type", "unknown")] += 1

            query_detail["gold_docs"].append({
                "doc_id": doc_id,
                "name": doc.get("name", ""),
                "type": gold_type,
                "populated_fields": [f for f, v in populated.items() if v],
            })

        per_query_details.append(query_detail)

    return group_field_counts, group_doc_types, group_query_count, per_query_details


def generate_memory_context(group_field_counts, group_doc_types, group_query_count):
    """Generate memory context with direct boost recommendations.

    Groups by (gold_entity_type, negation_pattern) — uses ground truth
    entity type, not Qwen3 prediction. For each group, recommends
    which fields to boost based on what's actually populated in gold docs.
    """
    lines = []
    lines.append("Retrieval re-routing rules learned from training data:")
    lines.append("For each query pattern, the recommended boost_fields are the fields")
    lines.append("most likely to contain evidence in the correct answer document.")
    lines.append("NOTE: answer_type below refers to the GOLD DOCUMENT's entity type.")
    lines.append("Negation pattern prefers cached Stage 1.5 labels when available, else regex fallback.")
    lines.append("")

    # Sort groups by query count (most common first)
    for group_key in sorted(group_query_count.keys(),
                            key=lambda k: -len(group_query_count[k])):
        count = len(group_query_count[group_key])
        if count < 3:
            continue

        answer_type, neg_pattern = group_key.split("|", 1)

        # Find top populated fields in gold docs
        field_counts = group_field_counts.get(group_key, {})
        total_docs = sum(field_counts.values()) if field_counts else 0
        if total_docs == 0:
            continue

        # Top fields = recommended boost
        top_fields = sorted(field_counts.items(), key=lambda x: -x[1])[:5]
        top_field_names = [f for f, _ in top_fields]

        # Format
        lines.append(f"When answer_type={answer_type} and query contains "
                     f"\"{neg_pattern.replace('_', ' ')}\" ({count} training examples):")
        lines.append(f"  Recommended boost_fields: {top_field_names}")
        field_pcts = ", ".join(f"{f} ({100*c//total_docs}%)" for f, c in top_fields)
        lines.append(f"  Gold doc field distribution: {field_pcts}")

        # Entity types
        entity_types = group_doc_types.get(group_key, {})
        if entity_types:
            et_total = sum(entity_types.values())
            et_str = ", ".join(f"{t} ({100*c//et_total}%)"
                              for t, c in entity_types.most_common(3))
            lines.append(f"  Gold doc entity types: {et_str}")
        lines.append("")

    lines.append("Important: Not all entity types have all fields.")
    lines.append("- drug entities have: indication, contraindication, side effect, "
                 "target, carrier, enzyme, transporter, synergistic interaction")
    lines.append("- disease entities have: associated with, phenotype present/absent, "
                 "parent-child, indication, contraindication (as drugs that treat/avoid them)")
    lines.append("- gene/protein entities have: ppi, interacts with, expression present/absent, "
                 "associated with, target")
    lines.append("- anatomy entities have: expression present/absent, parent-child")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build field confusion matrix grouped by (answer_type, negation_pattern)")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Splits to process (default: train)")
    args = parser.parse_args()

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("Loading corpus...")
    corpus_raw = load_corpus_raw(DATA_DIR)

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"  Building field confusion for split: {split}")
        print(f"{'='*60}")

        rerouted_path = os.path.join(ANALYSIS_DIR, f"rerouted_{split}.json")
        if not os.path.exists(rerouted_path):
            print(f"  ERROR: {rerouted_path} not found.")
            continue

        with open(rerouted_path) as f:
            data = json.load(f)
        rerouted_queries = data["rerouted_queries"]
        print(f"  Loaded {len(rerouted_queries)} rerouted queries")

        group_field_counts, group_doc_types, group_query_count, details = \
            build_confusion(rerouted_queries, corpus_raw)

        # Save detailed report
        report = {
            "split": split,
            "rerouted_count": len(rerouted_queries),
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

        out_path = os.path.join(ANALYSIS_DIR, f"field_confusion_{split}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved to {out_path}")

        # Print summary
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

        # Generate memory context
        memory_ctx = generate_memory_context(
            group_field_counts, group_doc_types, group_query_count)
        ctx_path = os.path.join(ANALYSIS_DIR, f"memory_context_{split}.txt")
        with open(ctx_path, "w") as f:
            f.write(memory_ctx)
        print(f"\n  Memory context saved to {ctx_path}")
        print(f"\n  Preview:")
        for line in memory_ctx.split("\n")[:20]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
