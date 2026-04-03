"""
Build Field Confusion Matrix for Training Type B Queries.

For each Type B query's gold docs, checks which relation fields are populated.
This reveals whether field re-routing can help (e.g., B.contraindication gold
docs DO have the contraindication field) or not (e.g., B.indication gold docs
have 0% indication field).

Run from project root:
  python failure_analysis/type_b_memory/build_field_confusion.py
  python failure_analysis/type_b_memory/build_field_confusion.py --split val test
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from failure_analysis.utils import RELATION_FIELDS, load_corpus

DATA_DIR = "data/prime"
ANALYSIS_DIR = "output/failure_analysis/type_b_memory/analysis"


def load_corpus_raw(data_dir):
    """Load corpus with full JSON (not just metadata) for field inspection."""
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


def build_confusion(rerouted_queries, corpus_raw):
    """Build field confusion matrix: per suppress group, which fields are populated in gold docs."""
    confusion = defaultdict(lambda: defaultdict(int))
    doc_types = defaultdict(lambda: Counter())
    per_query_details = []

    for entry in rerouted_queries:
        # Group by suppress fields (the "misleading" fields)
        suppress = entry.get("suppress_fields") or []
        group_key = ",".join(sorted(suppress)) if suppress else "no_suppress"
        gold_ids = entry.get("gold_ids", [])

        query_detail = {
            "qid": entry["qid"],
            "query": entry["query"],
            "boost_fields": entry.get("boost_fields", []),
            "suppress_fields": suppress,
            "gold_docs": [],
        }

        for doc_id in gold_ids:
            if doc_id not in corpus_raw:
                continue
            doc = corpus_raw[doc_id]

            # Check all relation fields
            populated = {}
            for field in RELATION_FIELDS:
                populated[field] = check_field_populated(doc, field)
                if populated[field]:
                    confusion[group_key][field] += 1

            # Track entity types
            doc_type = doc.get("type", "unknown")
            doc_types[group_key][doc_type] += 1

            query_detail["gold_docs"].append({
                "doc_id": doc_id,
                "name": doc.get("name", ""),
                "type": doc_type,
                "populated_fields": [f for f, v in populated.items() if v],
            })

        per_query_details.append(query_detail)

    return confusion, doc_types, per_query_details


def generate_memory_context(rerouted_queries, confusion, doc_types):
    """Generate memory_context.txt from training data statistics.

    This text gets injected into the Stage 2 Qwen3 prompt so the LLM
    has data-driven failure patterns instead of hardcoded guesses.
    """
    lines = []
    lines.append("Known failure patterns (from training data analysis):")
    lines.append("")

    # Group rerouted queries by suppress fields to find patterns
    from collections import Counter, defaultdict
    suppress_groups = defaultdict(list)
    for entry in rerouted_queries:
        suppress = entry.get("suppress_fields") or []
        key = ",".join(sorted(suppress)) if suppress else "none"
        suppress_groups[key].append(entry)

    for group_key in sorted(suppress_groups.keys(), key=lambda k: -len(suppress_groups[k])):
        entries = suppress_groups[group_key]
        if len(entries) < 3:  # skip very rare patterns
            continue

        # Answer type distribution
        at_counts = Counter(e.get("answer_type", "unknown") for e in entries)
        total = len(entries)

        # Gold doc field distribution from confusion matrix
        gold_fields = confusion.get(group_key, {})
        gold_total = sum(gold_fields.values()) if gold_fields else 0

        lines.append(f"- Suppress [{group_key}] queries ({total} examples):")
        lines.append(f"  Answer types: {', '.join(f'{t} ({100*c//total}%)' for t, c in at_counts.most_common(3))}")
        if gold_total:
            top_gold = sorted(gold_fields.items(), key=lambda x: -x[1])[:5]
            lines.append(f"  Gold doc fields actually populated: {', '.join(f'{f} ({100*c//gold_total}%)' for f, c in top_gold)}")

        # Gold doc entity types
        entity_types = doc_types.get(group_key, {})
        if entity_types:
            et_total = sum(entity_types.values())
            lines.append(f"  Gold doc entity types: {', '.join(f'{t} ({100*c//et_total}%)' for t, c in entity_types.most_common(3))}")
        lines.append("")

    lines.append("Important: Not all entity types have all fields. For example:")
    lines.append("- Diseases do NOT have indication/contraindication/side effect/target fields")
    lines.append("- Genes/proteins do NOT have indication/contraindication fields")
    lines.append("- Only drugs have indication, contraindication, side effect, target, carrier, enzyme, transporter")
    lines.append("Consider what fields the ANSWER entity type actually has before suggesting boost/suppress.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build field confusion matrix for Type B queries")
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

        # Load rerouted query extraction
        rerouted_path = os.path.join(ANALYSIS_DIR, f"rerouted_{split}.json")
        if not os.path.exists(rerouted_path):
            print(f"  ERROR: {rerouted_path} not found. Run failure_analysis/type_b_memory/extract_train_type_b.py first.")
            continue

        with open(rerouted_path) as f:
            data = json.load(f)
        rerouted_queries = data["rerouted_queries"]
        print(f"  Loaded {len(rerouted_queries)} rerouted queries")

        confusion, doc_types, details = build_confusion(rerouted_queries, corpus_raw)

        # Build report
        report = {
            "split": split,
            "rerouted_count": len(rerouted_queries),
            "field_confusion": {
                pair: dict(sorted(fields.items(), key=lambda x: -x[1]))
                for pair, fields in sorted(confusion.items())
            },
            "gold_doc_entity_types": {
                pair: dict(types.most_common())
                for pair, types in sorted(doc_types.items())
            },
            "per_query_details": details,
        }

        out_path = os.path.join(ANALYSIS_DIR, f"field_confusion_{split}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved to {out_path}")

        # Print summary
        print(f"\n  Field Confusion Matrix:")
        print(f"  {'Suppress Group':<35} {'Gold Field':<25} {'Count':<8}")
        print(f"  {'-'*68}")
        for pair in sorted(confusion.keys()):
            fields = confusion[pair]
            total_docs = sum(fields.values())
            for field, count in sorted(fields.items(), key=lambda x: -x[1])[:5]:
                pct = 100 * count / total_docs if total_docs else 0
                print(f"  {pair:<35} {field:<25} {count:<8} ({pct:.1f}%)")
            print()

        print(f"\n  Gold Doc Entity Types:")
        for pair in sorted(doc_types.keys()):
            types = doc_types[pair]
            print(f"    {pair}:")
            for dtype, count in types.most_common():
                print(f"      {dtype}: {count}")

        # Generate memory context for Stage 2 prompt
        memory_ctx = generate_memory_context(rerouted_queries, confusion, doc_types)
        ctx_path = os.path.join(ANALYSIS_DIR, f"memory_context_{split}.txt")
        with open(ctx_path, "w") as f:
            f.write(memory_ctx)
        print(f"\n  Memory context saved to {ctx_path}")
        print(f"\n  Preview:")
        for line in memory_ctx.split("\n")[:15]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
