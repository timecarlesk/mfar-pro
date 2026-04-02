"""
Field Populate Check — Verify gold docs have content in target fields
=====================================================================
Before testing weight override, check if gold docs actually populate
the field we want to route to.

Run from project root:
  python failure_analysis/negation/field_populate_check.py [val|test]
"""

import json
import re
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_queries, load_qrels, RELATION_FIELDS

sys.path.insert(0, os.path.dirname(__file__))
from negation_ablation import NEGATION_PATTERN, classify_negation, SPLIT_QRES

DATA_DIR = "data/prime"
OUT_DIR = "output/failure_analysis/negation"
os.makedirs(OUT_DIR, exist_ok=True)

# Map negation subtype → intended corpus field name
SUBTYPE_TO_CORPUS_FIELD = {
    "indication": "indication",
    "contraindication": "contraindication",
    "expression/phenotype_absent": "expression absent",
    "associated_with": "associated with",
    "target": "target",
    "side_effect": "side effect",
    "ppi": "ppi",
    "expression_present": "expression present",
}


def load_corpus_raw():
    """Load full corpus JSON (not the trimmed version)."""
    corpus = {}
    with open(f"{DATA_DIR}/corpus") as f:
        for line in f:
            idx, json_str = line.strip().split("\t", 1)
            corpus[idx] = json.loads(json_str)
    print(f"  Loaded {len(corpus):,} corpus documents (raw JSON)")
    return corpus


def check_field_populated(doc, field_name):
    """Check if a field is non-empty in the doc JSON."""
    val = doc.get(field_name)
    if val is None:
        return False, None
    if isinstance(val, str):
        return bool(val.strip()), val[:200] if val.strip() else None
    if isinstance(val, dict):
        return bool(val), str(val)[:200] if val else None
    if isinstance(val, list):
        return bool(val), str(val)[:200] if val else None
    return bool(val), str(val)[:200]


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    print(f"Field Populate Check — PRIME {split}")
    print("=" * 70)

    corpus = load_corpus_raw()
    queries = load_queries(DATA_DIR, split)
    qrels = load_qrels(DATA_DIR, split)

    # Classify all negation queries
    type_b_queries = []
    for qid, text in queries.items():
        if qid not in qrels:
            continue
        if not NEGATION_PATTERN.search(text):
            continue
        neg_type, neg_subtype, neg_fields = classify_negation(text)
        if neg_type != "B":
            continue
        type_b_queries.append({
            "qid": qid,
            "query": text,
            "neg_subtype": neg_subtype,
            "neg_fields": neg_fields,
            "gold_ids": qrels[qid],
        })

    print(f"\n  Total Type B queries: {len(type_b_queries)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 1: Contraindication deep dive (57 queries)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 1: Contraindication Queries — Field Population")
    print(f"{'='*70}")

    contra_queries = [q for q in type_b_queries if q["neg_subtype"] == "contraindication"]
    print(f"\n  Contraindication queries: {len(contra_queries)}")

    contra_has_field = 0
    contra_has_indication = 0
    contra_has_neither = 0

    for q in contra_queries:
        for gid in q["gold_ids"]:
            if gid not in corpus:
                continue
            doc = corpus[gid]

            has_contra, contra_val = check_field_populated(doc, "contraindication")
            has_indic, indic_val = check_field_populated(doc, "indication")

            if has_contra:
                contra_has_field += 1
            if has_indic:
                contra_has_indication += 1
            if not has_contra and not has_indic:
                contra_has_neither += 1

            print(f"\n  [{q['qid']}] gold={gid} type={doc.get('type','?')} name={doc.get('name','?')[:60]}")
            print(f"    contraindication: {'YES' if has_contra else 'NO'}"
                  f"{' → ' + contra_val if contra_val else ''}")
            print(f"    indication:       {'YES' if has_indic else 'NO'}"
                  f"{' → ' + indic_val if indic_val else ''}")
            print(f"    query: {q['query'][:150]}")
            break  # only check first gold doc

    n = len(contra_queries)
    print(f"\n  --- Contraindication Summary ---")
    print(f"  Gold docs with contraindication field: {contra_has_field}/{n} "
          f"({100*contra_has_field/max(n,1):.1f}%)")
    print(f"  Gold docs with indication field:       {contra_has_indication}/{n} "
          f"({100*contra_has_indication/max(n,1):.1f}%)")
    print(f"  Gold docs with neither:                {contra_has_neither}/{n} "
          f"({100*contra_has_neither/max(n,1):.1f}%)")

    if contra_has_field / max(n, 1) < 0.5:
        print(f"\n  ** WARNING: <50% of gold docs have contraindication field.")
        print(f"     Weight swap to contraindication has limited ceiling. **")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 2: Indication queries — field population
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 2: Indication Queries — Field Population")
    print(f"{'='*70}")

    indic_queries = [q for q in type_b_queries if q["neg_subtype"] == "indication"]
    print(f"\n  Indication queries: {len(indic_queries)}")

    indic_has_field = 0
    indic_has_contra = 0

    for q in indic_queries[:10]:  # print first 10 examples
        for gid in q["gold_ids"]:
            if gid not in corpus:
                continue
            doc = corpus[gid]
            has_indic, indic_val = check_field_populated(doc, "indication")
            has_contra, contra_val = check_field_populated(doc, "contraindication")

            print(f"\n  [{q['qid']}] gold={gid} type={doc.get('type','?')} name={doc.get('name','?')[:60]}")
            print(f"    indication:       {'YES' if has_indic else 'NO'}"
                  f"{' → ' + indic_val if indic_val else ''}")
            print(f"    contraindication: {'YES' if has_contra else 'NO'}")
            print(f"    query: {q['query'][:150]}")
            break

    # Count all (not just printed)
    for q in indic_queries:
        for gid in q["gold_ids"]:
            if gid not in corpus:
                continue
            doc = corpus[gid]
            has_indic, _ = check_field_populated(doc, "indication")
            has_contra, _ = check_field_populated(doc, "contraindication")
            if has_indic:
                indic_has_field += 1
            if has_contra:
                indic_has_contra += 1
            break

    n_i = len(indic_queries)
    print(f"\n  --- Indication Summary ---")
    print(f"  Gold docs with indication field:       {indic_has_field}/{n_i} "
          f"({100*indic_has_field/max(n_i,1):.1f}%)")
    print(f"  Gold docs with contraindication field:  {indic_has_contra}/{n_i} "
          f"({100*indic_has_contra/max(n_i,1):.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 3: ALL Type B — intended field population
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 3: All Type B — Intended Field Population")
    print(f"{'='*70}")

    subtype_stats = defaultdict(lambda: {"total": 0, "has_field": 0})

    for q in type_b_queries:
        intended_field = SUBTYPE_TO_CORPUS_FIELD.get(q["neg_subtype"])
        if intended_field is None:
            subtype_stats[q["neg_subtype"]]["total"] += 1
            continue

        for gid in q["gold_ids"]:
            if gid not in corpus:
                continue
            doc = corpus[gid]
            has_field, _ = check_field_populated(doc, intended_field)
            subtype_stats[q["neg_subtype"]]["total"] += 1
            if has_field:
                subtype_stats[q["neg_subtype"]]["has_field"] += 1
            break

    total_b = sum(s["total"] for s in subtype_stats.values())
    total_has = sum(s["has_field"] for s in subtype_stats.values())

    print(f"\n  {'Subtype':<30} {'Total':>6} {'Has Field':>10} {'%':>7}")
    print("  " + "-" * 56)
    for sub in sorted(subtype_stats):
        s = subtype_stats[sub]
        pct = 100 * s["has_field"] / max(s["total"], 1)
        print(f"  {sub:<30} {s['total']:>6} {s['has_field']:>10} {pct:>6.1f}%")
    print("  " + "-" * 56)
    overall_pct = 100 * total_has / max(total_b, 1)
    print(f"  {'TOTAL':<30} {total_b:>6} {total_has:>10} {overall_pct:>6.1f}%")

    # Gate
    print(f"\n  --- GATE ---")
    if overall_pct >= 50:
        print(f"  PASS: {overall_pct:.1f}% of gold docs have content in intended field.")
        print(f"  Weight override has meaningful ceiling. Proceed with experiment.")
    elif overall_pct >= 20:
        print(f"  MARGINAL: {overall_pct:.1f}% — weight override has limited ceiling.")
        print(f"  Proceed cautiously. The problem may be field coverage, not routing.")
    else:
        print(f"  FAIL: {overall_pct:.1f}% — intended field is mostly empty in gold docs.")
        print(f"  Weight swap won't help. Problem is KG coverage, not mFAR routing.")

    # Save report
    report = f"""# Field Populate Check — PRIME {split}

## Contraindication (n={len(contra_queries)})

| Metric | Count | % |
|--------|-------|---|
| Gold has contraindication field | {contra_has_field} | {100*contra_has_field/max(len(contra_queries),1):.1f}% |
| Gold has indication field | {contra_has_indication} | {100*contra_has_indication/max(len(contra_queries),1):.1f}% |
| Gold has neither | {contra_has_neither} | {100*contra_has_neither/max(len(contra_queries),1):.1f}% |

## Indication (n={len(indic_queries)})

| Metric | Count | % |
|--------|-------|---|
| Gold has indication field | {indic_has_field} | {100*indic_has_field/max(len(indic_queries),1):.1f}% |
| Gold has contraindication field | {indic_has_contra} | {100*indic_has_contra/max(len(indic_queries),1):.1f}% |

## All Type B — Intended Field Population

| Subtype | Total | Has Field | % |
|---------|-------|-----------|---|
"""
    for sub in sorted(subtype_stats):
        s = subtype_stats[sub]
        pct = 100 * s["has_field"] / max(s["total"], 1)
        report += f"| {sub} | {s['total']} | {s['has_field']} | {pct:.1f}% |\n"
    report += f"| **TOTAL** | **{total_b}** | **{total_has}** | **{overall_pct:.1f}%** |\n"

    report += f"\n## Gate: {'PASS' if overall_pct >= 50 else 'MARGINAL' if overall_pct >= 20 else 'FAIL'} ({overall_pct:.1f}%)\n"

    path = f"{OUT_DIR}/FIELD_POPULATE_{split}.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
