"""
Multi-Hop (Cross-Type) Failure Analysis — PRIME Validation Set
===============================================================
Implements the slide's classification logic:
  1. Extract relation keywords from query text via regex
  2. Map each relation to entity types that carry that field
  3. If gold doc type ∉ expected types ⇒ cross-type (multi-hop)

Run from project root (multifield-adaptive-retrieval/):
  python failure_analysis/multi-hop/multi_hop_analysis.py
"""

import json
import re
import os
import sys
import math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_queries, load_qrels, load_retrieved

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data/prime"
EVAL_DIR = "output/prime_eval"
OUT_DIR = "output/failure_analysis/multi-hop"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Relation field regex hints (from failure_analysis.py) ─────────────────────
# Maps relation field name → regex that detects when a query "hints" at that field
FIELD_HINTS = {
    "ppi":                    r"\binteract(?:s|ing|ion)?\b|protein.protein|ppi\b",
    "expression absent":      r"(?:not|lack|absent|no)\s+express|unexpress",
    "expression present":     r"express(?:ed|ion)?\s+in",
    "indication":             r"\bindication|treat(?:s|ed|ing|ment)?\b|therap|clinical\s+trial|phase\s+[IVX]|approv|prescri",
    "target":                 r"\btarget(?:s|ed|ing)?\b",
    "side effect":            r"side[\s-]?effect|adverse|toxici",
    "parent-child":           r"hierarch|above.*below|below.*above|connect.*pathway|parent|child|subtype|sub[\s-]?type|subcategor|belong|classif",
    "phenotype present":      r"\bphenotype\b|observed\s+effect|symptom",
    "phenotype absent":       r"without\s+phenotype|lack.*phenotype",
    "enzyme":                 r"\benzyme\b|metaboli|catalyz|breakdown",
    "carrier":                r"\bcarrier\b",
    "transporter":            r"\btransport(?:s|er|ed|ing)?\b",
    "contraindication":       r"\bcontraindic",
    "associated with":        r"\bassociat(?:ed|ion|es)?\b",
    "linked to":              r"\blinked?\s+(?:to|with)\b",
    "synergistic interaction": r"\bsynergist",
    "off-label use":          r"off[\s-]label",
    "interacts with":         r"\bpathway\b.*\b(?:interact|connect|signal)",
}

# ── Field → Entity types that carry that field (from corpus statistics) ───────
FIELD_TO_ENTITY_TYPES = {
    "ppi":                     {"gene/protein"},
    "carrier":                 {"gene/protein", "drug"},
    "enzyme":                  {"gene/protein", "drug"},
    "target":                  {"gene/protein", "drug"},
    "transporter":             {"gene/protein", "drug"},
    "contraindication":        {"drug", "disease"},
    "indication":              {"drug", "disease"},
    "off-label use":           {"drug", "disease"},
    "synergistic interaction": {"drug"},
    "associated with":         {"gene/protein", "disease", "effect/phenotype"},
    "parent-child":            {"disease", "effect/phenotype", "anatomy", "exposure",
                                "biological_process", "cellular_component",
                                "molecular_function", "pathway"},
    "phenotype absent":        {"disease", "effect/phenotype"},
    "phenotype present":       {"disease", "effect/phenotype"},
    "side effect":             {"drug", "effect/phenotype"},
    "interacts with":          {"gene/protein", "exposure", "biological_process",
                                "cellular_component", "molecular_function", "pathway"},
    "linked to":               {"disease", "exposure"},
    "expression present":      {"gene/protein", "anatomy"},
    "expression absent":       {"gene/protein", "anatomy"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus():
    """Local corpus loader — only needs name and type."""
    corpus = {}
    with open(f"{DATA_DIR}/corpus") as f:
        for line in f:
            idx, json_str = line.strip().split("\t", 1)
            doc = json.loads(json_str)
            corpus[idx] = {
                "name": doc.get("name", ""),
                "type": doc.get("type", ""),
            }
    print(f"  Loaded {len(corpus):,} corpus documents")
    return corpus


def _load_queries(split="val"):
    return load_queries(DATA_DIR, split)


def _load_qrels(split="val"):
    return load_qrels(DATA_DIR, split)


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-Type Classification (Slide Logic)
# ══════════════════════════════════════════════════════════════════════════════

def get_hinted_fields(query_text):
    """Extract which relation fields the query hints at, using FIELD_HINTS regex."""
    hinted = set()
    query_lower = query_text.lower()
    for field, pattern in FIELD_HINTS.items():
        if re.search(pattern, query_lower):
            hinted.add(field)
    return hinted


def get_expected_entity_types(hinted_fields):
    """Map hinted fields → union of entity types that carry those fields."""
    expected = set()
    for field in hinted_fields:
        if field in FIELD_TO_ENTITY_TYPES:
            expected |= FIELD_TO_ENTITY_TYPES[field]
    return expected


def classify_cross_type(query_text, gold_types):
    """
    Classify a query as same-type, cross-type, or unclassified.

    Returns: ("same_type" | "cross_type" | "unclassified", hinted_fields, expected_types)
    """
    hinted = get_hinted_fields(query_text)

    if not hinted:
        return "unclassified", hinted, set()

    expected = get_expected_entity_types(hinted)

    if not expected:
        return "unclassified", hinted, expected

    # Check if ANY gold doc type is in the expected types
    # If all gold doc types are outside expected → cross-type
    gold_in_expected = any(t in expected for t in gold_types)

    if gold_in_expected:
        return "same_type", hinted, expected
    else:
        return "cross_type", hinted, expected


# ══════════════════════════════════════════════════════════════════════════════
#  Per-Query Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_per_query(queries, qrels, retrieved, corpus):
    rows = []
    for qid, query_text in queries.items():
        if qid not in qrels:
            continue
        gold = qrels[qid]
        docs = retrieved.get(qid, [])
        top100 = docs[:100]
        top_ids = [d[0] for d in top100]

        # Reciprocal rank
        first_rel_rank = next((i for i, did in enumerate(top_ids) if did in gold), -1)
        rr = 0.0 if first_rel_rank < 0 else 1.0 / (first_rel_rank + 1)

        # Hit@1
        hit1 = 1 if first_rel_rank == 0 else 0

        # Complete miss
        miss = 1 if first_rel_rank < 0 else 0

        # Gold doc entity types
        gold_types = [corpus[did]["type"] for did in gold if did in corpus]

        # Cross-type classification
        label, hinted, expected = classify_cross_type(query_text, gold_types)

        rows.append({
            "qid": qid,
            "query": query_text,
            "rr": rr,
            "hit1": hit1,
            "miss": miss,
            "first_rel_rank": first_rel_rank,
            "gold_types": gold_types,
            "label": label,
            "hinted_fields": hinted,
            "expected_types": expected,
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Group Statistics
# ══════════════════════════════════════════════════════════════════════════════

def group_stats(rows, label):
    group = [r for r in rows if r["label"] == label]
    n = len(group)
    if n == 0:
        return {"label": label, "n": 0, "mrr": 0, "hit1_pct": 0, "miss_pct": 0}
    mrr = sum(r["rr"] for r in group) / n
    hit1_pct = 100 * sum(r["hit1"] for r in group) / n
    miss_pct = 100 * sum(r["miss"] for r in group) / n
    return {"label": label, "n": n, "mrr": mrr, "hit1_pct": hit1_pct, "miss_pct": miss_pct}


# ══════════════════════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════════════════════

def print_report(rows):
    total = len(rows)
    same = group_stats(rows, "same_type")
    cross = group_stats(rows, "cross_type")
    unclass = group_stats(rows, "unclassified")

    classified = same["n"] + cross["n"]
    cross_pct = 100 * cross["n"] / classified if classified > 0 else 0

    print("\n" + "=" * 70)
    print("MULTI-HOP (CROSS-TYPE) FAILURE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal val queries:     {total}")
    print(f"  Same-type:           {same['n']} ({100*same['n']/total:.1f}%)")
    print(f"  Cross-type:          {cross['n']} ({100*cross['n']/total:.1f}%)")
    print(f"  Unclassified:        {unclass['n']} ({100*unclass['n']/total:.1f}%)")
    print(f"\nCross-type fraction (of classified): {cross_pct:.1f}%")

    print(f"\n{'':18} {'Same-type':>12} {'Cross-type':>12} {'Delta':>10}")
    print("-" * 55)
    print(f"  {'n':<16} {same['n']:>12} {cross['n']:>12}")

    mrr_delta = (cross["mrr"] - same["mrr"]) / same["mrr"] * 100 if same["mrr"] > 0 else 0
    print(f"  {'MRR':<16} {same['mrr']:>12.3f} {cross['mrr']:>12.3f} {mrr_delta:>+9.1f}%")
    print(f"  {'Hit@1':<16} {same['hit1_pct']:>11.1f}% {cross['hit1_pct']:>11.1f}%")

    miss_ratio = cross["miss_pct"] / same["miss_pct"] if same["miss_pct"] > 0 else 0
    print(f"  {'Miss Rate':<16} {same['miss_pct']:>11.1f}% {cross['miss_pct']:>11.1f}% {miss_ratio:>8.1f}x")

    return same, cross, unclass


def save_report(rows, same, cross, unclass):
    classified = same["n"] + cross["n"]
    cross_pct = 100 * cross["n"] / classified if classified > 0 else 0
    mrr_delta = (cross["mrr"] - same["mrr"]) / same["mrr"] * 100 if same["mrr"] > 0 else 0
    miss_ratio = cross["miss_pct"] / same["miss_pct"] if same["miss_pct"] > 0 else 0

    report = f"""# Multi-Hop (Cross-Type) Failure Analysis Report
## PRIME Validation Set

**Classification method** (from slide):
1. Extract relation keywords from query text (FIELD_HINTS regex)
2. Map each relation to entity types that carry that field (FIELD_TO_ENTITY_TYPES)
3. If gold doc type not in expected types → cross-type (multi-hop)

---

## Summary

| | Same-type (single-hop) | Cross-type (multi-hop) |
|---|---|---|
| n | {same['n']} | {cross['n']} |
| MRR | {same['mrr']:.3f} | {cross['mrr']:.3f} ({mrr_delta:+.1f}%) |
| Hit@1 | {same['hit1_pct']:.1f}% | {cross['hit1_pct']:.1f}% |
| Miss Rate | {same['miss_pct']:.1f}% | {cross['miss_pct']:.1f}% ({miss_ratio:.1f}x) |

Cross-type fraction (of classified queries): **{cross_pct:.1f}%**
Unclassified (no field hint matched): {unclass['n']}

---

## Cross-Type Examples (worst failures)

"""
    # Add worst cross-type examples
    cross_rows = sorted(
        [r for r in rows if r["label"] == "cross_type"],
        key=lambda r: r["rr"]
    )
    for ex in cross_rows[:10]:
        report += f"- **[{ex['qid']}]** RR={ex['rr']:.2f}, gold_type={ex['gold_types']}, "
        report += f"expected={ex['expected_types']}, hinted={ex['hinted_fields']}\n"
        report += f"  - {ex['query'][:150]}\n\n"

    report += """---

## Methodology Detail

### FIELD_HINTS (query text → relation field)
"""
    for field, pattern in FIELD_HINTS.items():
        report += f"- `{field}`: `{pattern}`\n"

    report += """
### FIELD_TO_ENTITY_TYPES (relation field → entity types that have it)
"""
    for field, types in sorted(FIELD_TO_ENTITY_TYPES.items()):
        report += f"- `{field}`: {sorted(types)}\n"

    report += f"\n---\n\n*Analysis: {len(rows)} val queries, classification by cross-type method*\n"

    with open(f"{OUT_DIR}/REPORT.md", "w") as f:
        f.write(report)
    print(f"\n  Saved: {OUT_DIR}/REPORT.md")


# ══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(same, cross):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Single-Hop vs Multi-Hop (Cross-Type) Performance", fontsize=13, fontweight="bold")

    labels = ["Same-type\n(single-hop)", "Cross-type\n(multi-hop)"]
    colors = ["#2ecc71", "#e74c3c"]

    # MRR
    ax = axes[0]
    vals = [same["mrr"], cross["mrr"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    ax.set_ylabel("MRR")
    ax.set_ylim(0, max(vals) * 1.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Hit@1
    ax = axes[1]
    vals = [same["hit1_pct"], cross["hit1_pct"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    ax.set_ylabel("Hit@1 (%)")
    ax.set_ylim(0, max(vals) * 1.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Miss Rate
    ax = axes[2]
    vals = [same["miss_pct"], cross["miss_pct"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    ax.set_ylabel("Miss Rate (%)")
    ax.set_ylim(0, max(vals) * 1.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/cross_type_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/cross_type_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Detailed Breakdown
# ══════════════════════════════════════════════════════════════════════════════

def print_detailed_breakdown(rows):
    """Show cross-type stats broken down by hinted field and gold entity type."""
    cross_rows = [r for r in rows if r["label"] == "cross_type"]
    if not cross_rows:
        return

    print("\n" + "-" * 70)
    print("CROSS-TYPE BREAKDOWN BY HINTED FIELD")
    print("-" * 70)
    field_groups = defaultdict(list)
    for r in cross_rows:
        for f in r["hinted_fields"]:
            field_groups[f].append(r)

    print(f"  {'Hinted Field':<25} {'n':>5} {'MRR':>8} {'Miss%':>8}")
    print("  " + "-" * 50)
    for field, frows in sorted(field_groups.items(), key=lambda x: -len(x[1])):
        n = len(frows)
        mrr = sum(r["rr"] for r in frows) / n
        miss = 100 * sum(r["miss"] for r in frows) / n
        print(f"  {field:<25} {n:>5} {mrr:>8.3f} {miss:>7.1f}%")

    print("\n" + "-" * 70)
    print("CROSS-TYPE BREAKDOWN BY GOLD DOC ENTITY TYPE")
    print("-" * 70)
    type_groups = defaultdict(list)
    for r in cross_rows:
        for t in set(r["gold_types"]):
            type_groups[t].append(r)

    print(f"  {'Gold Entity Type':<25} {'n':>5} {'MRR':>8} {'Miss%':>8}")
    print("  " + "-" * 50)
    for etype, erows in sorted(type_groups.items(), key=lambda x: -len(x[1])):
        n = len(erows)
        mrr = sum(r["rr"] for r in erows) / n
        miss = 100 * sum(r["miss"] for r in erows) / n
        print(f"  {etype:<25} {n:>5} {mrr:>8.3f} {miss:>7.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["val"]
    # Map each split to its retrieval result file
    SPLIT_QRES = {
        "val":  f"{EVAL_DIR}/final-all-0.qres",
        "test": f"{EVAL_DIR}/final-additional-all-0.qres",
    }

    print(f"Loading data for splits: {splits}")
    corpus = load_corpus()

    # Merge queries, qrels, retrieved across splits
    queries = {}
    qrels_all = defaultdict(set)
    retrieved_all = defaultdict(list)

    for split in splits:
        q = _load_queries(split)
        queries.update(q)
        qr = _load_qrels(split)
        for qid, docs in qr.items():
            qrels_all[qid] |= docs
        ret = load_retrieved(SPLIT_QRES[split])
        for qid, docs in ret.items():
            if qid not in retrieved_all:
                retrieved_all[qid] = docs

    print(f"\n  Total merged: {len(queries)} queries")

    print("\nComputing per-query metrics and cross-type classification...")
    rows = compute_per_query(queries, qrels_all, retrieved_all, corpus)

    same, cross, unclass = print_report(rows)
    print_detailed_breakdown(rows)
    save_report(rows, same, cross, unclass)
    plot_comparison(same, cross)

    print("\nDone.")


if __name__ == "__main__":
    main()
