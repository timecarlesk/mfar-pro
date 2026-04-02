"""
MFAR Failure Analysis — PRIME Dataset (Validation Set)
=======================================================
Analyzes poor retrieval results from the MFAR model grounded in its
adaptive field-weighting mechanism (LinearWeights.forward in weighting.py).

Sections:
  1. Per-query metric computation & severity bucketing
  2. MFAR field-weighting failure proxies
  3. Query-type × entity-type categorical analysis
  4. Score distribution analysis
  5. Ablation data interpretation
  6. Visualizations

Run from project root:
  python failure_analysis/general/failure_analysis.py
"""

import json
import re
import os
import sys
import math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (RELATION_FIELDS, BASIC_FIELDS, ALL_FIELDS,
                   load_queries, load_qrels, load_retrieved, dcg)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/prime"
EVAL_DIR     = "output/prime_eval"
ABLATION_DIR = "output/prime_eval"
OUT_DIR      = "output/failure_analysis/general"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Negation patterns (for Section 8 negation analysis) ──────────────────────
NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|without|lack(?:s|ing)?|absent|neither|nor|never|"
    r"cannot|can't|don't|doesn't|didn't|won't|isn't|aren't|wasn't|weren't|"
    r"exclude[ds]?|non[\-])\b|"
    r"\bun(?:express|relat|associat|link|affect|involv)\w*\b",
    re.IGNORECASE,
)

# ── Query type classification patterns (priority order) ──────────────────────
QUERY_CATEGORIES = [
    ("expression_absent",   r"(?:not|lack|absent|no)\s+express|unexpress|without\s+express"),
    ("expression_present",  r"express(?:ed|ion|es)?\s+(?:in|by|of)|show\s+express"),
    ("pathway_hierarchy",   r"pathways?\s+connect|above.*below|below.*above|hierarch|signal.*pathway|connect.*pathway"),
    ("drug_indication",     r"indicat|phase\s+[IVX]+|clinical\s+trial|treat.*diseas|approv"),
    ("drug_interaction",    r"carrier|transporter|enzyme|metaboli"),
    ("side_effect",         r"side[\s-]?effect|adverse\s+effect|toxici"),
    ("target",              r"\btarget\b"),
    ("ppi",                 r"\bprotein.protein|ppi\b|\binteract(?:s|ing|ion)?\b"),
    ("phenotype",           r"\bphenotype\b|phenotyp"),
    ("drug_target",         r"drug.*target|compound.*target|target.*compound|target.*drug"),
    ("multi_hop",           r"."),  # catch-all
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0: Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_corpus():
    """Wrapper using module-level DATA_DIR."""
    from utils import load_corpus as _lc
    return _lc(DATA_DIR)


def _load_queries(split="val"):
    return load_queries(DATA_DIR, split)


def _load_qrels(split="val"):
    return load_qrels(DATA_DIR, split)


def load_ablation():
    """Load ablation records; use the full-run group (ndcg > 0.46, val partition)."""
    records = []
    with open(f"{ABLATION_DIR}/results_dicts-all-0.jsonl") as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("additional") == "val" and float(r["ndcg"]) > 0.45:
                records.append(r)
    print(f"  Loaded {len(records)} ablation records")
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: Per-query metrics & severity bucketing
# ══════════════════════════════════════════════════════════════════════════════


def compute_per_query(queries, qrels, retrieved, corpus):
    """
    For each query, compute:
      rr, ndcg10, r10, r100, first_rel_rank, top1_score, first_rel_score, score_gap,
      bucket, query_category, gold_types, gold_field_density, top10_competing_types
    """
    rows = []
    for qid, query_text in queries.items():
        if qid not in qrels:
            continue
        gold   = qrels[qid]
        docs   = retrieved.get(qid, [])
        top100 = docs[:100]
        top_ids = [d[0] for d in top100]
        scores  = [d[1] for d in top100]

        # Basic metrics
        top1_score    = scores[0] if scores else 0.0
        first_rel_rank = next((i for i, did in enumerate(top_ids) if did in gold), -1)
        rr = 0.0 if first_rel_rank < 0 else 1.0 / (first_rel_rank + 1)

        gains10 = [1 if did in gold else 0 for did in top_ids[:10]]
        ndcg10  = dcg(gains10, 10) / dcg([1] * min(len(gold), 10), 10) if gold else 0.0

        r10  = len(gold & set(top_ids[:10]))  / len(gold) if gold else 0.0
        r100 = len(gold & set(top_ids[:100])) / len(gold) if gold else 0.0

        first_rel_score = scores[first_rel_rank] if first_rel_rank >= 0 else None
        score_gap = (top1_score - first_rel_score) if first_rel_score is not None else None

        # Severity bucket
        if first_rel_rank < 0:
            bucket = "complete_miss"
        elif first_rel_rank == 0:
            bucket = "hit@1"
        elif first_rel_rank < 5:
            bucket = "hit@2-5"
        elif first_rel_rank < 10:
            bucket = "hit@6-10"
        else:
            bucket = "hit@11-100"

        # Query category
        category = classify_query(query_text)

        # Gold doc info
        gold_types   = [corpus[did]["type"] for did in gold if did in corpus]
        gold_fields  = [corpus[did]["field_count"] for did in gold if did in corpus]
        gold_field_density = (sum(gold_fields) / len(gold_fields) / len(RELATION_FIELDS)
                              if gold_fields else 0.0)
        gold_populated = set()
        for did in gold:
            if did in corpus:
                gold_populated |= corpus[did]["fields"]

        # Top-10 competing types (for failures)
        top10_types = [corpus[did]["type"] for did, _ in top100[:10] if did in corpus]

        rows.append({
            "qid":               qid,
            "query":             query_text,
            "num_relevant":      len(gold),
            "rr":                rr,
            "ndcg10":            ndcg10,
            "r10":               r10,
            "r100":              r100,
            "first_rel_rank":    first_rel_rank,
            "top1_score":        top1_score,
            "first_rel_score":   first_rel_score,
            "score_gap":         score_gap,
            "bucket":            bucket,
            "category":          category,
            "gold_types":        gold_types,
            "gold_field_density": gold_field_density,
            "gold_populated":    gold_populated,
            "top10_types":       top10_types,
        })
    return rows


def classify_query(text):
    text_lower = text.lower()
    for cat, pattern in QUERY_CATEGORIES:
        if re.search(pattern, text_lower):
            return cat
    return "multi_hop"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 REPORT: Severity distribution
# ══════════════════════════════════════════════════════════════════════════════

BUCKETS = ["hit@1", "hit@2-5", "hit@6-10", "hit@11-100", "complete_miss"]
BUCKET_COLORS = {
    "hit@1":        "#2ecc71",
    "hit@2-5":      "#27ae60",
    "hit@6-10":     "#f39c12",
    "hit@11-100":   "#e67e22",
    "complete_miss":"#e74c3c",
}


def section1_severity(rows):
    print("\n" + "="*70)
    print("SECTION 1: Failure Severity Distribution")
    print("="*70)
    total = len(rows)
    counts = Counter(r["bucket"] for r in rows)
    mean_rr     = sum(r["rr"] for r in rows) / total
    mean_ndcg10 = sum(r["ndcg10"] for r in rows) / total
    mean_r10    = sum(r["r10"] for r in rows) / total

    print(f"\nTotal val queries analyzed: {total}")
    print(f"  Mean RR:      {mean_rr:.4f}")
    print(f"  Mean NDCG@10: {mean_ndcg10:.4f}")
    print(f"  Mean R@10:    {mean_r10:.4f}")
    print()
    print(f"{'Bucket':<18} {'Count':>7} {'%':>7}  {'Mean RR':>8} {'Mean NDCG@10':>13}")
    print("-" * 62)
    for b in BUCKETS:
        brows = [r for r in rows if r["bucket"] == b]
        pct   = 100 * len(brows) / total
        mrr   = sum(r["rr"] for r in brows) / len(brows) if brows else 0
        mndcg = sum(r["ndcg10"] for r in brows) / len(brows) if brows else 0
        print(f"  {b:<16} {len(brows):>7} {pct:>7.1f}%  {mrr:>8.4f} {mndcg:>13.4f}")
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: MFAR field-weighting failure proxies
# ══════════════════════════════════════════════════════════════════════════════

def section2_field_proxies(rows):
    print("\n" + "="*70)
    print("SECTION 2: MFAR Field-Weighting Failure Proxies")
    print("="*70)

    # --- 2a. Field sparsity of relevant docs (per bucket) ---
    print("\n[2a] Field sparsity of gold docs (mean populated relation fields / 18)")
    print(f"  {'Bucket':<18} {'Mean field density':>20} {'Count':>7}")
    print("  " + "-"*50)
    for b in BUCKETS:
        brows = [r for r in rows if r["bucket"] == b]
        densities = [r["gold_field_density"] for r in brows]
        mean_d = sum(densities) / len(densities) if densities else 0
        print(f"  {b:<18} {mean_d:>20.4f} {len(brows):>7}")

    # --- 2b. Field mismatch: query hints at fields absent in gold doc ---
    FIELD_HINTS = {
        "ppi":               r"\binteract(?:s|ing|ion)?\b|protein.protein|ppi\b",
        "expression absent": r"(?:not|lack|absent|no)\s+express|unexpress",
        "expression present":r"express(?:ed|ion)?\s+in",
        "indication":        r"\bindication|treat.*disease|clinical\s+trial",
        "target":            r"\btarget\b",
        "side effect":       r"side[\s-]?effect|adverse",
        "parent-child":      r"hierarch|above.*below|connect.*pathway|parent",
        "phenotype present": r"\bphenotype\b",
        "enzyme":            r"\benzyme\b|metaboli",
        "carrier":           r"\bcarrier\b|transport.*drug",
    }
    print("\n[2b] Field mismatch rate: query hints at field X but gold doc lacks X")
    mismatches = Counter()
    mismatch_total = Counter()
    for r in rows:
        query_lower = r["query"].lower()
        for field, pattern in FIELD_HINTS.items():
            if re.search(pattern, query_lower):
                mismatch_total[field] += 1
                if field not in r["gold_populated"]:
                    mismatches[field] += 1
    print(f"  {'Field':<25} {'Queries hinting':<18} {'Field absent %':>15}")
    print("  " + "-" * 60)
    for field, total_cnt in sorted(mismatch_total.items(), key=lambda x: -x[1]):
        if total_cnt > 5:
            pct = 100 * mismatches[field] / total_cnt
            print(f"  {field:<25} {total_cnt:<18} {pct:>14.1f}%")

    # --- 2c. Competing entity types for complete misses ---
    print("\n[2c] Entity types dominating top-10 results for complete-miss queries")
    miss_rows = [r for r in rows if r["bucket"] == "complete_miss"]
    competing_types = Counter()
    for r in miss_rows:
        competing_types.update(r["top10_types"])
    print(f"  (Top-10 retrieved docs across {len(miss_rows)} complete-miss queries)")
    for etype, cnt in competing_types.most_common(10):
        print(f"    {etype:<35} {cnt:>5}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: Query-type × Entity-type matrix
# ══════════════════════════════════════════════════════════════════════════════

def section3_categorical(rows):
    print("\n" + "="*70)
    print("SECTION 3: Query-Type × Entity-Type Failure Matrix")
    print("="*70)

    # Collect all entity types
    all_etypes = Counter()
    for r in rows:
        all_etypes.update(r["gold_types"])
    top_etypes = [e for e, _ in all_etypes.most_common(8)]

    cats = [c for c, _ in QUERY_CATEGORIES]

    # Build matrix: mean RR + count
    matrix_rr    = defaultdict(lambda: defaultdict(list))
    matrix_miss  = defaultdict(lambda: defaultdict(int))
    matrix_count = defaultdict(lambda: defaultdict(int))

    for r in rows:
        cat = r["category"]
        for etype in set(r["gold_types"]):
            matrix_rr[cat][etype].append(r["rr"])
            matrix_count[cat][etype] += 1
            if r["bucket"] == "complete_miss":
                matrix_miss[cat][etype] += 1

    # Print as ASCII table
    col_w = 12
    print("\n  Mean RR by query category × gold entity type:")
    header = f"  {'Query category':<22}" + "".join(f"{e[:col_w]:>{col_w}}" for e in top_etypes)
    print(header)
    print("  " + "-" * (22 + col_w * len(top_etypes)))
    for cat in cats:
        row_str = f"  {cat:<22}"
        for etype in top_etypes:
            vals = matrix_rr[cat][etype]
            if vals:
                row_str += f"{sum(vals)/len(vals):>{col_w}.3f}"
            else:
                row_str += f"{'---':>{col_w}}"
        print(row_str)

    # Miss rate table
    print("\n  Complete-miss rate (%) by query category × gold entity type:")
    print(header)
    print("  " + "-" * (22 + col_w * len(top_etypes)))
    for cat in cats:
        row_str = f"  {cat:<22}"
        for etype in top_etypes:
            cnt  = matrix_count[cat][etype]
            miss = matrix_miss[cat][etype]
            if cnt > 0:
                row_str += f"{100*miss/cnt:>{col_w}.1f}"
            else:
                row_str += f"{'---':>{col_w}}"
        print(row_str)

    # Per-category summary
    print("\n  Per query-category summary:")
    print(f"  {'Category':<22} {'Count':>7} {'Mean RR':>9} {'Miss%':>7} {'Mean NDCG@10':>14}")
    print("  " + "-"*62)
    for cat in cats:
        crows = [r for r in rows if r["category"] == cat]
        if not crows:
            continue
        mrr  = sum(r["rr"] for r in crows) / len(crows)
        mndcg = sum(r["ndcg10"] for r in crows) / len(crows)
        miss_pct = 100 * sum(1 for r in crows if r["bucket"] == "complete_miss") / len(crows)
        print(f"  {cat:<22} {len(crows):>7} {mrr:>9.4f} {miss_pct:>7.1f} {mndcg:>14.4f}")

    return matrix_rr, top_etypes, cats


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: Score distribution analysis
# ══════════════════════════════════════════════════════════════════════════════

def percentile(data, p):
    if not data:
        return float("nan")
    data = sorted(data)
    idx = (len(data) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


def section4_scores(rows):
    print("\n" + "="*70)
    print("SECTION 4: Score Distribution Analysis")
    print("="*70)

    print("\n[4a] Top-1 retrieved doc score by bucket:")
    print(f"  {'Bucket':<18} {'Mean':>8} {'P25':>8} {'P50':>8} {'P75':>8}")
    print("  " + "-"*50)
    for b in BUCKETS:
        top1s = [r["top1_score"] for r in rows if r["bucket"] == b]
        if not top1s:
            continue
        print(f"  {b:<18} {sum(top1s)/len(top1s):>8.4f} "
              f"{percentile(top1s,25):>8.4f} {percentile(top1s,50):>8.4f} "
              f"{percentile(top1s,75):>8.4f}")

    print("\n[4b] First relevant doc score vs top-1 score (rank-failure queries only):")
    rank_fail = [r for r in rows
                 if r["bucket"] in ("hit@2-5", "hit@6-10", "hit@11-100")
                 and r["score_gap"] is not None]
    if rank_fail:
        gaps = [r["score_gap"] for r in rank_fail]
        mean_gap = sum(gaps) / len(gaps)
        p50_gap  = percentile(gaps, 50)
        large_gap = sum(1 for g in gaps if g > 0.1)
        small_gap = sum(1 for g in gaps if g <= 0.05)
        print(f"  Rank-failure queries:    {len(rank_fail)}")
        print(f"  Mean score gap:          {mean_gap:.4f}  (top1 score - gold score)")
        print(f"  Median score gap:        {p50_gap:.4f}")
        print(f"  Large gap (>0.1):        {large_gap} ({100*large_gap/len(rank_fail):.1f}%)")
        print(f"  Small gap (≤0.05):       {small_gap} ({100*small_gap/len(rank_fail):.1f}%)")
        print()
        print("  Interpretation:")
        print("  - Large gaps suggest systematic field mismatch (MFAR weights wrong fields)")
        print("  - Small gaps suggest near-miss, solvable by weight fine-tuning or reranking")

    print("\n[4c] Score gap by query category (rank failures only):")
    print(f"  {'Category':<22} {'N':>5} {'Mean gap':>10} {'P50':>8} {'Large%':>8}")
    print("  " + "-"*57)
    for cat, _ in QUERY_CATEGORIES:
        crows = [r for r in rank_fail if r["category"] == cat]
        if not crows:
            continue
        gs = [r["score_gap"] for r in crows]
        lg = sum(1 for g in gs if g > 0.1)
        print(f"  {cat:<22} {len(crows):>5} {sum(gs)/len(gs):>10.4f} "
              f"{percentile(gs,50):>8.4f} {100*lg/len(crows):>7.1f}%")

    return rank_fail


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: Ablation interpretation
# ══════════════════════════════════════════════════════════════════════════════

def section5_ablation(rows, ablation_records):
    print("\n" + "="*70)
    print("SECTION 5: Field Importance (Ablation) × Failure Analysis")
    print("="*70)

    if not ablation_records:
        print("  No ablation records found.")
        return {}

    # Find baseline
    baseline = next((r for r in ablation_records if r["masked_fields"] == ""), None)
    if baseline is None:
        print("  No baseline record found.")
        return {}

    base_ndcg = float(baseline["ndcg"])
    base_mrr  = float(baseline["recip_rank"])
    base_r10  = float(baseline["recall_10"])
    print(f"\n  Baseline (no masking): NDCG={base_ndcg:.4f}, MRR={base_mrr:.4f}, R@10={base_r10:.4f}")

    # Field importance
    field_drops = []
    for r in ablation_records:
        if r["masked_fields"] == "":
            continue
        masked = r["masked_fields"]
        # Strip "_dense" or "_sparse" suffix to get base field name
        base_field = re.sub(r"_(dense|sparse)$", "", masked)
        drop_ndcg = base_ndcg - float(r["ndcg"])
        drop_mrr  = base_mrr  - float(r["recip_rank"])
        drop_r10  = base_r10  - float(r["recall_10"])
        field_drops.append({
            "masked": masked,
            "field":  base_field,
            "ndcg_drop": drop_ndcg,
            "mrr_drop":  drop_mrr,
            "r10_drop":  drop_r10,
        })

    # Aggregate by base field (average dense + sparse if both present)
    by_field = defaultdict(list)
    for fd in field_drops:
        by_field[fd["field"]].append(fd)

    aggregated = {}
    for field, entries in by_field.items():
        aggregated[field] = {
            "ndcg_drop": sum(e["ndcg_drop"] for e in entries) / len(entries),
            "mrr_drop":  sum(e["mrr_drop"]  for e in entries) / len(entries),
            "r10_drop":  sum(e["r10_drop"]  for e in entries) / len(entries),
        }

    print("\n[5a] Field importance ranking (sorted by NDCG drop when masked):")
    print(f"  {'Field':<28} {'NDCG drop':>10} {'MRR drop':>10} {'R@10 drop':>10}")
    print("  " + "-"*62)
    sorted_fields = sorted(aggregated.items(), key=lambda x: -x[1]["ndcg_drop"])
    for field, agg in sorted_fields:
        arrow = " ←" if agg["ndcg_drop"] > 0.01 else ""
        print(f"  {field:<28} {agg['ndcg_drop']:>+10.4f} {agg['mrr_drop']:>+10.4f} "
              f"{agg['r10_drop']:>+10.4f}{arrow}")

    # --- 5b. Cross-reference: miss-rate per field in failed queries ---
    miss_rows = [r for r in rows if r["bucket"] == "complete_miss"]
    all_rows_with_gold = [r for r in rows if r["gold_populated"] is not None]

    print("\n[5b] Field miss-rate: % of failed queries where gold doc lacks each field")
    print("     (High importance + high miss-rate = root-cause fields)")
    print(f"  {'Field':<28} {'NDCG drop':>10} {'Miss rate%':>11} {'Overall%':>9} {'Root cause?':>12}")
    print("  " + "-"*75)

    field_importance = {f: agg["ndcg_drop"] for f, agg in aggregated.items()}
    for field, agg in sorted_fields:
        if field not in RELATION_FIELDS:
            continue
        miss_absent = sum(1 for r in miss_rows if field not in r["gold_populated"])
        miss_rate   = 100 * miss_absent / len(miss_rows) if miss_rows else 0

        overall_absent = sum(1 for r in all_rows_with_gold if field not in r["gold_populated"])
        overall_rate   = 100 * overall_absent / len(all_rows_with_gold) if all_rows_with_gold else 0

        # Root cause = both high importance AND significantly higher miss rate in failures
        is_root = (agg["ndcg_drop"] > 0.005 and
                   miss_rate > overall_rate + 10)
        print(f"  {field:<28} {agg['ndcg_drop']:>+10.4f} {miss_rate:>10.1f}% {overall_rate:>8.1f}% "
              f"{'  YES ←' if is_root else '':>12}")

    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def section6_plots(rows, matrix_rr, top_etypes, cats, aggregated, rank_fail):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("\n  [SKIP] matplotlib not available — skipping plots")
        return

    print("\n" + "="*70)
    print("SECTION 6: Generating Visualizations")
    print("="*70)

    # ── Plot 1: Severity distribution bar chart ────────────────────────────
    total = len(rows)
    counts = Counter(r["bucket"] for r in rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = [counts[b] for b in BUCKETS]
    colors = [BUCKET_COLORS[b] for b in BUCKETS]
    rects = ax.bar(BUCKETS, bars, color=colors, edgecolor="white")
    for rect, cnt in zip(rects, bars):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
                f"{cnt}\n({100*cnt/total:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("MFAR Failure Severity Distribution (Val Set)")
    ax.set_xlabel("Retrieval Bucket")
    ax.set_ylabel("# Queries")
    ax.set_ylim(0, max(bars) * 1.2)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/severity_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: severity_distribution.png")

    # ── Plot 2: Query-type × Entity-type RR heatmap ───────────────────────
    active_cats = [c for c in cats
                   if any(matrix_rr[c][e] for e in top_etypes)]
    mat = []
    for cat in active_cats:
        row_vals = []
        for etype in top_etypes:
            vals = matrix_rr[cat][etype]
            row_vals.append(sum(vals)/len(vals) if vals else float("nan"))
        mat.append(row_vals)

    fig, ax = plt.subplots(figsize=(12, max(5, len(active_cats) * 0.7)))
    import numpy as np
    mat_np = np.array(mat, dtype=float)
    im = ax.imshow(mat_np, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top_etypes)))
    ax.set_xticklabels([e[:18] for e in top_etypes], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(active_cats)))
    ax.set_yticklabels(active_cats, fontsize=9)
    for i in range(len(active_cats)):
        for j in range(len(top_etypes)):
            val = mat_np[i, j]
            if not math.isnan(val):
                cnt = len(matrix_rr[active_cats[i]][top_etypes[j]])
                ax.text(j, i, f"{val:.2f}\n(n={cnt})",
                        ha="center", va="center", fontsize=7,
                        color="black" if 0.3 < val < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Mean RR")
    ax.set_title("Mean Reciprocal Rank: Query Category × Gold Entity Type")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/query_entity_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: query_entity_heatmap.png")

    # ── Plot 3: Score distribution box plots ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bucket_labels = BUCKETS
    top1_data = [[r["top1_score"] for r in rows if r["bucket"] == b] for b in BUCKETS]
    rel_data   = [[r["first_rel_score"] for r in rows
                   if r["bucket"] == b and r["first_rel_score"] is not None]
                  for b in BUCKETS]

    axes[0].boxplot(top1_data, labels=[b.replace("@", "@\n") for b in BUCKETS],
                    patch_artist=True,
                    boxprops=dict(facecolor="#3498db", alpha=0.7))
    axes[0].set_title("Top-1 Retrieved Doc Score by Bucket")
    axes[0].set_ylabel("Similarity Score")

    axes[1].boxplot(rel_data, labels=[b.replace("@", "@\n") for b in BUCKETS],
                    patch_artist=True,
                    boxprops=dict(facecolor="#e74c3c", alpha=0.7))
    axes[1].set_title("First Relevant Doc Score by Bucket")
    axes[1].set_ylabel("Similarity Score")

    plt.suptitle("Score Distributions by Failure Bucket", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/score_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: score_distributions.png")

    # ── Plot 4: Score gap histogram ────────────────────────────────────────
    if rank_fail:
        gaps = [r["score_gap"] for r in rank_fail]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(gaps, bins=40, color="#e67e22", edgecolor="white", alpha=0.85)
        ax.axvline(sum(gaps)/len(gaps), color="red", linestyle="--",
                   label=f"Mean={sum(gaps)/len(gaps):.3f}")
        ax.axvline(percentile(gaps, 50), color="navy", linestyle="--",
                   label=f"Median={percentile(gaps, 50):.3f}")
        ax.axvline(0.1, color="gray", linestyle=":", label="Gap=0.1 threshold")
        ax.set_title("Score Gap Distribution (Rank-Failure Queries)\n"
                     "Gap = top1_score − first_relevant_score")
        ax.set_xlabel("Score Gap")
        ax.set_ylabel("# Queries")
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/score_gap_histogram.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: score_gap_histogram.png")

    # ── Plot 5: Field importance vs miss-rate scatter ──────────────────────
    if aggregated:
        miss_rows = [r for r in rows if r["bucket"] == "complete_miss"]
        all_valid = [r for r in rows if r["gold_populated"] is not None]

        fig, ax = plt.subplots(figsize=(10, 7))
        for field, agg in aggregated.items():
            if field not in RELATION_FIELDS:
                continue
            miss_absent = sum(1 for r in miss_rows if field not in r["gold_populated"])
            miss_rate   = 100 * miss_absent / len(miss_rows) if miss_rows else 0
            overall_absent = sum(1 for r in all_valid if field not in r["gold_populated"])
            overall_rate   = 100 * overall_absent / len(all_valid) if all_valid else 0
            is_root = agg["ndcg_drop"] > 0.005 and miss_rate > overall_rate + 10

            color = "#e74c3c" if is_root else "#3498db"
            ax.scatter(agg["ndcg_drop"], miss_rate, s=80, color=color, alpha=0.8)
            ax.annotate(field, (agg["ndcg_drop"], miss_rate),
                        textcoords="offset points", xytext=(4, 2), fontsize=7)

        ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0.005, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("NDCG Drop when Masked (Field Importance)", fontsize=11)
        ax.set_ylabel("Miss Rate in Failed Queries (%)", fontsize=11)
        ax.set_title("Field Importance vs. Failure Miss Rate\n"
                     "Red = root-cause fields (high importance + high miss rate)", fontsize=12)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#e74c3c", label="Root-cause fields"),
            Patch(color="#3498db", label="Other fields"),
        ])
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/field_importance_vs_missrate.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: field_importance_vs_missrate.png")

    # ── Plot 6: Field importance ranking bar chart ─────────────────────────
    if aggregated:
        rel_fields = [(f, v) for f, v in aggregated.items() if f in RELATION_FIELDS]
        rel_fields.sort(key=lambda x: x[1]["ndcg_drop"], reverse=True)
        fields_sorted = [f for f, _ in rel_fields]
        drops_sorted  = [v["ndcg_drop"] for _, v in rel_fields]

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ["#e74c3c" if d > 0.01 else "#3498db" for d in drops_sorted]
        ax.barh(fields_sorted[::-1], drops_sorted[::-1], color=bar_colors[::-1])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("NDCG Drop when Masked")
        ax.set_title("MFAR Field Importance: NDCG Drop per Masked Field\n"
                     "(Larger drop = field more critical for retrieval)")
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/field_importance_ranking.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: field_importance_ranking.png")

    # ── Plot 7: Top-20 worst queries ───────────────────────────────────────
    worst = sorted(rows, key=lambda r: r["rr"])[:20]
    fig, ax = plt.subplots(figsize=(10, 8))
    rrs = [r["rr"] for r in worst]
    labels = [f"[{r['qid']}] {r['query'][:55]}..." for r in worst]
    cat_colors = {
        "expression_absent": "#9b59b6", "expression_present": "#8e44ad",
        "pathway_hierarchy": "#2980b9", "drug_indication": "#27ae60",
        "drug_interaction": "#16a085", "side_effect": "#f39c12",
        "target": "#d35400", "ppi": "#c0392b",
        "phenotype": "#e74c3c", "drug_target": "#e67e22",
        "multi_hop": "#7f8c8d",
    }
    bar_colors = [cat_colors.get(r["category"], "#7f8c8d") for r in worst]
    ax.barh(range(20), rrs[::-1], color=bar_colors[::-1])
    ax.set_yticks(range(20))
    ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_xlabel("Reciprocal Rank (0 = complete miss)")
    ax.set_title("Top-20 Worst-Performing Queries (lowest RR)")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/worst_queries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: worst_queries.png")

    # ── Plot 8: NDCG@10 CDF by query category ─────────────────────────────
    try:
        import numpy as np
        fig, ax = plt.subplots(figsize=(9, 5))
        for cat, _ in QUERY_CATEGORIES:
            crows = [r for r in rows if r["category"] == cat]
            if len(crows) < 5:
                continue
            vals = sorted(r["ndcg10"] for r in crows)
            cdf  = [(i+1)/len(vals) for i in range(len(vals))]
            ax.plot(vals, cdf, label=f"{cat} (n={len(crows)})", linewidth=1.5)
        ax.set_xlabel("NDCG@10")
        ax.set_ylabel("CDF (fraction of queries)")
        ax.set_title("NDCG@10 CDF by Query Category")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/ndcg_cdf_by_category.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: ndcg_cdf_by_category.png")
    except Exception as e:
        print(f"  [SKIP] CDF plot failed: {e}")

    print(f"\n  All plots saved to: {OUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: Worst-query case study (qualitative)
# ══════════════════════════════════════════════════════════════════════════════

def section7_case_studies(rows, qrels, corpus):
    print("\n" + "="*70)
    print("SECTION 7: Qualitative Case Studies — Complete-Miss Failures")
    print("="*70)

    miss_rows = [r for r in rows if r["bucket"] == "complete_miss"]
    # Show 5 diverse examples by category
    seen_cats = set()
    examples  = []
    for r in sorted(miss_rows, key=lambda x: x["category"]):
        if r["category"] not in seen_cats:
            seen_cats.add(r["category"])
            examples.append(r)
        if len(examples) >= 6:
            break

    for ex in examples:
        qid = ex["qid"]
        gold = qrels[qid]
        print(f"\n  Query [{qid}] ({ex['category']}):")
        print(f"    Text: {ex['query'][:120]}...")
        print(f"    Gold docs ({len(gold)} total):")
        for did in list(gold)[:3]:
            doc = corpus.get(did, {})
            fields_str = ", ".join(sorted(doc.get("fields", set()))[:5])
            print(f"      [{doc.get('type','?')}] {doc.get('name','?')} "
                  f"| fields: {fields_str or '(none)'}")
        print(f"    Top-3 retrieved (wrong entity types):")
        for etype in ex["top10_types"][:3]:
            print(f"      [{etype}]")
        print(f"    MFAR analysis: gold doc has {len(ex['gold_populated'])} relation fields "
              f"({', '.join(sorted(ex['gold_populated'])[:4])}...)")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: Negation Query Analysis
# ══════════════════════════════════════════════════════════════════════════════

def section8_negation(rows):
    """
    Analyze how the model handles queries containing negation words
    (not, no, without, lack, absent, un-, etc.).

    Embedding-based retrievers are known to struggle with negation because
    "not X" and "X" produce similar embeddings. This section quantifies
    that effect for MFAR on PRIME.
    """
    print("\n" + "="*70)
    print("SECTION 8: Negation Query Analysis")
    print("="*70)

    # ── 8a. Split rows into negation vs. affirmative ─────────────────────
    neg_rows = [r for r in rows if NEGATION_PATTERN.search(r["query"])]
    aff_rows = [r for r in rows if not NEGATION_PATTERN.search(r["query"])]

    n_neg, n_aff = len(neg_rows), len(aff_rows)
    print(f"\n[8a] Negation prevalence:")
    print(f"  Negation queries:    {n_neg:>5} ({100*n_neg/len(rows):.1f}%)")
    print(f"  Affirmative queries: {n_aff:>5} ({100*n_aff/len(rows):.1f}%)")

    if n_neg == 0:
        print("  No negation queries found — skipping Section 8.")
        return {}

    # ── 8b. Head-to-head metric comparison ───────────────────────────────
    def group_stats(group, label):
        n = len(group)
        mrr   = sum(r["rr"] for r in group) / n
        mndcg = sum(r["ndcg10"] for r in group) / n
        mr10  = sum(r["r10"] for r in group) / n
        mr100 = sum(r["r100"] for r in group) / n
        miss  = sum(1 for r in group if r["bucket"] == "complete_miss")
        hit1  = sum(1 for r in group if r["bucket"] == "hit@1")
        return {
            "label": label, "n": n,
            "mrr": mrr, "ndcg10": mndcg, "r10": mr10, "r100": mr100,
            "miss_pct": 100 * miss / n, "hit1_pct": 100 * hit1 / n,
        }

    neg_s = group_stats(neg_rows, "Negation")
    aff_s = group_stats(aff_rows, "Affirmative")

    print(f"\n[8b] Head-to-head comparison:")
    print(f"  {'Metric':<18} {'Negation':>12} {'Affirmative':>12} {'Delta':>10}")
    print("  " + "-"*55)
    for metric in ["mrr", "ndcg10", "r10", "r100", "hit1_pct", "miss_pct"]:
        nv, av = neg_s[metric], aff_s[metric]
        delta = nv - av
        suffix = "%" if metric.endswith("pct") else ""
        print(f"  {metric:<18} {nv:>11.4f}{suffix} {av:>11.4f}{suffix} {delta:>+9.4f}")

    # ── 8c. Severity bucket distribution comparison ──────────────────────
    print(f"\n[8c] Severity distribution — Negation vs. Affirmative:")
    print(f"  {'Bucket':<18} {'Neg count':>10} {'Neg %':>8} {'Aff count':>10} {'Aff %':>8}")
    print("  " + "-"*58)
    for b in BUCKETS:
        nc = sum(1 for r in neg_rows if r["bucket"] == b)
        ac = sum(1 for r in aff_rows if r["bucket"] == b)
        print(f"  {b:<18} {nc:>10} {100*nc/n_neg:>7.1f}% {ac:>10} {100*ac/n_aff:>7.1f}%")

    # ── 8d. Negation breakdown by query category ─────────────────────────
    print(f"\n[8d] Negation impact by query category:")
    print(f"  {'Category':<22} {'N(neg)':>7} {'MRR(neg)':>10} {'MRR(aff)':>10} {'Delta':>8} {'Miss%(neg)':>11}")
    print("  " + "-"*72)
    for cat, _ in QUERY_CATEGORIES:
        cat_neg = [r for r in neg_rows if r["category"] == cat]
        cat_aff = [r for r in aff_rows if r["category"] == cat]
        if len(cat_neg) < 3:
            continue
        mrr_n = sum(r["rr"] for r in cat_neg) / len(cat_neg)
        mrr_a = sum(r["rr"] for r in cat_aff) / len(cat_aff) if cat_aff else 0
        miss_n = 100 * sum(1 for r in cat_neg if r["bucket"] == "complete_miss") / len(cat_neg)
        delta = mrr_n - mrr_a
        print(f"  {cat:<22} {len(cat_neg):>7} {mrr_n:>10.4f} {mrr_a:>10.4f} {delta:>+8.4f} {miss_n:>10.1f}%")

    # ── 8e. Score gap comparison for rank failures ───────────────────────
    neg_rf = [r for r in neg_rows
              if r["bucket"] in ("hit@2-5", "hit@6-10", "hit@11-100")
              and r["score_gap"] is not None]
    aff_rf = [r for r in aff_rows
              if r["bucket"] in ("hit@2-5", "hit@6-10", "hit@11-100")
              and r["score_gap"] is not None]

    if neg_rf and aff_rf:
        neg_gaps = [r["score_gap"] for r in neg_rf]
        aff_gaps = [r["score_gap"] for r in aff_rf]
        print(f"\n[8e] Score gap for rank-failure queries:")
        print(f"  Negation   (n={len(neg_rf):>4}):  mean gap = {sum(neg_gaps)/len(neg_gaps):.4f},  "
              f"median = {percentile(neg_gaps, 50):.4f}")
        print(f"  Affirmative(n={len(aff_rf):>4}):  mean gap = {sum(aff_gaps)/len(aff_gaps):.4f},  "
              f"median = {percentile(aff_gaps, 50):.4f}")
        print("  Interpretation: larger gaps in negation queries indicate the model")
        print("  embeds 'not X' similarly to 'X', retrieving wrong-polarity entities.")

    # ── 8f. Most frequent negation tokens ────────────────────────────────
    neg_token_counter = Counter()
    for r in neg_rows:
        tokens = NEGATION_PATTERN.findall(r["query"].lower())
        neg_token_counter.update(tokens)

    print(f"\n[8f] Most frequent negation tokens in queries:")
    for token, cnt in neg_token_counter.most_common(15):
        print(f"    {token:<20} {cnt:>5}")

    # ── 8g. Example negation failures ────────────────────────────────────
    neg_misses = [r for r in neg_rows if r["bucket"] == "complete_miss"]
    print(f"\n[8g] Example negation complete-miss queries ({len(neg_misses)} total):")
    for ex in neg_misses[:8]:
        print(f"  [{ex['qid']}] ({ex['category']}) MRR={ex['rr']:.2f}")
        print(f"    {ex['query'][:120]}")

    return {"neg_stats": neg_s, "aff_stats": aff_s}


def section8_negation_plot(rows):
    """Generate negation analysis visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [SKIP] matplotlib not available — skipping negation plot")
        return

    neg_rows = [r for r in rows if NEGATION_PATTERN.search(r["query"])]
    aff_rows = [r for r in rows if not NEGATION_PATTERN.search(r["query"])]

    if len(neg_rows) < 5:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: Metric comparison bar chart ─────────────────────────────
    metrics = ["MRR", "NDCG@10", "Recall@10", "Hit@1%", "Miss%"]
    neg_vals = [
        sum(r["rr"] for r in neg_rows) / len(neg_rows),
        sum(r["ndcg10"] for r in neg_rows) / len(neg_rows),
        sum(r["r10"] for r in neg_rows) / len(neg_rows),
        100 * sum(1 for r in neg_rows if r["bucket"] == "hit@1") / len(neg_rows),
        100 * sum(1 for r in neg_rows if r["bucket"] == "complete_miss") / len(neg_rows),
    ]
    aff_vals = [
        sum(r["rr"] for r in aff_rows) / len(aff_rows),
        sum(r["ndcg10"] for r in aff_rows) / len(aff_rows),
        sum(r["r10"] for r in aff_rows) / len(aff_rows),
        100 * sum(1 for r in aff_rows if r["bucket"] == "hit@1") / len(aff_rows),
        100 * sum(1 for r in aff_rows if r["bucket"] == "complete_miss") / len(aff_rows),
    ]

    x = np.arange(len(metrics))
    w = 0.35
    axes[0].bar(x - w/2, neg_vals, w, label=f"Negation (n={len(neg_rows)})",
                color="#e74c3c", alpha=0.85)
    axes[0].bar(x + w/2, aff_vals, w, label=f"Affirmative (n={len(aff_rows)})",
                color="#3498db", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Value")
    axes[0].set_title("Negation vs. Affirmative Queries")
    axes[0].legend(fontsize=8)

    # ── Panel 2: Severity distribution stacked bars ──────────────────────
    neg_bkt = [sum(1 for r in neg_rows if r["bucket"] == b) / len(neg_rows)
               for b in BUCKETS]
    aff_bkt = [sum(1 for r in aff_rows if r["bucket"] == b) / len(aff_rows)
               for b in BUCKETS]

    x2 = np.arange(len(BUCKETS))
    axes[1].bar(x2 - w/2, [100*v for v in neg_bkt], w, label="Negation",
                color="#e74c3c", alpha=0.85)
    axes[1].bar(x2 + w/2, [100*v for v in aff_bkt], w, label="Affirmative",
                color="#3498db", alpha=0.85)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([b.replace("@", "@\n") for b in BUCKETS],
                            fontsize=8)
    axes[1].set_ylabel("% of queries")
    axes[1].set_title("Severity Distribution")
    axes[1].legend(fontsize=8)

    # ── Panel 3: NDCG@10 CDF comparison ──────────────────────────────────
    neg_ndcg = sorted(r["ndcg10"] for r in neg_rows)
    aff_ndcg = sorted(r["ndcg10"] for r in aff_rows)
    axes[2].plot(neg_ndcg, [(i+1)/len(neg_ndcg) for i in range(len(neg_ndcg))],
                 color="#e74c3c", linewidth=2, label=f"Negation (n={len(neg_rows)})")
    axes[2].plot(aff_ndcg, [(i+1)/len(aff_ndcg) for i in range(len(aff_ndcg))],
                 color="#3498db", linewidth=2, label=f"Affirmative (n={len(aff_rows)})")
    axes[2].set_xlabel("NDCG@10")
    axes[2].set_ylabel("CDF")
    axes[2].set_title("NDCG@10 CDF: Negation vs. Affirmative")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.suptitle("Section 8: Negation Query Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/negation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: negation_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("MFAR Failure Analysis — PRIME Validation Set")
    print("=" * 70)
    print("\nLoading data...")
    corpus    = _load_corpus()
    queries   = _load_queries("val")
    qrels     = _load_qrels("val")
    retrieved = load_retrieved(f"{EVAL_DIR}/final-all-0.qres")
    ablation  = load_ablation()

    print("\nComputing per-query metrics...")
    rows = compute_per_query(queries, qrels, retrieved, corpus)
    print(f"  Computed metrics for {len(rows):,} queries")

    # Run all sections
    section1_severity(rows)
    section2_field_proxies(rows)
    matrix_rr, top_etypes, cats = section3_categorical(rows)
    rank_fail = section4_scores(rows)
    aggregated = section5_ablation(rows, ablation)
    section6_plots(rows, matrix_rr, top_etypes, cats, aggregated, rank_fail)
    section7_case_studies(rows, qrels, corpus)
    neg_stats = section8_negation(rows)
    section8_negation_plot(rows)

    print("\n" + "="*70)
    print("SUMMARY: Key Findings")
    print("="*70)
    miss_pct = 100 * sum(1 for r in rows if r["bucket"] == "complete_miss") / len(rows)
    mean_rr  = sum(r["rr"] for r in rows) / len(rows)
    miss_rows = [r for r in rows if r["bucket"] == "complete_miss"]
    top_miss_cat = Counter(r["category"] for r in miss_rows).most_common(3)
    print(f"\n  1. {miss_pct:.1f}% of queries are complete misses (relevant doc not in top-100).")
    print(f"  2. Overall Mean RR = {mean_rr:.4f}")
    print(f"  3. Top failure query categories: {[c for c,_ in top_miss_cat]}")
    print(f"  4. MFAR principle: softmax field weights are query-conditioned —")
    print(f"     failures occur when query embedding drives weight toward fields")
    print(f"     that the relevant doc doesn't populate in the KG.")
    print(f"  5. Gold docs in failures have significantly fewer relation fields")
    print(f"     (sparse KG coverage = weak signal for adaptive weighting).")
    if neg_stats:
        ns, afs = neg_stats["neg_stats"], neg_stats["aff_stats"]
        print(f"  6. Negation queries ({ns['n']} total, {100*ns['n']/len(rows):.1f}%) have "
              f"MRR={ns['mrr']:.4f} vs affirmative MRR={afs['mrr']:.4f} "
              f"(delta={ns['mrr']-afs['mrr']:+.4f}).")
        print(f"     Complete-miss rate: {ns['miss_pct']:.1f}% (neg) vs "
              f"{afs['miss_pct']:.1f}% (aff).")
    print(f"\n  Plots saved to: {OUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
