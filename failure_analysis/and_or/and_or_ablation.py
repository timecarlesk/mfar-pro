"""
AND/OR Constraint Ablation Study — PRIME Dataset
==================================================
Staged execution:
  Stage 1: field count classification + MRR vs field_count + permutation test
  Stage 2: softmax bottleneck + additive aggregation (if Stage 1 passes)
  Stage 3: oracle field non-empty check (if Stage 2 passes)

Run from project root:
  python failure_analysis/and_or/and_or_ablation.py [val] [test]
"""

import json
import re
import os
import sys
import math
import random
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    RELATION_FIELDS, load_corpus_full, load_queries, load_qrels,
    load_retrieved, dcg,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/prime"
EVAL_DIR = "output/prime_eval"
CKPT_DIR = "output/prime"
OUT_DIR  = "output/failure_analysis/and_or"
os.makedirs(OUT_DIR, exist_ok=True)

SPLIT_QRES = {
    "val":  f"{EVAL_DIR}/final-all-0.qres",
    "test": f"{EVAL_DIR}/final-additional-all-0.qres",
}

# ── Field Constraint Detectors ───────────────────────────────────────────────
FIELD_DETECTORS = {
    "ppi":                re.compile(r"\binteract(?:s|ing|ion)?\b.*(?:protein|gene)|protein.protein|\bppi\b", re.I),
    "expression_present": re.compile(r"express(?:ed|ion)?\s+in|found\s+in\s+(?:\w+\s+){0,2}tissue", re.I),
    "expression_absent":  re.compile(r"(?:not|lack|absent|no)\s+(?:\w+\s+){0,3}express|unexpress", re.I),
    "indication":         re.compile(r"\bindication\b|(?:treat(?:s|ed|ing|ment)?|therap\w*)\s+(?:of|for)\b", re.I),
    "target":             re.compile(r"\btarget(?:s|ed|ing)?\b", re.I),
    "side_effect":        re.compile(r"side[\s-]?effect|adverse\s+(?:effect|reaction)|toxici", re.I),
    "parent-child":       re.compile(r"hierarch|subtype|sub[\s-]?type|classif.*(?:disease|condition)", re.I),
    "phenotype":          re.compile(r"\bphenotype\b|(?:exhibit|present|display)\s+(?:\w+\s+){0,2}symptom", re.I),
    "enzyme":             re.compile(r"\benzyme\b|metaboli[sz]|catalyz", re.I),
    "carrier":            re.compile(r"\bcarrier\b", re.I),
    "transporter":        re.compile(r"\btransport(?:s|er|ed|ing)?\b", re.I),
    "contraindication":   re.compile(r"\bcontraindic|should\s+not\s+be\s+(?:treat|manage|prescri|use)", re.I),
    "synergistic":        re.compile(r"\bsynergist", re.I),
}

# Field detector key → BM25 index directory name mapping
FIELD_TO_BM25_DIR = {
    "ppi": "ppi",
    "expression_present": "expression present",
    "expression_absent": "expression absent",
    "indication": "indication",
    "target": "target",
    "side_effect": "side effect",
    "parent-child": "parent-child",
    "phenotype": "phenotype present",
    "enzyme": "enzyme",
    "carrier": "carrier",
    "transporter": "transporter",
    "contraindication": "contraindication",
    "synergistic": "synergistic interaction",
}

OR_PATTERN = re.compile(r"\bor\b(?!\s+(?:more|less|other))", re.I)

# Query category patterns (from negation study)
QUERY_CATEGORIES = [
    ("expression_absent",  r"(?:not|lack|absent|no)\s+express|unexpress|without\s+express"),
    ("expression_present", r"express(?:ed|ion|es)?\s+(?:in|by|of)|show\s+express"),
    ("pathway_hierarchy",  r"pathways?\s+connect|above.*below|hierarch|signal.*pathway"),
    ("drug_indication",    r"indicat|phase\s+[IVX]+|clinical\s+trial|treat.*diseas|approv"),
    ("drug_interaction",   r"carrier|transporter|enzyme|metaboli"),
    ("side_effect",        r"side[\s-]?effect|adverse\s+effect|toxici"),
    ("target",             r"\btarget\b"),
    ("ppi",                r"\bprotein.protein|ppi\b|\binteract(?:s|ing|ion)?\b"),
    ("phenotype",          r"\bphenotype\b|phenotyp"),
    ("other",              r"."),
]


def classify_query_category(text):
    for cat, pat in QUERY_CATEGORIES:
        if re.search(pat, text, re.I):
            return cat
    return "other"


def detect_fields(query_text):
    """Detect which mFAR fields a query constrains."""
    detected = set()
    for field, pattern in FIELD_DETECTORS.items():
        if pattern.search(query_text):
            detected.add(field)
    return detected


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_row(qid, query_text, gold, docs, corpus):
    top100 = docs[:100]
    top_ids = [d[0] for d in top100]
    scores = [d[1] for d in top100]

    first_rel = next((i for i, did in enumerate(top_ids) if did in gold), -1)
    rr = 0.0 if first_rel < 0 else 1.0 / (first_rel + 1)
    hit1 = 1 if first_rel == 0 else 0
    hit5 = 1 if 0 <= first_rel < 5 else 0
    miss = 1 if first_rel < 0 else 0

    gains10 = [1 if did in gold else 0 for did in top_ids[:10]]
    ideal = dcg([1] * min(len(gold), 10), 10)
    ndcg10 = dcg(gains10, 10) / ideal if ideal > 0 else 0.0

    gold_types = [corpus[did]["type"] for did in gold if did in corpus]
    category = classify_query_category(query_text)
    detected = detect_fields(query_text)

    # Top-1 wrong doc (for additive analysis)
    top1_wrong = None
    for did, sc in top100:
        if did not in gold:
            top1_wrong = did
            break

    return {
        "qid": qid, "query": query_text,
        "rr": rr, "hit1": hit1, "hit5": hit5, "miss": miss,
        "ndcg10": ndcg10, "first_rel": first_rel,
        "gold_ids": gold, "gold_types": gold_types,
        "category": category,
        "detected_fields": detected,
        "field_count": len(detected),
        "has_or": bool(OR_PATTERN.search(query_text)),
        "top1_wrong": top1_wrong,
    }


def compute_all_rows(queries, qrels, retrieved, corpus):
    rows = []
    for qid, text in queries.items():
        if qid not in qrels:
            continue
        rows.append(compute_row(qid, text, qrels[qid],
                                retrieved.get(qid, []), corpus))
    return rows


def group_metrics(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "mrr": 0, "hit1": 0, "hit5": 0, "miss_pct": 0, "ndcg10": 0}
    return {
        "n": n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "hit1": 100 * sum(r["hit1"] for r in rows) / n,
        "hit5": 100 * sum(r["hit5"] for r in rows) / n,
        "miss_pct": 100 * sum(r["miss"] for r in rows) / n,
        "ndcg10": sum(r["ndcg10"] for r in rows) / n,
    }


def pm(label, m, indent="  "):
    print(f"{indent}{label:<22} n={m['n']:>5}  MRR={m['mrr']:.3f}  "
          f"H@1={m['hit1']:5.1f}%  Miss={m['miss_pct']:5.1f}%  "
          f"NDCG@10={m['ndcg10']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: Field Count + MRR + Permutation Test
# ══════════════════════════════════════════════════════════════════════════════

def stage1(rows):
    print("\n" + "=" * 70)
    print("STAGE 1: Field Count Distribution & MRR vs Field Count")
    print("=" * 70)

    # Distribution
    fc_counter = Counter(r["field_count"] for r in rows)
    total = len(rows)
    print(f"\n  Field count distribution (n={total}):")
    for fc in sorted(fc_counter):
        print(f"    {fc}-field: {fc_counter[fc]:>5} ({100*fc_counter[fc]/total:5.1f}%)")

    # MRR by field count
    buckets = {0: [], 1: [], 2: [], "3+": []}
    for r in rows:
        fc = r["field_count"]
        if fc >= 3:
            buckets["3+"].append(r)
        else:
            buckets[fc].append(r)

    print(f"\n  Performance by field count:")
    for label in [0, 1, 2, "3+"]:
        pm(f"{label}-field", group_metrics(buckets[label]))

    # Precision audit: print 50 random 2-field queries
    two_field = buckets[2]
    audit_sample = random.sample(two_field, min(50, len(two_field)))
    print(f"\n  --- Precision Audit: 50 random 2-field queries ---")
    for i, r in enumerate(audit_sample):
        print(f"  [{r['qid']}] fields={sorted(r['detected_fields'])}")
        print(f"    {r['query'][:150]}")
        if i < 49:
            print()

    # Permutation test: 1-field vs 2-field
    print(f"\n  --- Permutation Test: 1-field vs 2-field ---")
    one_field = buckets[1]
    if len(one_field) < 10 or len(two_field) < 10:
        print(f"  [SKIP] Insufficient data (1-field={len(one_field)}, 2-field={len(two_field)})")
        return buckets, None

    observed_delta = group_metrics(one_field)["mrr"] - group_metrics(two_field)["mrr"]
    pooled = one_field + two_field
    n1 = len(one_field)
    N_PERM = 10000
    count_ge = 0
    random.seed(42)
    for _ in range(N_PERM):
        random.shuffle(pooled)
        perm_mrr1 = sum(r["rr"] for r in pooled[:n1]) / n1
        perm_mrr2 = sum(r["rr"] for r in pooled[n1:]) / (len(pooled) - n1)
        if perm_mrr1 - perm_mrr2 >= observed_delta:
            count_ge += 1
    p_value = count_ge / N_PERM

    # Effect size (Cohen's d)
    rr1 = [r["rr"] for r in one_field]
    rr2 = [r["rr"] for r in two_field]
    mean1, mean2 = sum(rr1) / len(rr1), sum(rr2) / len(rr2)
    var1 = sum((x - mean1) ** 2 for x in rr1) / max(len(rr1) - 1, 1)
    var2 = sum((x - mean2) ** 2 for x in rr2) / max(len(rr2) - 1, 1)
    pooled_std = math.sqrt((var1 * (len(rr1) - 1) + var2 * (len(rr2) - 1)) /
                           max(len(rr1) + len(rr2) - 2, 1))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    # Bootstrap 95% CI
    boot_deltas = []
    random.seed(42)
    for _ in range(5000):
        b1 = [random.choice(rr1) for _ in range(len(rr1))]
        b2 = [random.choice(rr2) for _ in range(len(rr2))]
        boot_deltas.append(sum(b1) / len(b1) - sum(b2) / len(b2))
    boot_deltas.sort()
    ci_lo = boot_deltas[int(0.025 * len(boot_deltas))]
    ci_hi = boot_deltas[int(0.975 * len(boot_deltas))]

    print(f"\n  Observed MRR delta (1-field - 2-field): {observed_delta:+.4f}")
    print(f"  Permutation p-value: {p_value:.4f} (N={N_PERM})")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  95% Bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    gate_pass = p_value < 0.05 and observed_delta > 0.05
    if gate_pass:
        print(f"\n  ** GATE PASSED: significant AND effect (p={p_value:.4f}, "
              f"delta={observed_delta:+.4f}) **")
    else:
        print(f"\n  GATE FAILED: AND effect is weak or not significant.")
        print(f"  Recommendation: stop here, AND is not a structural problem.")

    # Stratified comparison
    print(f"\n  --- Stratified Comparison (within category × entity_type) ---")
    strata = defaultdict(lambda: {"1f": [], "2f": []})
    for r in one_field + two_field:
        primary_type = r["gold_types"][0] if r["gold_types"] else "?"
        key = (r["category"], primary_type)
        if r["field_count"] == 1:
            strata[key]["1f"].append(r)
        else:
            strata[key]["2f"].append(r)

    MIN_PER = 5
    deltas = []
    for (cat, etype), groups in sorted(strata.items()):
        g1, g2 = groups["1f"], groups["2f"]
        if len(g1) < MIN_PER or len(g2) < MIN_PER:
            continue
        m1 = sum(r["rr"] for r in g1) / len(g1)
        m2 = sum(r["rr"] for r in g2) / len(g2)
        deltas.append((m1 - m2, len(g2), cat, etype, len(g1), len(g2), m1, m2))

    if deltas:
        tot_w = sum(w for _, w, *_ in deltas)
        weighted = sum(d * w for d, w, *_ in deltas) / tot_w
        print(f"\n  Strata with >= {MIN_PER} per group: {len(deltas)}")
        print(f"  Weighted-average MRR delta (1f - 2f): {weighted:+.4f}")
        for d, w, cat, etype, n1s, n2s, m1s, m2s in sorted(deltas, key=lambda x: -x[0])[:10]:
            print(f"    {cat:<20} {etype:<16} n1={n1s:>3} n2={n2s:>3} "
                  f"MRR1={m1s:.3f} MRR2={m2s:.3f} d={d:+.3f}")

    stats = {
        "observed_delta": observed_delta,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "ci": (ci_lo, ci_hi),
        "gate_pass": gate_pass,
    }
    return buckets, stats


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: Softmax Bottleneck + Additive Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def stage2(rows_2field, all_retrieved):
    print("\n" + "=" * 70)
    print("STAGE 2: Softmax Bottleneck & Additive Aggregation Analysis")
    print("=" * 70)

    # ── 2a: Softmax weight analysis ──────────────────────────────────────────
    print("\n  [2a] Loading checkpoint for softmax weight extraction...")
    try:
        import torch
        from mfar.modeling.util import prepare_model
        from mfar.data.schema import resolve_fields
    except ImportError as e:
        print(f"  [SKIP] Cannot import mFAR: {e}")
        return None

    best_txt = f"{CKPT_DIR}/best.txt"
    if not os.path.exists(best_txt):
        print(f"  [SKIP] No checkpoint at {best_txt}")
        return None

    with open(best_txt) as f:
        ckpt_suffix = f.read().strip().split("/")[-1]
    ckpt_path = f"{CKPT_DIR}/{ckpt_suffix}"

    field_info = resolve_fields(
        "all_dense,all_sparse,single_dense,single_sparse", "prime")
    field_names = list(field_info.keys())

    # Build field detector key → field_info index mapping
    # detector "target" → "target_dense" idx and "target_sparse" idx
    detector_to_indices = {}
    for det_key, bm25_name in FIELD_TO_BM25_DIR.items():
        dense_idx = next((i for i, k in enumerate(field_names)
                          if k == f"{bm25_name}_dense"), None)
        sparse_idx = next((i for i, k in enumerate(field_names)
                           if k == f"{bm25_name}_sparse"), None)
        detector_to_indices[det_key] = (dense_idx, sparse_idx)

    tokenizer, encoder, _ = prepare_model(
        "facebook/contriever-msmarco", normalize=False, with_decoder=False)

    import torch
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    w_key = next((k for k in state_dict
                  if "mixture_of_fields_layer.weight" in k), None)
    W = state_dict[w_key]

    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k.replace("encoder.", "", 1)] = v
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()

    print(f"  Checkpoint loaded. Analyzing {len(rows_2field)} 2-field queries...")

    bottleneck_data = []
    for r in rows_2field:
        fields = sorted(r["detected_fields"])
        if len(fields) != 2:
            continue

        tokens = tokenizer(r["query"], max_length=64, truncation=True,
                           padding=True, return_tensors="pt")
        with torch.no_grad():
            q_emb = encoder(tokens)["sentence_embedding"]
            raw = q_emb @ W
            weights = torch.softmax(raw, dim=1).squeeze(0)

        # Get weight for each detected field (use dense index)
        w_A = 0.0
        w_B = 0.0
        idx_A = detector_to_indices.get(fields[0], (None, None))[0]
        idx_B = detector_to_indices.get(fields[1], (None, None))[0]

        if idx_A is not None:
            w_A = weights[idx_A].item()
        if idx_B is not None:
            w_B = weights[idx_B].item()

        bottleneck = min(w_A, w_B)
        entropy = -sum(w * math.log(max(w, 1e-10))
                       for w in weights.tolist())

        bottleneck_data.append({
            "qid": r["qid"],
            "fields": fields,
            "w_A": w_A, "w_B": w_B,
            "bottleneck": bottleneck,
            "entropy": entropy,
            "rr": r["rr"],
            "miss": r["miss"],
            "first_rel": r["first_rel"],
            "top1_wrong": r["top1_wrong"],
            "query": r["query"],
        })

    if not bottleneck_data:
        print("  No 2-field queries with valid field indices.")
        return None

    # Bottleneck stats
    bn_values = [d["bottleneck"] for d in bottleneck_data]
    rr_values = [d["rr"] for d in bottleneck_data]
    low_bn = [d for d in bottleneck_data if d["bottleneck"] < 0.01]
    high_bn = [d for d in bottleneck_data if d["bottleneck"] >= 0.01]

    print(f"\n  Bottleneck weight (min(α_A, α_B)) distribution:")
    print(f"    Mean: {sum(bn_values)/len(bn_values):.4f}")
    print(f"    < 0.01 (effectively single-field): {len(low_bn)} "
          f"({100*len(low_bn)/len(bottleneck_data):.1f}%)")
    print(f"    >= 0.01 (distributed): {len(high_bn)} "
          f"({100*len(high_bn)/len(bottleneck_data):.1f}%)")

    m_low = group_metrics([{"rr": d["rr"], "hit1": 1 if d["rr"]==1 else 0,
                            "hit5": 1 if d["rr"]>=0.2 else 0,
                            "miss": d["miss"], "ndcg10": d["rr"]}
                           for d in low_bn])
    m_high = group_metrics([{"rr": d["rr"], "hit1": 1 if d["rr"]==1 else 0,
                             "hit5": 1 if d["rr"]>=0.2 else 0,
                             "miss": d["miss"], "ndcg10": d["rr"]}
                            for d in high_bn])

    print(f"\n  Performance by bottleneck weight:")
    print(f"    Low  (< 0.01): n={m_low['n']}, MRR={m_low['mrr']:.3f}, "
          f"Miss={m_low['miss_pct']:.1f}%")
    print(f"    High (>= 0.01): n={m_high['n']}, MRR={m_high['mrr']:.3f}, "
          f"Miss={m_high['miss_pct']:.1f}%")

    # Entropy tercile analysis (within 2-field only)
    bottleneck_data.sort(key=lambda d: d["entropy"])
    n = len(bottleneck_data)
    t1 = bottleneck_data[:n // 3]
    t2 = bottleneck_data[n // 3:2 * n // 3]
    t3 = bottleneck_data[2 * n // 3:]

    print(f"\n  Entropy tercile analysis (within 2-field queries):")
    for label, tercile in [("Low entropy", t1), ("Mid entropy", t2),
                            ("High entropy", t3)]:
        mrr = sum(d["rr"] for d in tercile) / max(len(tercile), 1)
        miss = 100 * sum(d["miss"] for d in tercile) / max(len(tercile), 1)
        ent = sum(d["entropy"] for d in tercile) / max(len(tercile), 1)
        print(f"    {label:<14} n={len(tercile):>4}  mean_H={ent:.2f}  "
              f"MRR={mrr:.3f}  Miss={miss:.1f}%")

    # ── 2b: Additive aggregation analysis ────────────────────────────────────
    print(f"\n  [2b] Additive aggregation: per-field BM25 scores for wrong docs...")

    try:
        from mfar.data.index import BM25sSparseIndex
    except ImportError:
        print("  [SKIP] Cannot import BM25sSparseIndex")
        return bottleneck_data

    # Load needed BM25 indices
    needed_fields = set()
    for d in bottleneck_data:
        if d["first_rel"] > 10 or d["miss"]:
            needed_fields.update(d["fields"])

    bm25_indices = {}
    for det_key in needed_fields:
        bm25_name = FIELD_TO_BM25_DIR.get(det_key)
        if bm25_name is None:
            continue
        idx_path = f"{DATA_DIR}/{bm25_name}_sparse_sparse_index"
        if os.path.exists(idx_path):
            try:
                bm25_indices[det_key] = BM25sSparseIndex.load(idx_path)
            except Exception as e:
                print(f"    [WARN] Failed to load {idx_path}: {e}")

    print(f"  Loaded {len(bm25_indices)} per-field BM25 indices")

    # For failures (rank > 10 or miss), score top-1 wrong doc on both fields
    additive_failures = []
    for d in bottleneck_data:
        if d["first_rel"] <= 10 and not d["miss"]:
            continue
        if d["top1_wrong"] is None:
            continue

        field_A, field_B = d["fields"]
        score_A, score_B = 0.0, 0.0

        if field_A in bm25_indices:
            try:
                s = bm25_indices[field_A].score(d["query"], [d["top1_wrong"]])
                score_A = float(s[0])
            except Exception:
                pass
        if field_B in bm25_indices:
            try:
                s = bm25_indices[field_B].score(d["query"], [d["top1_wrong"]])
                score_B = float(s[0])
            except Exception:
                pass

        min_score = min(score_A, score_B)
        additive_failures.append({
            **d,
            "wrong_score_A": score_A,
            "wrong_score_B": score_B,
            "wrong_min_score": min_score,
        })

    if additive_failures:
        one_field_match = sum(1 for f in additive_failures
                              if f["wrong_min_score"] < 0.1
                              and max(f["wrong_score_A"], f["wrong_score_B"]) > 0.1)
        both_match = sum(1 for f in additive_failures
                         if f["wrong_min_score"] >= 0.1)
        neither = sum(1 for f in additive_failures
                      if max(f["wrong_score_A"], f["wrong_score_B"]) < 0.1)

        print(f"\n  Additive failure analysis ({len(additive_failures)} failures):")
        print(f"    Wrong doc matches ONLY 1 field (additive exploit): "
              f"{one_field_match} ({100*one_field_match/len(additive_failures):.1f}%)")
        print(f"    Wrong doc matches BOTH fields: "
              f"{both_match} ({100*both_match/len(additive_failures):.1f}%)")
        print(f"    Wrong doc matches NEITHER: "
              f"{neither} ({100*neither/len(additive_failures):.1f}%)")
    else:
        print("  No failures with wrong doc to analyze.")

    return bottleneck_data


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: Oracle — Field Non-Empty Check
# ══════════════════════════════════════════════════════════════════════════════

def stage3(rows_2field, corpus, bottleneck_data=None):
    print("\n" + "=" * 70)
    print("STAGE 3: Oracle — Field Non-Empty Check (failures vs successes)")
    print("=" * 70)

    failures = [r for r in rows_2field if r["miss"] or r["first_rel"] >= 20]
    successes = [r for r in rows_2field if not r["miss"] and r["first_rel"] < 5]

    print(f"\n  2-field failures (miss or rank > 20): {len(failures)}")
    print(f"  2-field successes (rank <= 5):        {len(successes)}")

    for label, group in [("Failures", failures), ("Successes", successes)]:
        if not group:
            continue

        # Identify the weaker field via bottleneck_data if available
        # Otherwise just check both fields
        non_empty_but_ignored = 0
        empty_field = 0
        total_checked = 0

        for r in group:
            fields = sorted(r["detected_fields"])
            if len(fields) != 2:
                continue

            # Determine which field is "weaker" (if we have bottleneck data)
            weaker = fields[1]  # default: second alphabetically
            if bottleneck_data:
                bd = next((d for d in bottleneck_data if d["qid"] == r["qid"]), None)
                if bd and bd["w_A"] < bd["w_B"]:
                    weaker = fields[0]

            # Map detector name to corpus field name
            corpus_field = FIELD_TO_BM25_DIR.get(weaker, weaker)

            # Check gold doc
            for gid in r["gold_ids"]:
                if gid not in corpus:
                    continue
                doc_fields = corpus[gid].get("fields", set())
                if corpus_field in doc_fields:
                    non_empty_but_ignored += 1
                else:
                    empty_field += 1
                total_checked += 1
                break

        if total_checked > 0:
            ne_pct = 100 * non_empty_but_ignored / total_checked
            em_pct = 100 * empty_field / total_checked
            print(f"\n  {label} (n={total_checked}):")
            print(f"    Weaker field non-empty in gold doc (signal exists): "
                  f"{non_empty_but_ignored} ({ne_pct:.1f}%)")
            print(f"    Weaker field empty in gold doc (KG gap):            "
                  f"{empty_field} ({em_pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  OR Check
# ══════════════════════════════════════════════════════════════════════════════

def or_check(rows):
    print("\n" + "=" * 70)
    print("OR Check")
    print("=" * 70)

    or_queries = [r for r in rows if r["has_or"]]
    no_or = [r for r in rows if not r["has_or"]]

    print(f"\n  Queries with explicit 'or': {len(or_queries)} "
          f"({100*len(or_queries)/len(rows):.1f}%)")
    print(f"  Queries without 'or':       {len(no_or)}")

    pm("With OR", group_metrics(or_queries))
    pm("Without OR", group_metrics(no_or))

    if len(or_queries) < 30:
        print("\n  Insufficient data for OR analysis (< 30 queries).")


# ══════════════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════════════

def save_report(split, rows, buckets, stats, bottleneck_data=None):
    m = {k: group_metrics(v) for k, v in buckets.items()}

    report = f"# AND/OR Ablation Report — PRIME {split}\n\n"
    report += "## Field Count Distribution\n\n"
    report += "| Field Count | n | MRR | Hit@1 | Miss% | NDCG@10 |\n"
    report += "|-------------|---|-----|-------|-------|---------|\n"
    for label in [0, 1, 2, "3+"]:
        d = m[label]
        report += (f"| {label} | {d['n']} | {d['mrr']:.3f} | "
                   f"{d['hit1']:.1f}% | {d['miss_pct']:.1f}% | "
                   f"{d['ndcg10']:.3f} |\n")

    if stats:
        report += f"\n## Statistical Test: 1-field vs 2-field\n\n"
        report += f"- Observed MRR delta: {stats['observed_delta']:+.4f}\n"
        report += f"- Permutation p-value: {stats['p_value']:.4f}\n"
        report += f"- Cohen's d: {stats['cohens_d']:.3f}\n"
        report += f"- 95% Bootstrap CI: [{stats['ci'][0]:+.4f}, {stats['ci'][1]:+.4f}]\n"
        report += f"- Gate: {'PASSED' if stats['gate_pass'] else 'FAILED'}\n"

    if bottleneck_data:
        low_bn = [d for d in bottleneck_data if d["bottleneck"] < 0.01]
        high_bn = [d for d in bottleneck_data if d["bottleneck"] >= 0.01]
        report += f"\n## Softmax Bottleneck (2-field queries)\n\n"
        report += f"- Low bottleneck (< 0.01, single-field): {len(low_bn)}\n"
        report += f"- High bottleneck (>= 0.01, distributed): {len(high_bn)}\n"

    report += f"\n---\n*Analysis: {len(rows)} {split} queries*\n"

    path = f"{OUT_DIR}/REPORT_{split}.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  Saved: {path}")


def generate_plots(split, buckets):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    labels = ["0-field", "1-field", "2-field", "3+-field"]
    keys = [0, 1, 2, "3+"]
    metrics = [group_metrics(buckets[k]) for k in keys]
    colors = ["#95a5a6", "#2ecc71", "#e74c3c", "#9b59b6"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"AND Analysis — PRIME {split}", fontsize=13, fontweight="bold")

    for ax, metric_key, ylabel in zip(
        axes, ["mrr", "hit1", "miss_pct"], ["MRR", "Hit@1 (%)", "Miss Rate (%)"]
    ):
        vals = [m[metric_key] for m in metrics]
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(vals) * 1.3, 0.01))
        for bar, v in zip(bars, vals):
            fmt = f"{v:.3f}" if metric_key == "mrr" else f"{v:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = f"{OUT_DIR}/mrr_by_field_count_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["val", "test"]

    print("AND/OR Constraint Ablation Study — PRIME")
    print("=" * 70)

    corpus = load_corpus_full(DATA_DIR)

    for split in splits:
        print(f"\n{'=' * 70}")
        print(f"  SPLIT: {split}")
        print(f"{'=' * 70}")

        if split not in SPLIT_QRES:
            print(f"  [SKIP] No qres for '{split}'")
            continue

        queries = load_queries(DATA_DIR, split)
        qrels = load_qrels(DATA_DIR, split)
        retrieved = load_retrieved(SPLIT_QRES[split])

        rows = compute_all_rows(queries, qrels, retrieved, corpus)
        print(f"  Computed metrics for {len(rows):,} queries")

        # Stage 1
        buckets, stats = stage1(rows)

        gate_pass = stats is not None and stats["gate_pass"]

        # Stage 2 (conditional)
        bottleneck_data = None
        if gate_pass:
            two_field = buckets[2]
            bottleneck_data = stage2(two_field, retrieved)
        else:
            print("\n  [SKIP Stage 2] Gate not passed.")

        # Stage 3 (conditional)
        if gate_pass and bottleneck_data:
            stage3(buckets[2], corpus, bottleneck_data)
        else:
            print("\n  [SKIP Stage 3]")

        # OR check (always, quick)
        or_check(rows)

        # Save
        save_report(split, rows, buckets, stats, bottleneck_data)
        generate_plots(split, buckets)

    print(f"\n{'=' * 70}")
    print("Done. Reports in output/failure_analysis/and_or/")


if __name__ == "__main__":
    main()
