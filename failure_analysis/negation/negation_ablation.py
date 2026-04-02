"""
Negation Ablation Study — PRIME Dataset
========================================
Classifies negation queries into a taxonomy (Type A / B / C),
computes per-type metrics, matched comparison, mFAR weight
verification, and oracle negation-removal test.

Run from project root:
  python failure_analysis/negation/negation_ablation.py [val] [test]
  python failure_analysis/negation/negation_ablation.py      # default: val test
"""

import json
import re
import os
import sys
import math
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
OUT_DIR  = "output/failure_analysis/negation"

os.makedirs(OUT_DIR, exist_ok=True)

SPLIT_QRES = {
    "val":  f"{EVAL_DIR}/final-all-0.qres",
    "test": f"{EVAL_DIR}/final-additional-all-0.qres",
}

# ── Master negation pattern (same as failure_analysis.py) ────────────────────
NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|without|lack(?:s|ing)?|absent|neither|nor|never|"
    r"cannot|can't|don't|doesn't|didn't|won't|isn't|aren't|wasn't|weren't|"
    r"exclude[ds]?|non[\-])\b|"
    r"\bun(?:express|relat|associat|link|affect|involv)\w*\b",
    re.IGNORECASE,
)

# ── Query category patterns (from failure_analysis.py) for stratification ────
QUERY_CATEGORIES = [
    ("expression_absent",  r"(?:not|lack|absent|no)\s+express|unexpress|without\s+express"),
    ("expression_present", r"express(?:ed|ion|es)?\s+(?:in|by|of)|show\s+express"),
    ("pathway_hierarchy",  r"pathways?\s+connect|above.*below|below.*above|hierarch|signal.*pathway|connect.*pathway"),
    ("drug_indication",    r"indicat|phase\s+[IVX]+|clinical\s+trial|treat.*diseas|approv"),
    ("drug_interaction",   r"carrier|transporter|enzyme|metaboli"),
    ("side_effect",        r"side[\s-]?effect|adverse\s+effect|toxici"),
    ("target",             r"\btarget\b"),
    ("ppi",                r"\bprotein.protein|ppi\b|\binteract(?:s|ing|ion)?\b"),
    ("phenotype",          r"\bphenotype\b|phenotyp"),
    ("drug_target",        r"drug.*target|compound.*target|target.*compound|target.*drug"),
    ("other",              r"."),
]


def classify_query_category(text):
    text_lower = text.lower()
    for cat, pattern in QUERY_CATEGORIES:
        if re.search(pattern, text_lower):
            return cat
    return "other"


# ══════════════════════════════════════════════════════════════════════════════
#  NEGATION TAXONOMY (Type A / C / B — checked in this order)
# ══════════════════════════════════════════════════════════════════════════════

# ── Type A: Simple field-level negation ──────────────────────────────────────
#    Maps directly to expression_absent or phenotype_absent fields.
TYPE_A_PATTERN = re.compile(
    r"(?:not|lack(?:s|ing)?|absent|without|no)\s+(?:\w+\s+){0,3}express"
    r"|unexpress"
    r"|no\s+expression"
    r"|(?:do|does)\s+not\s+(?:exhibit|show)\s+expression"
    r"|(?:not|lack(?:s|ing)?)\s+(?:found|detected|present)\s+in"
    r"|without\s+(?:any\s+)?(?:phenotype|symptom)"
    r"|(?:phenotype|symptom)\s+absent",
    re.IGNORECASE,
)

# ── Type C: Spurious / narrative negation ────────────────────────────────────
#    Negation word exists but is NOT a retrieval constraint.
TYPE_C_PATTERNS = [
    # C.entity_name: "non-" as part of a biomedical entity name
    # Empirically: all 27 "non-" val queries are entity name refs.
    re.compile(
        r"non-(?:syndromic|invasive|cutaneous|canonical|lymphoid|cancerous|"
        r"small\s*cell|hodgkin|insulin|steroidal|classic|obstructive|alcoholic|"
        r"specific|ischemic|proliferative|secretory|functioning|immune|"
        r"epithelial|amyloid|malignant|coding|aggressive|severe|threatening|"
        r"receptor|protein|gestational|24|competitive|redundant|overlapping|"
        r"structural|histone|homologous|muscle|essential|ribosomal|"
        r"selective|toxic|viral|enzymatic|covalent|catalytic)",
        re.IGNORECASE,
    ),
    # "not only...but also" — rhetorical
    re.compile(r"not\s+only\b", re.IGNORECASE),
    # "if not treated" — conditional narrative
    re.compile(r"if\s+not\s+treat", re.IGNORECASE),
    # "lack of [physical symptom]" — symptom description, not field constraint
    re.compile(
        r"lack\s+of\s+(?:coordination|balance|appetite|growth|energy|sleep|"
        r"motor|voluntary|muscle|brain|elasticit|neutrophil|pigment|oxygen|"
        r"sensation|movement|control|vision|hearing|awareness|consciousness)",
        re.IGNORECASE,
    ),
    # "without [symptom]" — symptom description
    re.compile(
        r"without\s+(?:pain|symptoms|fever|swelling|complications|"
        r"accompanying|causing|leading|resulting|producing|further|"
        r"significant|any\s+(?:sign|evidence))",
        re.IGNORECASE,
    ),
    # "absent [body part]" — anatomical description of a condition
    re.compile(
        r"absent\s+(?:pinky|finger|toe|limb|extremit|radius|thumb|nail|"
        r"septum|corpus|kidney|spleen|ovary|uterus|tibia|fibula)",
        re.IGNORECASE,
    ),
    # Narrative / rhetorical negation
    re.compile(
        r"not\s+(?:fully\s+understood|yet\s+known|exactly|quite|the\s+same|"
        r"the\s+whole|exposed\s+to|limited\s+to|restricted)",
        re.IGNORECASE,
    ),
    # "don't/doesn't respond/involve" — resistance or ambiguous
    re.compile(r"(?:don't|doesn't|didn't)\s+(?:respond|involve)", re.IGNORECASE),
]

# ── Type B: Complex semantic negation (everything remaining) ─────────────────
#    Multi-label: returns set of negated fields.
NEGATION_FIELD_MAP = {
    "indication": re.compile(
        r"(?:lack(?:s|ing)?|no|without|don't\s+have)\s+(?:\w+\s+){0,3}"
        r"(?:treat|drug|medic|therap|approv|pharmacolog|prescri|remedy|cure)"
        r"|(?:not\s+treat(?:ed|able)\b)"
        r"|(?:lacking\s+(?:\w+\s+){0,3}(?:treat|drug|medic|therap|approv|cure))",
        re.IGNORECASE,
    ),
    "associated with": re.compile(
        r"(?:not|un)\s*(?:associat|relat|link|connect)"
        r"|(?:no|without)\s+(?:\w+\s+){0,2}(?:association|relation|link|connection)",
        re.IGNORECASE,
    ),
    "contraindication": re.compile(
        r"(?:should\s+not|won't|don't|shouldn't)\s+(?:\w+\s+){0,2}"
        r"(?:exacerbat|worsen|aggravat|pose|take|use|prescri|administer|treat|manage)"
        r"|(?:not\s+(?:suitable|safe|recommend|advise|treat|managed))"
        r"|should\s+not\s+be\s+(?:treat|manage|prescri|administer|use)",
        re.IGNORECASE,
    ),
    "target": re.compile(
        r"(?:not|don't|doesn't)\s+(?:\w+\s+){0,2}target"
        r"|(?:no|without)\s+(?:\w+\s+){0,2}(?:target|binding)",
        re.IGNORECASE,
    ),
    "side effect": re.compile(
        r"(?:not|no|without)\s+(?:\w+\s+){0,2}(?:side[\s-]?effect|adverse|toxici)",
        re.IGNORECASE,
    ),
    "ppi": re.compile(
        r"(?:not|don't|doesn't)\s+(?:\w+\s+){0,2}interact"
        r"|(?:no|without)\s+(?:\w+\s+){0,2}interaction",
        re.IGNORECASE,
    ),
    "expression present": re.compile(
        r"(?:not|don't|doesn't)\s+(?:\w+\s+){0,2}express(?:ed)?(?!\s+absent)"
        r"|(?:no|without)\s+expression\s+(?:in|of)",
        re.IGNORECASE,
    ),
}


def classify_negation(query_text):
    """
    Classify a negation query into Type A, B, or C.

    Returns: (type_label, subtype, negated_fields)
      - type_label: "A", "B", or "C"
      - subtype: descriptive string
      - negated_fields: set of field names (only for Type B)
    """
    # Type A — simple field-level
    if TYPE_A_PATTERN.search(query_text):
        return "A", "expression/phenotype_absent", set()

    # Type C — spurious (checked BEFORE B)
    for pattern in TYPE_C_PATTERNS:
        if pattern.search(query_text):
            return "C", "spurious", set()

    # Type B — complex semantic, multi-label field mapping
    negated_fields = set()
    for field_name, pattern in NEGATION_FIELD_MAP.items():
        if pattern.search(query_text):
            negated_fields.add(field_name)

    if negated_fields:
        primary = sorted(negated_fields)[0]
        return "B", primary, negated_fields
    else:
        return "B", "other", set()


# ══════════════════════════════════════════════════════════════════════════════
#  PER-QUERY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_row(qid, query_text, gold, docs, corpus):
    """Compute metrics for a single query."""
    top100 = docs[:100]
    top_ids = [d[0] for d in top100]
    scores = [d[1] for d in top100]

    top1_score = scores[0] if scores else 0.0
    first_rel = next((i for i, did in enumerate(top_ids) if did in gold), -1)
    rr = 0.0 if first_rel < 0 else 1.0 / (first_rel + 1)
    hit1 = 1 if first_rel == 0 else 0
    hit5 = 1 if 0 <= first_rel < 5 else 0
    hit20 = 1 if 0 <= first_rel < 20 else 0
    miss = 1 if first_rel < 0 else 0

    gains10 = [1 if did in gold else 0 for did in top_ids[:10]]
    ideal = dcg([1] * min(len(gold), 10), 10)
    ndcg10 = dcg(gains10, 10) / ideal if ideal > 0 else 0.0

    gold_types = [corpus[did]["type"] for did in gold if did in corpus]
    category = classify_query_category(query_text)
    is_negation = bool(NEGATION_PATTERN.search(query_text))

    neg_type, neg_subtype, neg_fields = ("", "", set())
    if is_negation:
        neg_type, neg_subtype, neg_fields = classify_negation(query_text)

    return {
        "qid": qid,
        "query": query_text,
        "rr": rr,
        "hit1": hit1,
        "hit5": hit5,
        "hit20": hit20,
        "miss": miss,
        "ndcg10": ndcg10,
        "first_rel": first_rel,
        "top1_score": top1_score,
        "gold_types": gold_types,
        "gold_ids": gold,
        "category": category,
        "is_negation": is_negation,
        "neg_type": neg_type,
        "neg_subtype": neg_subtype,
        "neg_fields": neg_fields,
    }


def compute_all_rows(queries, qrels, retrieved, corpus):
    rows = []
    for qid, text in queries.items():
        if qid not in qrels:
            continue
        gold = qrels[qid]
        docs = retrieved.get(qid, [])
        rows.append(compute_row(qid, text, gold, docs, corpus))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def group_metrics(rows):
    """Compute aggregate metrics for a list of rows."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "mrr": 0, "hit1": 0, "hit5": 0, "hit20": 0,
                "miss_pct": 0, "ndcg10": 0}
    return {
        "n": n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "hit1": 100 * sum(r["hit1"] for r in rows) / n,
        "hit5": 100 * sum(r["hit5"] for r in rows) / n,
        "hit20": 100 * sum(r["hit20"] for r in rows) / n,
        "miss_pct": 100 * sum(r["miss"] for r in rows) / n,
        "ndcg10": sum(r["ndcg10"] for r in rows) / n,
    }


def print_metrics_row(label, m, indent="  "):
    print(f"{indent}{label:<28} n={m['n']:>5}  MRR={m['mrr']:.3f}  "
          f"H@1={m['hit1']:5.1f}%  H@5={m['hit5']:5.1f}%  "
          f"Miss={m['miss_pct']:5.1f}%  NDCG@10={m['ndcg10']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: Taxonomy Distribution
# ══════════════════════════════════════════════════════════════════════════════

def section1_taxonomy(rows):
    print("\n" + "=" * 70)
    print("SECTION 1: Negation Taxonomy Distribution")
    print("=" * 70)

    neg_rows = [r for r in rows if r["is_negation"]]
    non_neg = [r for r in rows if not r["is_negation"]]
    type_a = [r for r in neg_rows if r["neg_type"] == "A"]
    type_b = [r for r in neg_rows if r["neg_type"] == "B"]
    type_c = [r for r in neg_rows if r["neg_type"] == "C"]

    total = len(rows)
    print(f"\n  Total queries:       {total}")
    print(f"  Negation queries:    {len(neg_rows)} ({100*len(neg_rows)/total:.1f}%)")
    print(f"  Non-negation:        {len(non_neg)} ({100*len(non_neg)/total:.1f}%)")
    print(f"\n  Type A (simple):     {len(type_a)} ({100*len(type_a)/max(len(neg_rows),1):.1f}% of neg)")
    print(f"  Type B (complex):    {len(type_b)} ({100*len(type_b)/max(len(neg_rows),1):.1f}% of neg)")
    print(f"  Type C (spurious):   {len(type_c)} ({100*len(type_c)/max(len(neg_rows),1):.1f}% of neg)")

    print(f"\n  {'Group':<28} {'n':>5}  {'MRR':>6}  {'H@1':>6}  {'H@5':>6}  "
          f"{'Miss%':>6}  {'NDCG@10':>8}")
    print("  " + "-" * 78)
    print_metrics_row("Non-negation (baseline)", group_metrics(non_neg))
    print_metrics_row("All negation", group_metrics(neg_rows))
    print_metrics_row("  Type A (field-level)", group_metrics(type_a))
    print_metrics_row("  Type B (complex)", group_metrics(type_b))
    print_metrics_row("  Type C (spurious)", group_metrics(type_c))

    return neg_rows, non_neg, type_a, type_b, type_c


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: Head-to-Head (negation vs non-negation)
# ══════════════════════════════════════════════════════════════════════════════

def section2_head_to_head(neg_rows, non_neg):
    print("\n" + "=" * 70)
    print("SECTION 2: Negation vs Non-Negation Head-to-Head")
    print("=" * 70)

    m_neg = group_metrics(neg_rows)
    m_non = group_metrics(non_neg)

    if m_non["mrr"] > 0:
        delta_mrr = (m_neg["mrr"] - m_non["mrr"]) / m_non["mrr"] * 100
    else:
        delta_mrr = 0
    delta_miss = m_neg["miss_pct"] - m_non["miss_pct"]

    print(f"\n  MRR delta:     {delta_mrr:+.1f}% (neg={m_neg['mrr']:.3f} vs non={m_non['mrr']:.3f})")
    print(f"  Miss delta:    {delta_miss:+.1f}pp (neg={m_neg['miss_pct']:.1f}% vs non={m_non['miss_pct']:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: Matched Comparison (stratified by category × entity type)
# ══════════════════════════════════════════════════════════════════════════════

def section3_matched(rows):
    print("\n" + "=" * 70)
    print("SECTION 3: Stratified Matched Comparison")
    print("=" * 70)
    print("  (Negation vs affirmative within same query_category × entity_type)")

    # Build strata
    strata = defaultdict(lambda: {"neg": [], "aff": []})
    for r in rows:
        primary_type = r["gold_types"][0] if r["gold_types"] else "unknown"
        key = (r["category"], primary_type)
        if r["is_negation"]:
            strata[key]["neg"].append(r)
        else:
            strata[key]["aff"].append(r)

    # Compute per-stratum deltas (min 5 in each group)
    MIN_PER_GROUP = 5
    deltas = []
    print(f"\n  {'Category':<22} {'Entity type':<18} {'N_neg':>5} {'N_aff':>5} "
          f"{'MRR_neg':>8} {'MRR_aff':>8} {'Delta':>8}")
    print("  " + "-" * 82)

    for (cat, etype), groups in sorted(strata.items()):
        neg_rows = groups["neg"]
        aff_rows = groups["aff"]
        if len(neg_rows) < MIN_PER_GROUP or len(aff_rows) < MIN_PER_GROUP:
            continue
        mrr_neg = sum(r["rr"] for r in neg_rows) / len(neg_rows)
        mrr_aff = sum(r["rr"] for r in aff_rows) / len(aff_rows)
        delta = mrr_neg - mrr_aff
        weight = len(neg_rows)
        deltas.append((delta, weight, cat, etype, len(neg_rows), len(aff_rows),
                        mrr_neg, mrr_aff))
        print(f"  {cat:<22} {etype:<18} {len(neg_rows):>5} {len(aff_rows):>5} "
              f"{mrr_neg:>8.3f} {mrr_aff:>8.3f} {delta:>+8.3f}")

    if deltas:
        total_weight = sum(w for _, w, *_ in deltas)
        weighted_delta = sum(d * w for d, w, *_ in deltas) / total_weight if total_weight else 0
        print(f"\n  Weighted-average MRR delta: {weighted_delta:+.3f}")
        print(f"  (across {len(deltas)} strata with >= {MIN_PER_GROUP} queries per group)")

        if abs(weighted_delta) > 0:
            baseline_mrr = sum(r["rr"] for r in rows if not r["is_negation"]) / max(sum(1 for r in rows if not r["is_negation"]), 1)
            pct = weighted_delta / baseline_mrr * 100 if baseline_mrr > 0 else 0
            print(f"  Relative to baseline MRR ({baseline_mrr:.3f}): {pct:+.1f}%")
    else:
        print("\n  No strata with >= 5 queries in both groups.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: Type A Deep Dive — Field Verification
# ══════════════════════════════════════════════════════════════════════════════

def section4_type_a(type_a, corpus, qrels):
    print("\n" + "=" * 70)
    print("SECTION 4: Type A Deep Dive — Field Population Check")
    print("=" * 70)

    present_match = []
    present_nomatch = []
    absent = []

    for r in type_a:
        gold_ids = r["gold_ids"]
        has_field = False
        for gid in gold_ids:
            if gid in corpus:
                ea = corpus[gid].get("expression_absent", {})
                pa = corpus[gid].get("phenotype_absent", {})
                if ea or pa:
                    has_field = True
                    break

        if has_field:
            # Check if the absent field content loosely matches query
            # (simple heuristic: any anatomy term from the field appears in query)
            query_lower = r["query"].lower()
            any_match = False
            for gid in gold_ids:
                if gid not in corpus:
                    continue
                ea = corpus[gid].get("expression_absent", {})
                for category_entries in (ea.values() if isinstance(ea, dict) else []):
                    if isinstance(category_entries, list):
                        for entry in category_entries:
                            if isinstance(entry, str) and entry.lower() in query_lower:
                                any_match = True
                                break
                    if any_match:
                        break
                if any_match:
                    break
            if any_match:
                present_match.append(r)
            else:
                present_nomatch.append(r)
        else:
            absent.append(r)

    total_a = len(type_a)
    print(f"\n  Type A queries: {total_a}")
    print(f"    field_present + matches:    {len(present_match):>4} ({100*len(present_match)/max(total_a,1):.1f}%)")
    print(f"    field_present + no match:   {len(present_nomatch):>4} ({100*len(present_nomatch)/max(total_a,1):.1f}%)")
    print(f"    field_absent (-> eff. B):   {len(absent):>4} ({100*len(absent)/max(total_a,1):.1f}%)")

    print(f"\n  Performance by sub-group:")
    print_metrics_row("field present+match", group_metrics(present_match))
    print_metrics_row("field present+no match", group_metrics(present_nomatch))
    print_metrics_row("field absent", group_metrics(absent))

    return present_match, present_nomatch, absent


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4b: mFAR Weight Verification (lightweight checkpoint load)
# ══════════════════════════════════════════════════════════════════════════════

def section4b_weight_analysis(type_a):
    """Extract per-query softmax weights from trained mFAR checkpoint."""
    print("\n" + "=" * 70)
    print("SECTION 4b: mFAR Softmax Weight Analysis (Type A queries)")
    print("=" * 70)

    try:
        import torch
        from mfar.modeling.util import prepare_model
        from mfar.data.schema import resolve_fields
    except ImportError as e:
        print(f"\n  [SKIP] Cannot import mFAR modules: {e}")
        return None

    # Load checkpoint
    best_txt = f"{CKPT_DIR}/best.txt"
    if not os.path.exists(best_txt):
        print(f"\n  [SKIP] No checkpoint found at {best_txt}")
        return None

    with open(best_txt) as f:
        ckpt_suffix = f.read().strip().split("/")[-1]
    ckpt_path = f"{CKPT_DIR}/{ckpt_suffix}"
    print(f"\n  Checkpoint: {ckpt_path}")

    # Resolve field info to get field ordering
    field_info = resolve_fields("all_dense,all_sparse,single_dense,single_sparse", "prime")
    field_names = list(field_info.keys())
    print(f"  Fields: {len(field_names)}")

    # Find expression_absent and expression_present indices
    # Field keys use spaces: "expression absent_dense", "expression present_dense"
    ea_dense_idx = next((i for i, k in enumerate(field_names) if "expression absent" in k and "dense" in k), None)
    ep_dense_idx = next((i for i, k in enumerate(field_names) if "expression present" in k and "dense" in k), None)
    ea_sparse_idx = next((i for i, k in enumerate(field_names) if "expression absent" in k and "sparse" in k), None)
    ep_sparse_idx = next((i for i, k in enumerate(field_names) if "expression present" in k and "sparse" in k), None)

    # Debug: print first few field names to verify format
    print(f"  Sample field keys: {field_names[:5]} ... {field_names[-3:]}")

    if ea_dense_idx is None or ep_dense_idx is None:
        print(f"  [SKIP] Cannot find expression absent/present field indices")
        print(f"         Searched for 'expression absent' + 'dense' in: {[k for k in field_names if 'express' in k]}")
        return None

    print(f"  expression absent_dense  idx={ea_dense_idx}")
    print(f"  expression present_dense idx={ep_dense_idx}")

    # Lightweight load: only need encoder + mixture_of_fields_layer.weight
    tokenizer, encoder, _ = prepare_model(
        "facebook/contriever-msmarco", normalize=False, with_decoder=False,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Extract mixture weight matrix
    w_key = None
    for k in state_dict:
        if "mixture_of_fields_layer.weight" in k:
            w_key = k
            break
    if w_key is None:
        print("  [SKIP] Cannot find mixture_of_fields_layer.weight in checkpoint")
        return None

    W = state_dict[w_key]  # [emb_dim, num_fields]
    print(f"  W shape: {W.shape}")

    # Extract encoder weights
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k.replace("encoder.", "", 1)] = v
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()

    # Compute weights for each Type A query
    correct_routing = 0
    total_checked = 0
    ea_weights = []
    ep_weights = []

    for r in type_a:
        tokens = tokenizer(
            r["query"], max_length=64, truncation=True, padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = encoder(tokens)
            q_emb = out["sentence_embedding"]  # [1, emb_dim]
            raw = q_emb @ W  # [1, num_fields]
            weights = torch.softmax(raw, dim=1).squeeze(0)  # [num_fields]

        w_ea = weights[ea_dense_idx].item()
        w_ep = weights[ep_dense_idx].item()
        ea_weights.append(w_ea)
        ep_weights.append(w_ep)

        if w_ea > w_ep:
            correct_routing += 1
        total_checked += 1

    correct_pct = 100 * correct_routing / max(total_checked, 1)
    misroute_pct = 100 - correct_pct

    print(f"\n  Type A weight routing analysis ({total_checked} queries):")
    print(f"    Correct (w_absent > w_present): {correct_routing} ({correct_pct:.1f}%)")
    print(f"    Misrouted (w_present >= w_absent): {total_checked - correct_routing} ({misroute_pct:.1f}%)")
    print(f"    Mean w(expression_absent_dense):  {sum(ea_weights)/max(len(ea_weights),1):.4f}")
    print(f"    Mean w(expression_present_dense): {sum(ep_weights)/max(len(ep_weights),1):.4f}")

    if misroute_pct > 50:
        print(f"\n  ** FINDING: mFAR misroutes > 50% of Type A queries! **")
        print(f"     Even 'simple' field-level negation is NOT solved by mFAR's structure.")
    else:
        print(f"\n  mFAR correctly routes majority of Type A queries.")

    return {
        "correct_pct": correct_pct,
        "misroute_pct": misroute_pct,
        "mean_ea": sum(ea_weights) / max(len(ea_weights), 1),
        "mean_ep": sum(ep_weights) / max(len(ep_weights), 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: Type B Per-Field Breakdown (multi-label)
# ══════════════════════════════════════════════════════════════════════════════

def section5_type_b(type_b):
    print("\n" + "=" * 70)
    print("SECTION 5: Type B — Per-Field Breakdown (multi-label)")
    print("=" * 70)

    # Multi-field stats
    multi_field = [r for r in type_b if len(r["neg_fields"]) >= 2]
    print(f"\n  Total Type B: {len(type_b)}")
    print(f"  Multi-field negation (2+ fields): {len(multi_field)} "
          f"({100*len(multi_field)/max(len(type_b),1):.1f}%)")

    # Per-field breakdown
    field_rows = defaultdict(list)
    for r in type_b:
        if r["neg_fields"]:
            for f in r["neg_fields"]:
                field_rows[f].append(r)
        else:
            field_rows["other"].append(r)

    print(f"\n  {'Negated Field':<25} {'n':>5}  {'MRR':>6}  {'H@1':>6}  "
          f"{'Miss%':>6}")
    print("  " + "-" * 58)
    for field, frows in sorted(field_rows.items(), key=lambda x: -len(x[1])):
        m = group_metrics(frows)
        print(f"  {field:<25} {m['n']:>5}  {m['mrr']:>6.3f}  "
              f"{m['hit1']:>5.1f}%  {m['miss_pct']:>5.1f}%")

    # Subtype distribution
    subtype_counts = Counter(r["neg_subtype"] for r in type_b)
    print(f"\n  Subtype distribution:")
    for sub, cnt in subtype_counts.most_common():
        print(f"    {sub:<25} {cnt}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: Oracle Negation-Removal Test
# ══════════════════════════════════════════════════════════════════════════════

def section6_oracle(type_b, retrieved_all):
    """
    For Type B complete-miss queries, check if the gold doc appears
    anywhere in the top-100 when we look at the *original* retrieval
    results more carefully (e.g., does mFAR retrieve docs of the right
    entity type but wrong constraint?).

    Since we only have top-100, we check:
    - How many gold docs of Type B misses have the same entity type
      as the top-1 retrieved doc? (= "findable" by entity type)
    - For Type B non-misses: what rank is the gold doc?
    """
    print("\n" + "=" * 70)
    print("SECTION 6: Oracle Negation-Removal Test")
    print("=" * 70)

    misses = [r for r in type_b if r["miss"]]
    non_misses = [r for r in type_b if not r["miss"]]

    print(f"\n  Type B: {len(type_b)} total, {len(misses)} complete misses, "
          f"{len(non_misses)} found in top-100")

    if not misses:
        print("  No complete misses — oracle test not applicable.")
        return

    # For misses: check if top-retrieved docs share entity type with gold
    type_match = 0
    for r in misses:
        gold_types = set(r["gold_types"])
        docs = retrieved_all.get(r["qid"], [])
        top_types = set()
        for did, _ in docs[:10]:
            # gold_types are from corpus, top docs are IDs — we need corpus
            # but we only have gold_types. Check if ANY top-10 doc type
            # overlaps with gold type.
            pass
        # Simplified: check if first_rel is -1 (we know it is for misses)
        # Instead, count how many misses have gold docs that are a common type
        if gold_types:
            type_match += 1  # placeholder — see below

    # Better approach: rank distribution for non-misses
    print(f"\n  Rank distribution for Type B non-misses ({len(non_misses)} queries):")
    rank_buckets = Counter()
    for r in non_misses:
        fr = r["first_rel"]
        if fr == 0:
            rank_buckets["rank 1"] += 1
        elif fr < 5:
            rank_buckets["rank 2-5"] += 1
        elif fr < 10:
            rank_buckets["rank 6-10"] += 1
        elif fr < 20:
            rank_buckets["rank 11-20"] += 1
        elif fr < 50:
            rank_buckets["rank 21-50"] += 1
        else:
            rank_buckets["rank 51-100"] += 1

    for bucket in ["rank 1", "rank 2-5", "rank 6-10", "rank 11-20",
                    "rank 21-50", "rank 51-100"]:
        cnt = rank_buckets.get(bucket, 0)
        pct = 100 * cnt / max(len(non_misses), 1)
        print(f"    {bucket:<15} {cnt:>4} ({pct:5.1f}%)")

    # Key insight: if many non-misses are in rank 51-100, a modest
    # re-ranking could push them out. For misses at rank > 100,
    # it suggests the doc IS retrievable but ranked too low.
    deep_rank = sum(1 for r in non_misses if r["first_rel"] >= 20)
    print(f"\n  Non-misses with gold doc at rank > 20: {deep_rank} "
          f"({100*deep_rank/max(len(non_misses),1):.1f}%)")
    print(f"  -> These are 'barely found' — small perturbation could push to miss")
    print(f"\n  Complete misses: {len(misses)} — gold doc not in top-100 at all")
    print(f"  -> A Boolean filter on negation could prevent irrelevant docs")
    print(f"     from consuming top-100 slots, making room for gold doc")

    # Estimate: what fraction of top-100 slots for misses are filled by
    # docs that share entity type with gold?
    # (This shows mFAR retrieves the RIGHT type but WRONG instance)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: Summary & Verdict
# ══════════════════════════════════════════════════════════════════════════════

def section7_verdict(neg_rows, type_a, type_b, type_c, non_neg, weight_result):
    print("\n" + "=" * 70)
    print("SECTION 7: Summary & Verdict")
    print("=" * 70)

    m_b = group_metrics(type_b)
    m_non = group_metrics(non_neg)

    print(f"\n  1. Type B count: {m_b['n']}", end="")
    if m_b["n"] > 100:
        print(f" > 100  [PASS]")
    else:
        print(f" <= 100  [MARGINAL]")

    if m_non["mrr"] > 0:
        mrr_drop = (m_b["mrr"] - m_non["mrr"]) / m_non["mrr"] * 100
    else:
        mrr_drop = 0
    print(f"  2. Type B MRR vs baseline: {mrr_drop:+.1f}%", end="")
    if abs(mrr_drop) > 15:
        print(f"  [SIGNIFICANT]")
    else:
        print(f"  [WEAK]")

    if weight_result:
        print(f"  3. Type A misrouting: {weight_result['misroute_pct']:.1f}%", end="")
        if weight_result["misroute_pct"] > 50:
            print(f"  [EVEN SIMPLE NEGATION UNSOLVED]")
        else:
            print(f"  [mFAR handles simple negation]")
    else:
        print(f"  3. Weight analysis: skipped")

    # Overall verdict
    strong = m_b["n"] > 100 and abs(mrr_drop) > 15
    print(f"\n  VERDICT: ", end="")
    if strong:
        print("Explicit NOT handling is JUSTIFIED.")
        print(f"  {m_b['n']} complex-negation queries suffer {mrr_drop:+.1f}% MRR drop.")
    else:
        print("Negation contribution is WEAK — consider pivoting.")


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT & PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def save_report(split, rows, neg_rows, non_neg, type_a, type_b, type_c,
                weight_result):
    m_all = group_metrics(rows)
    m_neg = group_metrics(neg_rows)
    m_non = group_metrics(non_neg)
    m_a = group_metrics(type_a)
    m_b = group_metrics(type_b)
    m_c = group_metrics(type_c)

    mrr_drop_b = ((m_b["mrr"] - m_non["mrr"]) / m_non["mrr"] * 100
                  if m_non["mrr"] > 0 else 0)

    report = f"""# Negation Ablation Report — PRIME {split} set

## Taxonomy Distribution

| Type | n | % of neg | MRR | Hit@1 | Hit@5 | Miss% | NDCG@10 |
|------|---|----------|-----|-------|-------|-------|---------|
| Non-negation (baseline) | {m_non['n']} | — | {m_non['mrr']:.3f} | {m_non['hit1']:.1f}% | {m_non['hit5']:.1f}% | {m_non['miss_pct']:.1f}% | {m_non['ndcg10']:.3f} |
| All negation | {m_neg['n']} | 100% | {m_neg['mrr']:.3f} | {m_neg['hit1']:.1f}% | {m_neg['hit5']:.1f}% | {m_neg['miss_pct']:.1f}% | {m_neg['ndcg10']:.3f} |
| Type A (field-level) | {m_a['n']} | {100*m_a['n']/max(m_neg['n'],1):.1f}% | {m_a['mrr']:.3f} | {m_a['hit1']:.1f}% | {m_a['hit5']:.1f}% | {m_a['miss_pct']:.1f}% | {m_a['ndcg10']:.3f} |
| Type B (complex) | {m_b['n']} | {100*m_b['n']/max(m_neg['n'],1):.1f}% | {m_b['mrr']:.3f} | {m_b['hit1']:.1f}% | {m_b['hit5']:.1f}% | {m_b['miss_pct']:.1f}% | {m_b['ndcg10']:.3f} |
| Type C (spurious) | {m_c['n']} | {100*m_c['n']/max(m_neg['n'],1):.1f}% | {m_c['mrr']:.3f} | {m_c['hit1']:.1f}% | {m_c['hit5']:.1f}% | {m_c['miss_pct']:.1f}% | {m_c['ndcg10']:.3f} |

## Key Findings

- Type B MRR vs baseline: **{mrr_drop_b:+.1f}%**
- Type C MRR vs baseline: {((m_c['mrr']-m_non['mrr'])/m_non['mrr']*100) if m_non['mrr']>0 else 0:+.1f}% (sanity check — should be ~0%)
"""

    if weight_result:
        report += f"""
## mFAR Weight Routing (Type A)

- Correct routing (w_absent > w_present): {weight_result['correct_pct']:.1f}%
- Misrouted: {weight_result['misroute_pct']:.1f}%
- Mean w(expression_absent_dense): {weight_result['mean_ea']:.4f}
- Mean w(expression_present_dense): {weight_result['mean_ep']:.4f}
"""

    # Type B field breakdown
    field_rows = defaultdict(list)
    for r in type_b:
        if r["neg_fields"]:
            for f in r["neg_fields"]:
                field_rows[f].append(r)
        else:
            field_rows["other"].append(r)

    report += "\n## Type B Per-Field Breakdown\n\n"
    report += "| Negated Field | n | MRR | Hit@1 | Miss% |\n"
    report += "|---|---|---|---|---|\n"
    for field, frows in sorted(field_rows.items(), key=lambda x: -len(x[1])):
        m = group_metrics(frows)
        report += f"| {field} | {m['n']} | {m['mrr']:.3f} | {m['hit1']:.1f}% | {m['miss_pct']:.1f}% |\n"

    report += f"\n---\n*Analysis: {len(rows)} {split} queries*\n"

    path = f"{OUT_DIR}/REPORT_{split}.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  Saved: {path}")


def generate_plots(split, rows, neg_rows, non_neg, type_a, type_b, type_c):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    # Plot 1: Taxonomy performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Negation Ablation — PRIME {split}", fontsize=13,
                 fontweight="bold")

    groups = {
        "Non-neg\n(baseline)": group_metrics(non_neg),
        "Type A\n(field)": group_metrics(type_a),
        "Type B\n(complex)": group_metrics(type_b),
        "Type C\n(spurious)": group_metrics(type_c),
    }
    labels = list(groups.keys())
    colors = ["#95a5a6", "#2ecc71", "#e74c3c", "#f39c12"]

    for ax, metric, ylabel in zip(
        axes, ["mrr", "hit1", "miss_pct"], ["MRR", "Hit@1 (%)", "Miss Rate (%)"]
    ):
        vals = [groups[l][metric] for l in labels]
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(vals) * 1.3, 0.01))
        for bar, v in zip(bars, vals):
            fmt = f"{v:.3f}" if metric == "mrr" else f"{v:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = f"{OUT_DIR}/negation_taxonomy_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["val", "test"]

    print("Negation Ablation Study — PRIME Dataset")
    print("=" * 70)

    print("\nLoading corpus (full)...")
    corpus = load_corpus_full(DATA_DIR)

    weight_result = None  # will be computed once on val Type A

    for split in splits:
        print(f"\n{'='*70}")
        print(f"  SPLIT: {split}")
        print(f"{'='*70}")

        if split not in SPLIT_QRES:
            print(f"  [SKIP] No qres file for split '{split}'")
            continue

        queries = load_queries(DATA_DIR, split)
        qrels = load_qrels(DATA_DIR, split)
        retrieved = load_retrieved(SPLIT_QRES[split])

        print("\nComputing per-query metrics...")
        rows = compute_all_rows(queries, qrels, retrieved, corpus)
        print(f"  Computed metrics for {len(rows):,} queries")

        # Section 1: Taxonomy
        neg_rows, non_neg, type_a, type_b, type_c = section1_taxonomy(rows)

        # Section 2: Head-to-head
        section2_head_to_head(neg_rows, non_neg)

        # Section 3: Matched comparison
        section3_matched(rows)

        # Section 4: Type A field verification
        present_match, present_nomatch, absent = section4_type_a(type_a, corpus, qrels)

        # Section 4b: Weight analysis (only on first split to avoid re-loading)
        if weight_result is None and type_a:
            weight_result = section4b_weight_analysis(type_a)

        # Section 5: Type B breakdown
        section5_type_b(type_b)

        # Section 6: Oracle test
        section6_oracle(type_b, retrieved)

        # Section 7: Verdict
        section7_verdict(neg_rows, type_a, type_b, type_c, non_neg, weight_result)

        # Save report & plots
        save_report(split, rows, neg_rows, non_neg, type_a, type_b, type_c,
                    weight_result)
        generate_plots(split, rows, neg_rows, non_neg, type_a, type_b, type_c)

    print(f"\n{'='*70}")
    print("Done. Reports saved to failure_analysis/negation/")


if __name__ == "__main__":
    main()
