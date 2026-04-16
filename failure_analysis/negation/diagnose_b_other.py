"""
Diagnose Type B.other queries — dump actual examples with negation tokens,
gold doc info, and attempted field-level categorization.

Also runs a causal oracle test: for each Type B miss, strip negation tokens
and check if the gold doc's BM25 score on the "affirmative core" is higher
than its score on the original query. If yes → negation is the blocker.

Run from project root:
  python failure_analysis/negation/diagnose_b_other.py [val|test]
"""

import json
import re
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_corpus_full, load_queries, load_qrels, load_retrieved

# Import the classification logic from negation_ablation
sys.path.insert(0, os.path.dirname(__file__))
from negation_ablation import (
    NEGATION_PATTERN, classify_negation, classify_query_category,
    compute_row, group_metrics, SPLIT_QRES,
)

DATA_DIR = "data/prime"
OUT_DIR = "output/failure_analysis/negation"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Negation token extractor ────────────────────────────────────────────────
NEGATION_TOKEN_RE = re.compile(
    r"\b(?:not|no|without|lack(?:s|ing)?|absent|neither|nor|never|"
    r"cannot|can't|don't|doesn't|didn't|won't|isn't|aren't|wasn't|weren't|"
    r"exclude[ds]?|non-)\b|"
    r"\bun(?:express|relat|associat|link|affect|involv)\w*\b",
    re.IGNORECASE,
)

# ── Negation stripping (for oracle test) ─────────────────────────────────────
NEGATION_STRIP_RE = re.compile(
    r"\b(?:not|no|without|lack(?:s|ing)?|absent|neither|nor|never|"
    r"cannot|can't|don't|doesn't|didn't|won't|isn't|aren't|wasn't|weren't|"
    r"exclude[ds]?)\b",
    re.IGNORECASE,
)


def strip_negation(query_text):
    """Remove negation tokens to get the 'affirmative core'."""
    return NEGATION_STRIP_RE.sub("", query_text).strip()


def get_context_around_negation(query_text, window=8):
    """Extract the words surrounding each negation token for categorization."""
    contexts = []
    for m in NEGATION_TOKEN_RE.finditer(query_text):
        start = max(0, m.start() - 80)
        end = min(len(query_text), m.end() + 80)
        snippet = query_text[start:end].strip()
        contexts.append((m.group(), snippet))
    return contexts


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    print(f"Diagnosing Type B.other — {split} split")
    print("=" * 70)

    corpus = load_corpus_full(DATA_DIR)
    queries = load_queries(DATA_DIR, split)
    qrels = load_qrels(DATA_DIR, split)
    retrieved = load_retrieved(SPLIT_QRES[split])

    # Classify all queries
    b_other = []
    b_mapped = []
    all_type_b = []

    for qid, text in queries.items():
        if qid not in qrels:
            continue
        if not NEGATION_PATTERN.search(text):
            continue
        neg_type, neg_subtype, neg_fields = classify_negation(text)
        if neg_type != "B":
            continue

        gold = qrels[qid]
        docs = retrieved.get(qid, [])
        row = compute_row(qid, text, gold, docs, corpus)
        row["neg_subtype"] = neg_subtype
        row["neg_fields"] = neg_fields

        all_type_b.append(row)
        if neg_subtype == "other":
            b_other.append(row)
        else:
            b_mapped.append(row)

    print(f"\n  Total Type B: {len(all_type_b)}")
    print(f"  B.mapped (has field): {len(b_mapped)}")
    print(f"  B.other (no field):   {len(b_other)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 1: Categorize B.other by negation context
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 1: B.other Query Categorization")
    print(f"{'='*70}")

    # Extract negation contexts and try to auto-categorize
    pattern_categories = {
        "should_not_treat": re.compile(
            r"should\s+not\s+be\s+(?:treat|manage|prescri)"
            r"|should\s+not\s+(?:\w+\s+){0,2}(?:treat|manage)",
            re.IGNORECASE,
        ),
        "lack_treatment": re.compile(
            r"lack(?:s|ing)?\s+(?:\w+\s+){0,3}(?:treat|drug|medic|therap|approv|cure|remedy)"
            r"|(?:no|without)\s+(?:\w+\s+){0,3}(?:treat|drug|medic|therap|cure)",
            re.IGNORECASE,
        ),
        "lack_expression_linked": re.compile(
            r"lack\w*\s+(?:the\s+)?(?:\w+\s+){0,3}expression\s+(?:\w+\s+){0,3}"
            r"(?:linked|connected|related|associated|involved|play|role|regulate)",
            re.IGNORECASE,
        ),
        "lack_expression_tissue": re.compile(
            r"lack\w*\s+(?:the\s+)?(?:\w+\s+){0,3}expression",
            re.IGNORECASE,
        ),
        "but_not_in": re.compile(
            r"but\s+not\s+(?:in|expressed|found|present|detected)",
            re.IGNORECASE,
        ),
        "not_interact": re.compile(
            r"(?:not|don't|doesn't|won't)\s+(?:\w+\s+){0,2}"
            r"(?:interact|engage|bind|affect|influence|modulate)",
            re.IGNORECASE,
        ),
        "not_aggravate_risk": re.compile(
            r"(?:not|don't|doesn't|won't|without)\s+(?:\w+\s+){0,2}"
            r"(?:aggravat|worsen|exacerbat|risk|pose|caus|damag|harm|increas)",
            re.IGNORECASE,
        ),
        "not_only_synergy": re.compile(r"not\s+only\b", re.IGNORECASE),
    }

    cat_counts = Counter()
    cat_examples = defaultdict(list)

    for r in b_other:
        assigned = False
        for cat_name, pat in pattern_categories.items():
            if pat.search(r["query"]):
                cat_counts[cat_name] += 1
                if len(cat_examples[cat_name]) < 5:
                    cat_examples[cat_name].append(r)
                assigned = True
                break
        if not assigned:
            cat_counts["uncategorized"] += 1
            if len(cat_examples["uncategorized"]) < 10:
                cat_examples["uncategorized"].append(r)

    print(f"\n  Auto-categorization of {len(b_other)} B.other queries:\n")
    print(f"  {'Category':<30} {'Count':>5} {'%':>6}")
    print("  " + "-" * 44)
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat:<30} {cnt:>5} {100*cnt/max(len(b_other),1):>5.1f}%")

    # Print examples per category
    for cat in [c for c, _ in cat_counts.most_common()]:
        print(f"\n  --- {cat} (n={cat_counts[cat]}) ---")
        for r in cat_examples[cat]:
            neg_tokens = NEGATION_TOKEN_RE.findall(r["query"])
            status = "MISS" if r["miss"] else f"rank={r['first_rel']+1}"
            gold_types = r["gold_types"]
            print(f"  [{r['qid']}] {status} MRR={r['rr']:.2f} gold={gold_types}")
            print(f"    neg_tokens: {neg_tokens}")
            print(f"    {r['query'][:200]}")
            print()

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 2: Is negation the PRIMARY cause of failure?
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 2: Negation as Primary vs Secondary Cause")
    print(f"{'='*70}")

    # For Type B queries that are complete misses or rank > 20:
    # Check if the query has other difficulty factors (multi-hop, rare entity type)
    b_failures = [r for r in all_type_b if r["miss"] or r["first_rel"] >= 20]
    b_successes = [r for r in all_type_b if not r["miss"] and r["first_rel"] < 5]

    print(f"\n  Type B failures (miss or rank > 20): {len(b_failures)}")
    print(f"  Type B successes (rank <= 5):         {len(b_successes)}")

    # Compare gold entity type distribution: failures vs successes
    fail_types = Counter()
    succ_types = Counter()
    for r in b_failures:
        fail_types.update(r["gold_types"])
    for r in b_successes:
        succ_types.update(r["gold_types"])

    all_types = set(list(fail_types.keys()) + list(succ_types.keys()))
    print(f"\n  Gold entity type distribution:")
    print(f"  {'Entity Type':<25} {'Failures':>10} {'Successes':>10}")
    print("  " + "-" * 48)
    for t in sorted(all_types):
        print(f"  {t:<25} {fail_types.get(t,0):>10} {succ_types.get(t,0):>10}")

    # Compare query length: failures vs successes
    fail_lens = [len(r["query"].split()) for r in b_failures]
    succ_lens = [len(r["query"].split()) for r in b_successes]
    print(f"\n  Query length (words):")
    print(f"    Failures: mean={sum(fail_lens)/max(len(fail_lens),1):.1f}")
    print(f"    Successes: mean={sum(succ_lens)/max(len(succ_lens),1):.1f}")

    # Compare negation token count
    fail_negs = [len(NEGATION_TOKEN_RE.findall(r["query"])) for r in b_failures]
    succ_negs = [len(NEGATION_TOKEN_RE.findall(r["query"])) for r in b_successes]
    print(f"\n  Negation token count per query:")
    print(f"    Failures: mean={sum(fail_negs)/max(len(fail_negs),1):.1f}")
    print(f"    Successes: mean={sum(succ_negs)/max(len(succ_negs),1):.1f}")

    # Compare query categories
    fail_cats = Counter(r["category"] for r in b_failures)
    succ_cats = Counter(r["category"] for r in b_successes)
    print(f"\n  Query category distribution:")
    print(f"  {'Category':<25} {'Failures':>10} {'Successes':>10}")
    print("  " + "-" * 48)
    for cat in sorted(set(list(fail_cats.keys()) + list(succ_cats.keys()))):
        print(f"  {cat:<25} {fail_cats.get(cat,0):>10} {succ_cats.get(cat,0):>10}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 3: Causal Oracle Test — Strip Negation, Check Gold Doc Rank
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 3: Causal Oracle — BM25 Score With vs Without Negation")
    print(f"{'='*70}")

    try:
        from mfar.data.index import BM25sSparseIndex
        bm25_available = True
    except ImportError:
        bm25_available = False

    if not bm25_available:
        print("\n  [SKIP] Cannot import BM25sSparseIndex — run with PYTHONPATH set")
        print("  export PYTHONPATH=$(pwd):$PYTHONPATH")
        _oracle_fallback(all_type_b, corpus, retrieved)
        return

    # Load BM25 index (single_sparse — the full-document index)
    index_path = "data/prime/single_sparse_sparse_index"
    if not os.path.exists(index_path):
        print(f"\n  [SKIP] BM25 index not found at {index_path}")
        _oracle_fallback(all_type_b, corpus, retrieved)
        return

    print(f"\n  Loading BM25 index from {index_path}...")
    bm25_index = BM25sSparseIndex.load(index_path)
    print(f"  BM25 index loaded.")

    # For each Type B query: score gold doc with original query vs affirmative core
    improved = 0
    degraded = 0
    unchanged = 0
    total_tested = 0
    skipped_no_key = 0
    skipped_error = 0
    improvements = []

    type_b_misses = [r for r in all_type_b if r["miss"]]
    type_b_found = [r for r in all_type_b if not r["miss"]]

    print(f"\n  Testing on {len(all_type_b)} Type B queries "
          f"({len(type_b_misses)} misses, {len(type_b_found)} found)...")

    for r in all_type_b:
        original_query = r["query"]
        affirmative = strip_negation(original_query)

        if not affirmative.strip():
            continue

        for gold_id in r["gold_ids"]:
            # BM25sSparseIndex.score(query: str, keys: Sequence[str]) -> np.ndarray
            if gold_id not in bm25_index.key_to_id:
                skipped_no_key += 1
                continue
            try:
                orig_scores = bm25_index.score(original_query, [gold_id])
                affirm_scores = bm25_index.score(affirmative, [gold_id])
                orig_score = float(orig_scores[0]) if len(orig_scores) > 0 else 0.0
                affirm_score = float(affirm_scores[0]) if len(affirm_scores) > 0 else 0.0
            except Exception as e:
                if skipped_error == 0:
                    print(f"    [DEBUG] First scoring error: {type(e).__name__}: {e}")
                    print(f"            gold_id={gold_id}, query[:50]={original_query[:50]}")
                skipped_error += 1
                continue

            total_tested += 1
            delta = affirm_score - orig_score

            if delta > 0.01:
                improved += 1
                improvements.append({
                    "qid": r["qid"],
                    "orig_score": orig_score,
                    "affirm_score": affirm_score,
                    "delta": delta,
                    "was_miss": r["miss"],
                    "query_short": original_query[:100],
                })
            elif delta < -0.01:
                degraded += 1
            else:
                unchanged += 1
            break  # only test first gold doc

    print(f"\n  Skipped (gold_id not in index): {skipped_no_key}")
    print(f"  Skipped (scoring error): {skipped_error}")
    print(f"\n  Oracle results ({total_tested} query-doc pairs tested):")
    print(f"    Gold doc score IMPROVED without negation: {improved} "
          f"({100*improved/max(total_tested,1):.1f}%)")
    print(f"    Gold doc score DEGRADED without negation: {degraded} "
          f"({100*degraded/max(total_tested,1):.1f}%)")
    print(f"    Unchanged:                                {unchanged} "
          f"({100*unchanged/max(total_tested,1):.1f}%)")

    # Break down by miss vs found
    imp_miss = sum(1 for x in improvements if x["was_miss"])
    imp_found = sum(1 for x in improvements if not x["was_miss"])
    print(f"\n    Of the {improved} improved:")
    print(f"      Were complete misses: {imp_miss}")
    print(f"      Were found (rank improvement): {imp_found}")

    if improved > 0:
        pct_of_misses = 100 * imp_miss / max(len(type_b_misses), 1)
        print(f"\n    {pct_of_misses:.1f}% of Type B misses would benefit from "
              f"negation stripping")
        print(f"    → A Boolean filter that strips negation for retrieval, then")
        print(f"      re-applies it for post-filtering, could recover these.")

    # Print top improvements
    improvements.sort(key=lambda x: -x["delta"])
    print(f"\n  Top-10 largest score improvements:")
    for x in improvements[:10]:
        status = "MISS" if x["was_miss"] else "found"
        print(f"    [{x['qid']}] {status} "
              f"orig={x['orig_score']:.3f} → affirm={x['affirm_score']:.3f} "
              f"(+{x['delta']:.3f})")
        print(f"      {x['query_short']}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 4: mFAR Weight Analysis — indication vs contraindication routing
    # ══════════════════════════════════════════════════════════════════════════
    weight_report = section4_weight_analysis(all_type_b)

    # Save results
    _save_report(split, b_other, cat_counts, cat_examples, all_type_b,
                 improved, degraded, unchanged, total_tested,
                 type_b_misses, imp_miss, improvements, weight_report)


CKPT_DIR = "output/contriever/prime"


def section4_weight_analysis(all_type_b):
    """
    For Type B indication/contraindication queries, extract mFAR's softmax
    field weights and check if the model correctly routes to the right field.

    Key question: for "should not be treated with X" (contraindication),
    does w(contraindication) > w(indication)? Or does the model conflate them?
    """
    print(f"\n{'='*70}")
    print("PART 4: mFAR Weight Routing — indication vs contraindication")
    print(f"{'='*70}")

    try:
        import torch
        from mfar.modeling.util import prepare_model
        from mfar.data.schema import resolve_fields
    except ImportError as e:
        print(f"\n  [SKIP] Cannot import mFAR modules: {e}")
        return None

    best_txt = f"{CKPT_DIR}/best.txt"
    if not os.path.exists(best_txt):
        print(f"\n  [SKIP] No checkpoint at {best_txt}")
        return None

    with open(best_txt) as f:
        ckpt_suffix = f.read().strip().split("/")[-1]
    ckpt_path = f"{CKPT_DIR}/{ckpt_suffix}"
    print(f"\n  Checkpoint: {ckpt_path}")

    field_info = resolve_fields(
        "all_dense,all_sparse,single_dense,single_sparse", "prime")
    field_names = list(field_info.keys())

    # Find field indices (keys use spaces: "indication_dense", "contraindication_dense")
    ind_dense = next((i for i, k in enumerate(field_names)
                      if k == "indication_dense"), None)
    con_dense = next((i for i, k in enumerate(field_names)
                      if k == "contraindication_dense"), None)
    ind_sparse = next((i for i, k in enumerate(field_names)
                       if k == "indication_sparse"), None)
    con_sparse = next((i for i, k in enumerate(field_names)
                       if k == "contraindication_sparse"), None)

    if ind_dense is None or con_dense is None:
        print(f"  [SKIP] Cannot find indication/contraindication field indices")
        return None

    print(f"  indication_dense       idx={ind_dense}")
    print(f"  contraindication_dense idx={con_dense}")

    # Load encoder + weight matrix
    tokenizer, encoder, _ = prepare_model(
        "facebook/contriever-msmarco", normalize=False, with_decoder=False)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    w_key = next((k for k in state_dict
                  if "mixture_of_fields_layer.weight" in k), None)
    if w_key is None:
        print("  [SKIP] Cannot find mixture weight in checkpoint")
        return None

    W = state_dict[w_key]

    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k.replace("encoder.", "", 1)] = v
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()

    # Split Type B by subtype
    b_indication = [r for r in all_type_b if r["neg_subtype"] == "indication"]
    b_contraindication = [r for r in all_type_b
                          if r["neg_subtype"] == "contraindication"]

    print(f"\n  B.indication queries:       {len(b_indication)}")
    print(f"  B.contraindication queries: {len(b_contraindication)}")

    results = {}

    for label, group, expected_field, wrong_field, exp_idx, wrong_idx in [
        ("B.indication", b_indication,
         "indication_dense", "contraindication_dense", ind_dense, con_dense),
        ("B.contraindication", b_contraindication,
         "contraindication_dense", "indication_dense", con_dense, ind_dense),
    ]:
        if not group:
            continue

        correct = 0
        conflated = 0
        w_expected_list = []
        w_wrong_list = []
        top_fields_counter = Counter()
        per_query = []

        for r in group:
            tokens = tokenizer(
                r["query"], max_length=64, truncation=True,
                padding=True, return_tensors="pt")
            with torch.no_grad():
                q_emb = encoder(tokens)["sentence_embedding"]
                raw = q_emb @ W
                weights = torch.softmax(raw, dim=1).squeeze(0)

            w_exp = weights[exp_idx].item()
            w_wrn = weights[wrong_idx].item()
            w_expected_list.append(w_exp)
            w_wrong_list.append(w_wrn)

            if w_exp > w_wrn:
                correct += 1
            else:
                conflated += 1

            # Top-3 fields for this query
            top3_idx = torch.topk(weights, k=5).indices.tolist()
            top3_names = [field_names[i] for i in top3_idx]
            for fn in top3_names:
                top_fields_counter[fn] += 1

            per_query.append({
                "qid": r["qid"],
                "w_expected": w_exp,
                "w_wrong": w_wrn,
                "ratio": w_exp / max(w_wrn, 1e-8),
                "top3": top3_names,
                "miss": r["miss"],
                "rr": r["rr"],
            })

        n = len(group)
        mean_exp = sum(w_expected_list) / n
        mean_wrn = sum(w_wrong_list) / n
        correct_pct = 100 * correct / n
        conflated_pct = 100 * conflated / n

        print(f"\n  --- {label} ({n} queries) ---")
        print(f"    Correct routing (w_{expected_field} > w_{wrong_field}): "
              f"{correct} ({correct_pct:.1f}%)")
        print(f"    Conflated  (w_{wrong_field} >= w_{expected_field}):     "
              f"{conflated} ({conflated_pct:.1f}%)")
        print(f"    Mean w({expected_field}):  {mean_exp:.4f}")
        print(f"    Mean w({wrong_field}): {mean_wrn:.4f}")
        print(f"    Ratio (expected/wrong): {mean_exp/max(mean_wrn, 1e-8):.1f}x")

        print(f"\n    Top-5 fields that get highest weight for {label}:")
        for fn, cnt in top_fields_counter.most_common(5):
            print(f"      {fn:<35} appears in top-5 of {cnt}/{n} queries")

        # Show conflated examples
        conflated_queries = [q for q in per_query if q["w_expected"] <= q["w_wrong"]]
        if conflated_queries:
            print(f"\n    Conflated examples (wrong field wins):")
            for q in sorted(conflated_queries, key=lambda x: x["ratio"])[:5]:
                status = "MISS" if q["miss"] else f"RR={q['rr']:.2f}"
                print(f"      [{q['qid']}] {status}  "
                      f"w_exp={q['w_expected']:.4f} w_wrn={q['w_wrong']:.4f} "
                      f"ratio={q['ratio']:.2f}")
                print(f"        top-5 fields: {q['top3']}")

        results[label] = {
            "n": n,
            "correct": correct,
            "conflated": conflated,
            "correct_pct": correct_pct,
            "conflated_pct": conflated_pct,
            "mean_expected": mean_exp,
            "mean_wrong": mean_wrn,
        }

    # Cross-analysis: for conflated queries, is conflation correlated with failure?
    print(f"\n  --- Conflation × Failure correlation ---")
    for label, group in [("B.indication", b_indication),
                         ("B.contraindication", b_contraindication)]:
        if not group:
            continue
        exp_idx_l = ind_dense if "indication" in label else con_dense
        wrn_idx_l = con_dense if "indication" in label else ind_dense

        correctly_routed = []
        conflated_routed = []
        for r in group:
            tokens = tokenizer(
                r["query"], max_length=64, truncation=True,
                padding=True, return_tensors="pt")
            with torch.no_grad():
                q_emb = encoder(tokens)["sentence_embedding"]
                raw = q_emb @ W
                weights = torch.softmax(raw, dim=1).squeeze(0)
            if weights[exp_idx_l] > weights[wrn_idx_l]:
                correctly_routed.append(r)
            else:
                conflated_routed.append(r)

        m_correct = group_metrics(correctly_routed) if correctly_routed else {"mrr": 0, "miss_pct": 0, "n": 0}
        m_conflated = group_metrics(conflated_routed) if conflated_routed else {"mrr": 0, "miss_pct": 0, "n": 0}

        print(f"\n    {label}:")
        print(f"      Correctly routed: n={m_correct['n']}, "
              f"MRR={m_correct['mrr']:.3f}, Miss={m_correct['miss_pct']:.1f}%")
        print(f"      Conflated:        n={m_conflated['n']}, "
              f"MRR={m_conflated['mrr']:.3f}, Miss={m_conflated['miss_pct']:.1f}%")
        if m_correct["mrr"] > 0 and m_conflated["n"] > 0:
            delta = (m_conflated["mrr"] - m_correct["mrr"]) / m_correct["mrr"] * 100
            print(f"      Conflation MRR penalty: {delta:+.1f}%")

    return results


def _oracle_fallback(all_type_b, corpus, retrieved):
    """Fallback oracle: analyze top-100 retrieval overlap without BM25."""
    print("\n  Fallback: Analyzing top-100 entity type overlap for Type B misses")

    misses = [r for r in all_type_b if r["miss"]]
    if not misses:
        print("  No Type B misses.")
        return

    # For each miss: do the top-10 retrieved docs share entity type with gold?
    type_overlap = 0
    for r in misses:
        gold_types = set(r["gold_types"])
        docs = retrieved.get(r["qid"], [])
        top10_types = set()
        for did, _ in docs[:10]:
            if did in corpus:
                top10_types.add(corpus[did]["type"])
        if gold_types & top10_types:
            type_overlap += 1

    print(f"  {len(misses)} complete misses")
    print(f"  {type_overlap} ({100*type_overlap/max(len(misses),1):.1f}%) have "
          f"correct entity type in top-10")
    print(f"  → mFAR retrieves the RIGHT type but WRONG instance")
    print(f"    (negation constraint not enforced → wrong docs fill top-100)")


def _save_report(split, b_other, cat_counts, cat_examples, all_type_b,
                 improved, degraded, unchanged, total_tested,
                 type_b_misses, imp_miss, improvements, weight_report=None):
    """Save diagnostic report."""
    report = f"""# Type B Diagnosis — PRIME {split}

## B.other Auto-Categorization (n={len(b_other)})

| Category | Count | % |
|----------|-------|---|
"""
    for cat, cnt in cat_counts.most_common():
        report += f"| {cat} | {cnt} | {100*cnt/max(len(b_other),1):.1f}% |\n"

    report += f"""
## Causal Oracle: BM25 Score With vs Without Negation (n={total_tested})

| Outcome | Count | % |
|---------|-------|---|
| Gold score IMPROVED (negation was blocker) | {improved} | {100*improved/max(total_tested,1):.1f}% |
| Gold score DEGRADED | {degraded} | {100*degraded/max(total_tested,1):.1f}% |
| Unchanged | {unchanged} | {100*unchanged/max(total_tested,1):.1f}% |

Of {improved} improved: {imp_miss} were complete misses ({100*imp_miss/max(len(type_b_misses),1):.1f}% of all Type B misses).

**Interpretation**: If stripping negation improves gold doc BM25 score,
the negation tokens were actively pushing the gold doc down in ranking.
A Boolean filter approach (retrieve with affirmative core, then post-filter
by negation constraint) would directly address this.
"""

    if improvements:
        report += "\n## Top Score Improvements\n\n"
        for x in improvements[:15]:
            status = "MISS" if x["was_miss"] else "found"
            report += (f"- [{x['qid']}] {status}: "
                       f"{x['orig_score']:.3f} → {x['affirm_score']:.3f} "
                       f"(+{x['delta']:.3f})\n")
            report += f"  - {x['query_short']}\n"

    if weight_report:
        report += "\n## mFAR Weight Routing: indication vs contraindication\n\n"
        report += "| Subtype | n | Correct Routing | Conflated | Mean w(expected) | Mean w(wrong) |\n"
        report += "|---------|---|----------------|-----------|-----------------|---------------|\n"
        for label, data in weight_report.items():
            report += (f"| {label} | {data['n']} | "
                       f"{data['correct_pct']:.1f}% | {data['conflated_pct']:.1f}% | "
                       f"{data['mean_expected']:.4f} | {data['mean_wrong']:.4f} |\n")
        report += "\n"

    path = f"{OUT_DIR}/B_OTHER_DIAGNOSIS_{split}.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
