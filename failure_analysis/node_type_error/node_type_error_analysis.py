"""
Node Type Error Analysis — PRIME Dataset
=========================================
Classifies each query's expected answer entity type via local LLM (Ollama),
then compares against gold document types to find node type errors.

Run from project root (multifield-adaptive-retrieval/):
  python failure_analysis/node_type_error/node_type_error_analysis.py
  python failure_analysis/node_type_error/node_type_error_analysis.py --skip-llm
  python failure_analysis/node_type_error/node_type_error_analysis.py --model qwen3:8b
"""

import argparse
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_corpus, load_queries, load_qrels, load_retrieved

# Also import cross-type classifier from multi-hop
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "multi-hop"))
from multi_hop_analysis import (
    FIELD_HINTS,
    FIELD_TO_ENTITY_TYPES,
    classify_cross_type,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data/prime"
EVAL_DIR = "output/contriever/prime_eval"
OUT_DIR = "output/failure_analysis/node_type_error"
SPLIT_QRES = {
    "val": f"{EVAL_DIR}/final-all-0.qres",
    "test": f"{EVAL_DIR}/final-additional-all-0.qres",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Valid entity types in PRIME ────────────────────────────────────────────────
VALID_TYPES = {
    "gene/protein", "drug", "disease", "anatomy", "effect/phenotype",
    "exposure", "biological_process", "cellular_component",
    "molecular_function", "pathway",
}

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Given this query about a biomedical knowledge graph, what entity type is the query asking for as the answer?

Query: "{query}"

Example: "Which drug interacts with Protein X?" → drug
(The answer type is drug, not gene/protein)

Choose exactly one from: gene/protein, drug, disease, anatomy, effect/phenotype, exposure, biological_process, cellular_component, molecular_function, pathway

If unclear, output: unknown

Output only the type, nothing else."""


# ══════════════════════════════════════════════════════════════════════════════
#  LLM Classification via Ollama
# ══════════════════════════════════════════════════════════════════════════════

def call_ollama(prompt, model, endpoint):
    """Call Ollama API and return raw text response."""
    url = f"{endpoint}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 30},
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


def parse_llm_output(raw_output):
    """Robustly parse LLM output to one of the valid types or 'unknown'.

    Strategy: strip thinking tags → strip whitespace → exact match →
              substring match on last line → unknown.
    """
    cleaned = raw_output

    # Strip Qwen3 <think>...</think> blocks if present
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip().lower()

    # Exact match
    if cleaned in VALID_TYPES:
        return cleaned

    # Try last non-empty line only (avoid matching types in reasoning text)
    last_line = ""
    for line in reversed(cleaned.splitlines()):
        line = line.strip()
        if line:
            last_line = line
            break

    if last_line in VALID_TYPES:
        return last_line

    # Substring match on last line
    matches = [t for t in VALID_TYPES if t in last_line]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return max(matches, key=len)

    # Fallback: substring match on full text
    matches = [t for t in VALID_TYPES if t in cleaned]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return max(matches, key=len)

    return "unknown"


def classify_queries_with_llm(queries, model, endpoint, cache_path):
    """Classify all queries using Ollama, with caching for resume support."""
    # Load existing cache
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cached[entry["qid"]] = entry
        print(f"  Loaded {len(cached)} cached classifications")

    # Classify uncached queries
    remaining = {qid: q for qid, q in queries.items() if qid not in cached}
    if remaining:
        print(f"  Classifying {len(remaining)} queries with {model}...")
        with open(cache_path, "a") as f:
            for i, (qid, query_text) in enumerate(remaining.items()):
                prompt = PROMPT_TEMPLATE.replace("{query}", query_text)
                try:
                    raw = call_ollama(prompt, model, endpoint)
                except Exception as e:
                    print(f"    ERROR qid={qid}: {e}")
                    raw = ""

                predicted = parse_llm_output(raw)
                entry = {"qid": qid, "predicted_type": predicted, "raw_output": raw}
                cached[qid] = entry
                f.write(json.dumps(entry) + "\n")
                f.flush()

                if (i + 1) % 100 == 0:
                    print(f"    {i + 1}/{len(remaining)} done")

    print(f"  Total classifications: {len(cached)}")
    return cached


# ══════════════════════════════════════════════════════════════════════════════
#  Sample Validation
# ══════════════════════════════════════════════════════════════════════════════

def write_sample_validation(queries, classifications, n=50):
    """Write a random sample for manual spot-checking."""
    qids = list(classifications.keys())
    sample = random.sample(qids, min(n, len(qids)))
    sample.sort()

    path = f"{OUT_DIR}/sample_validation.txt"
    with open(path, "w") as f:
        f.write(f"Sample Validation — {len(sample)} queries\n")
        f.write("=" * 70 + "\n\n")
        for qid in sample:
            query = queries.get(qid, "???")
            pred = classifications[qid]["predicted_type"]
            raw = classifications[qid]["raw_output"].strip()
            f.write(f"[{qid}] predicted={pred}  raw=\"{raw}\"\n")
            f.write(f"  Q: {query[:200]}\n\n")
    print(f"  Saved sample validation: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(queries, classifications, corpus, qrels, retrieved, qid_to_split):
    """Compare LLM-predicted types against gold doc types, compute metrics."""
    rows = []
    multi_gold_type_count = 0

    for qid, query_text in queries.items():
        if qid not in qrels or qid not in classifications:
            continue

        gold_doc_ids = qrels[qid]
        gold_types = list({corpus[did]["type"] for did in gold_doc_ids if did in corpus})

        if len(gold_types) > 1:
            multi_gold_type_count += 1

        predicted = classifications[qid]["predicted_type"]

        # Retrieval metrics
        docs = retrieved.get(qid, [])
        top100_ids = [d[0] for d in docs[:100]]
        first_rel = next((i for i, did in enumerate(top100_ids) if did in gold_doc_ids), -1)
        rr = 0.0 if first_rel < 0 else 1.0 / (first_rel + 1)
        hit1 = 1 if first_rel == 0 else 0
        miss = 1 if first_rel < 0 else 0

        # Top-1 retrieved entity type
        top1_type = corpus[top100_ids[0]]["type"] if top100_ids and top100_ids[0] in corpus else ""

        # Node type error: ALL gold types differ from predicted
        if predicted == "unknown":
            label = "unknown_prediction"
        elif not gold_types:
            label = "no_gold_in_corpus"
        elif all(t != predicted for t in gold_types):
            label = "node_type_error"
        else:
            label = "match"

        # Single-hop vs multi-hop (via cross-type classifier)
        hop_type = ""
        if label == "node_type_error":
            ct_label, _, _ = classify_cross_type(query_text, gold_types)
            hop_type = "multi_hop" if ct_label == "cross_type" else "single_hop"

        rows.append({
            "qid": qid,
            "query": query_text,
            "split": qid_to_split.get(qid, "unknown"),
            "predicted_type": predicted,
            "gold_types": gold_types,
            "top1_type": top1_type,
            "rr": rr,
            "hit1": hit1,
            "miss": miss,
            "label": label,
            "hop_type": hop_type,
        })

    return rows, multi_gold_type_count


def group_metrics(rows, label_filter):
    """Compute aggregate metrics for a subset of rows."""
    group = [r for r in rows if label_filter(r)]
    n = len(group)
    if n == 0:
        return {"n": 0, "mrr": 0, "hit1_pct": 0, "miss_pct": 0}
    mrr = sum(r["rr"] for r in group) / n
    hit1_pct = 100 * sum(r["hit1"] for r in group) / n
    miss_pct = 100 * sum(r["miss"] for r in group) / n
    return {"n": n, "mrr": mrr, "hit1_pct": hit1_pct, "miss_pct": miss_pct}


# ══════════════════════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════════════════════

def save_tsv(rows):
    """Save all node type error instances as TSV."""
    errors = [r for r in rows if r["label"] == "node_type_error"]
    errors.sort(key=lambda r: r["rr"])

    path = f"{OUT_DIR}/node_type_errors.tsv"
    with open(path, "w") as f:
        f.write("qid\tsplit\tquery\tpredicted_type\tgold_types\ttop1_type\tRR\thop_type\n")
        for r in errors:
            gold_str = ",".join(r["gold_types"])
            query_clean = r["query"].replace("\t", " ")[:300]
            f.write(f"{r['qid']}\t{r['split']}\t{query_clean}\t{r['predicted_type']}\t{gold_str}\t{r['top1_type']}\t{r['rr']:.4f}\t{r['hop_type']}\n")
    print(f"  Saved {len(errors)} error instances: {path}")


def save_report(rows, multi_gold_type_count):
    """Generate markdown report."""
    total = len(rows)
    m_match = group_metrics(rows, lambda r: r["label"] == "match")
    m_error = group_metrics(rows, lambda r: r["label"] == "node_type_error")
    m_single = group_metrics(rows, lambda r: r["label"] == "node_type_error" and r["hop_type"] == "single_hop")
    m_multi = group_metrics(rows, lambda r: r["label"] == "node_type_error" and r["hop_type"] == "multi_hop")
    m_unknown = group_metrics(rows, lambda r: r["label"] == "unknown_prediction")

    # Confusion: predicted_type → gold_type counts (errors only)
    confusion = Counter()
    for r in rows:
        if r["label"] == "node_type_error":
            for gt in r["gold_types"]:
                confusion[(r["predicted_type"], gt)] += 1

    error_rate = 100 * m_error['n'] / total if total > 0 else 0

    # Per-split breakdown
    splits = sorted({r["split"] for r in rows})

    report = f"""# Node Type Error Analysis Report
## PRIME Dataset

**Method**: LLM-based query answer type classification (Ollama) + comparison with gold doc types.

A **node type error** occurs when the entity type the query asks for (predicted by LLM) differs from ALL gold document entity types.

---

## Overall Summary

| Category | n | MRR | Hit@1 | Miss% |
|----------|---|-----|-------|-------|
| Match (no error) | {m_match['n']} | {m_match['mrr']:.3f} | {m_match['hit1_pct']:.1f}% | {m_match['miss_pct']:.1f}% |
| **Node type error (all)** | **{m_error['n']}** | **{m_error['mrr']:.3f}** | **{m_error['hit1_pct']:.1f}%** | **{m_error['miss_pct']:.1f}%** |
| — Single-hop | {m_single['n']} | {m_single['mrr']:.3f} | {m_single['hit1_pct']:.1f}% | {m_single['miss_pct']:.1f}% |
| — Multi-hop | {m_multi['n']} | {m_multi['mrr']:.3f} | {m_multi['hit1_pct']:.1f}% | {m_multi['miss_pct']:.1f}% |
| Unknown prediction | {m_unknown['n']} | {m_unknown['mrr']:.3f} | {m_unknown['hit1_pct']:.1f}% | {m_unknown['miss_pct']:.1f}% |

Node type error rate: **{error_rate:.1f}%** ({m_error['n']}/{total})

Queries with multiple gold doc types: **{multi_gold_type_count}**

---

## Per-Split Breakdown

| Split | Total | Match | Node Type Error | — Single-hop | — Multi-hop | Unknown | Error Rate |
|-------|-------|-------|-----------------|--------------|-------------|---------|------------|
"""
    for s in splits:
        s_rows = [r for r in rows if r["split"] == s]
        s_total = len(s_rows)
        s_match = sum(1 for r in s_rows if r["label"] == "match")
        s_error = sum(1 for r in s_rows if r["label"] == "node_type_error")
        s_single = sum(1 for r in s_rows if r["label"] == "node_type_error" and r["hop_type"] == "single_hop")
        s_multi = sum(1 for r in s_rows if r["label"] == "node_type_error" and r["hop_type"] == "multi_hop")
        s_unknown = sum(1 for r in s_rows if r["label"] == "unknown_prediction")
        s_rate = 100 * s_error / s_total if s_total > 0 else 0
        report += f"| {s} | {s_total} | {s_match} | {s_error} | {s_single} | {s_multi} | {s_unknown} | {s_rate:.1f}% |\n"

    # Per-split MRR for error vs match
    report += "\n| Split | Match MRR | Error MRR | Match Hit@1 | Error Hit@1 |\n"
    report += "|-------|-----------|-----------|-------------|-------------|\n"
    for s in splits:
        s_rows = [r for r in rows if r["split"] == s]
        sm = group_metrics(s_rows, lambda r: r["label"] == "match")
        se = group_metrics(s_rows, lambda r: r["label"] == "node_type_error")
        report += f"| {s} | {sm['mrr']:.3f} | {se['mrr']:.3f} | {sm['hit1_pct']:.1f}% | {se['hit1_pct']:.1f}% |\n"

    report += "\n"

    if confusion:
        gold_types_set = sorted({gt for _, gt in confusion})
        pred_types = sorted({pt for pt, _ in confusion})
        report += "---\n\n## Confusion Matrix (predicted → gold, errors only)\n\n"
        report += "| Predicted \\\\ Gold | " + " | ".join(gold_types_set) + " |\n"
        report += "|---" + "|---" * len(gold_types_set) + "|\n"
        for pt in pred_types:
            counts = [str(confusion.get((pt, gt), 0)) for gt in gold_types_set]
            report += f"| {pt} | {' | '.join(counts)} |\n"

    report += "\n---\n\n## All Node Type Error Instances\n\n"
    errors = sorted([r for r in rows if r["label"] == "node_type_error"], key=lambda r: (r["split"], r["rr"]))
    for r in errors:
        gold_str = ", ".join(r["gold_types"])
        report += f"- **[{r['qid']}]** ({r['split']}) RR={r['rr']:.2f}, predicted={r['predicted_type']}, gold={gold_str}, hop={r['hop_type']}\n"
        report += f"  - {r['query'][:200]}\n\n"

    report += f"\n---\n\n*Analysis: {total} queries*\n"

    path = f"{OUT_DIR}/REPORT.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved report: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    # ── Performance comparison bar chart ──
    m_match = group_metrics(rows, lambda r: r["label"] == "match")
    m_single = group_metrics(rows, lambda r: r["label"] == "node_type_error" and r["hop_type"] == "single_hop")
    m_multi = group_metrics(rows, lambda r: r["label"] == "node_type_error" and r["hop_type"] == "multi_hop")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Node Type Error: Performance Comparison", fontsize=13, fontweight="bold")

    labels = [f"Match\n(n={m_match['n']})", f"Single-hop\nerror (n={m_single['n']})", f"Multi-hop\nerror (n={m_multi['n']})"]
    colors = ["#2ecc71", "#e67e22", "#e74c3c"]

    for ax, metric, ylabel in zip(axes, ["mrr", "hit1_pct", "miss_pct"], ["MRR", "Hit@1 (%)", "Miss Rate (%)"]):
        vals = [m_match[metric], m_single[metric], m_multi[metric]]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.4 if max(vals) > 0 else 1)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.3f}" if metric == "mrr" else f"{v:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, fmt,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/performance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/performance_comparison.png")

    # ── Confusion matrix heatmap ──
    errors = [r for r in rows if r["label"] == "node_type_error"]
    if not errors:
        return

    confusion = Counter()
    for r in errors:
        for gt in r["gold_types"]:
            confusion[(r["predicted_type"], gt)] += 1

    pred_types = sorted({pt for pt, _ in confusion})
    gold_types = sorted({gt for _, gt in confusion})

    matrix = np.zeros((len(pred_types), len(gold_types)))
    for i, pt in enumerate(pred_types):
        for j, gt in enumerate(gold_types):
            matrix[i, j] = confusion.get((pt, gt), 0)

    fig, ax = plt.subplots(figsize=(max(8, len(gold_types)), max(6, len(pred_types) * 0.6)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(gold_types)))
    ax.set_xticklabels(gold_types, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pred_types)))
    ax.set_yticklabels(pred_types, fontsize=9)
    ax.set_xlabel("Gold Entity Type")
    ax.set_ylabel("Predicted (Query Asks For)")
    ax.set_title("Node Type Error Confusion Matrix", fontweight="bold")

    for i in range(len(pred_types)):
        for j in range(len(gold_types)):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Node Type Error Analysis")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model name")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434", help="Ollama API endpoint")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM classification, use cached results")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Splits to analyze")
    args = parser.parse_args()

    cache_path = f"{OUT_DIR}/llm_classifications.jsonl"

    # Load data
    print("Loading data...")
    corpus = load_corpus(DATA_DIR)

    queries = {}
    qid_to_split = {}  # track which split each query comes from
    qrels_all = defaultdict(set)
    retrieved_all = defaultdict(list)
    for split in args.splits:
        q = load_queries(DATA_DIR, split)
        for qid in q:
            qid_to_split[qid] = split
        queries.update(q)
        qr = load_qrels(DATA_DIR, split)
        for qid, docs in qr.items():
            qrels_all[qid] |= docs
        if split in SPLIT_QRES:
            ret = load_retrieved(SPLIT_QRES[split])
            for qid, docs in ret.items():
                if qid not in retrieved_all:
                    retrieved_all[qid] = docs

    print(f"\n  Total: {len(queries)} queries ({', '.join(f'{s}: {sum(1 for v in qid_to_split.values() if v==s)}' for s in args.splits)})")

    # LLM classification
    if args.skip_llm:
        print("\nLoading cached LLM classifications...")
        classifications = {}
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                classifications[entry["qid"]] = entry
        print(f"  Loaded {len(classifications)} classifications")
    else:
        print(f"\nRunning LLM classification ({args.model})...")
        classifications = classify_queries_with_llm(queries, args.model, args.endpoint, cache_path)

    # Parse stats
    type_counts = Counter(c["predicted_type"] for c in classifications.values())
    print("\n  Predicted type distribution:")
    for t, n in type_counts.most_common():
        print(f"    {t}: {n}")

    # Sample validation
    random.seed(42)
    write_sample_validation(queries, classifications)

    # Run analysis
    print("\nRunning node type error analysis...")
    rows, multi_gold_type_count = run_analysis(queries, classifications, corpus, qrels_all, retrieved_all, qid_to_split)

    # Print summary
    error_count = sum(1 for r in rows if r["label"] == "node_type_error")
    match_count = sum(1 for r in rows if r["label"] == "match")
    unknown_count = sum(1 for r in rows if r["label"] == "unknown_prediction")
    single_count = sum(1 for r in rows if r["label"] == "node_type_error" and r["hop_type"] == "single_hop")
    multi_count = sum(1 for r in rows if r["label"] == "node_type_error" and r["hop_type"] == "multi_hop")

    print(f"\n  Results:")
    print(f"    Match:           {match_count}")
    print(f"    Node type error: {error_count} (single-hop: {single_count}, multi-hop: {multi_count})")
    print(f"    Unknown:         {unknown_count}")
    print(f"    Multi-gold-type: {multi_gold_type_count}")

    # Save outputs
    print("\nGenerating outputs...")
    save_tsv(rows)
    save_report(rows, multi_gold_type_count)
    plot_results(rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
