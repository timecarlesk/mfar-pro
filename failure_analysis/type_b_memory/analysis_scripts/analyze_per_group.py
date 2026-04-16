"""Per-(answer_type, negation_pattern) MRR comparison at α=0.7.

For each rerouted query group, compute:
  - baseline MRR (mFAR alone)
  - Memory v1 MRR
  - Memory v2 MRR
  - No memory MRR
  - Δ v2 vs v1, Δ v2 vs no_memory

Run from project root:
  $PY failure_analysis/type_b_memory/analysis_scripts/analyze_per_group.py
Outputs:
  failure_analysis/type_b_memory/analysis_scripts/per_group_stats.json
  stdout table
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from failure_analysis.utils import load_qrels, load_retrieved

DATA_DIR = "data/prime"
STAGE12_DIR = "output/failure_analysis/type_b_memory/cache/stage12/qwen3_32b"
RUNS_ROOT = "output/failure_analysis/type_b_memory/runs/rerank/qwen3_32b"
OUT_JSON = "failure_analysis/type_b_memory/analysis_scripts/per_group_stats.json"
ALPHA = 0.7
SPLIT = "test"

BASELINE_QRES = {
    "val": "output/contriever/prime_eval/final-all-0.qres",
    "test": "output/contriever/prime_eval/final-additional-all-0.qres",
}

CONFIGS = {
    "memory_v1": f"{RUNS_ROOT}/memory_v1/alpha_{ALPHA}_top50",
    "memory_v2": f"{RUNS_ROOT}/memory_v2/alpha_{ALPHA}_top50",
    "no_memory": f"{RUNS_ROOT}/memory_v1_no_memory/alpha_{ALPHA}_top50",
}

RERANKED_QRES_NAME = {
    "val": "final-all-0.qres",
    "test": "final-additional-all-0.qres",
}


def mrr_of(ranking, gold_set):
    """Reciprocal rank of the first gold doc in the ranking list."""
    for rank, docid in enumerate(ranking, 1):
        if docid in gold_set:
            return 1.0 / rank
    return 0.0


def group_key(entry):
    at = entry.get("answer_type") or "unknown"
    np_ = entry.get("negation_pattern") or "other"
    return f"{at}|{np_}"


# Load Stage 1+2 cache (use memory_v1 as canonical grouping)
stage12_path = f"{STAGE12_DIR}/memory_v1/qwen3_cache_{SPLIT}.jsonl"
rerouted = {}  # qid -> (answer_type, negation_pattern)
with open(stage12_path) as f:
    for line in f:
        e = json.loads(line)
        if not e.get("needs_reroute"):
            continue
        boost = e.get("boost_fields") or []
        suppress = e.get("suppress_fields") or []
        unm_b = e.get("unmapped_boost_fields") or []
        unm_s = e.get("unmapped_suppress_fields") or []
        if not (boost or suppress or unm_b or unm_s):
            continue
        rerouted[e["qid"]] = group_key(e)

print(f"Found {len(rerouted)} rerouted queries on {SPLIT}")

# Load qrels
qrels = load_qrels(DATA_DIR, SPLIT)  # qid -> set of gold docids

# Load baseline + each memory variant's reranked qres
baseline_rank = {qid: [d for d, _ in docs] for qid, docs in load_retrieved(BASELINE_QRES[SPLIT]).items()}

config_rank = {}
for name, out_dir in CONFIGS.items():
    path = os.path.join(out_dir, RERANKED_QRES_NAME[SPLIT])
    if not os.path.exists(path):
        print(f"  missing: {path}")
        continue
    config_rank[name] = {qid: [d for d, _ in docs] for qid, docs in load_retrieved(path).items()}

# Aggregate per group
buckets = defaultdict(lambda: {"n": 0, "baseline": [], "memory_v1": [], "memory_v2": [], "no_memory": []})

for qid, gkey in rerouted.items():
    gold = qrels.get(qid, set())
    if not gold:
        continue
    buckets[gkey]["n"] += 1
    buckets[gkey]["baseline"].append(mrr_of(baseline_rank.get(qid, []), gold))
    for name in config_rank:
        buckets[gkey][name].append(mrr_of(config_rank[name].get(qid, []), gold))

# Compute group means and sort by N
rows = []
for gkey, bucket in buckets.items():
    if bucket["n"] < 1:
        continue
    b = sum(bucket["baseline"]) / bucket["n"]
    v1 = sum(bucket["memory_v1"]) / bucket["n"] if bucket["memory_v1"] else None
    v2 = sum(bucket["memory_v2"]) / bucket["n"] if bucket["memory_v2"] else None
    nm = sum(bucket["no_memory"]) / bucket["n"] if bucket["no_memory"] else None
    rows.append({
        "group": gkey,
        "n": bucket["n"],
        "baseline": round(b, 4),
        "v1": round(v1, 4) if v1 is not None else None,
        "v2": round(v2, 4) if v2 is not None else None,
        "no_memory": round(nm, 4) if nm is not None else None,
        "v2_minus_v1": round(v2 - v1, 4) if v1 is not None and v2 is not None else None,
        "v2_minus_no_mem": round(v2 - nm, 4) if nm is not None and v2 is not None else None,
    })

rows.sort(key=lambda r: -r["n"])

# Print table
print("\n" + "=" * 98)
print(f"Per-group MRR on {SPLIT} @ α={ALPHA}  (rerouted only)")
print("=" * 98)
header = f"{'group (answer_type|negation_pattern)':<42} {'N':>4}  {'base':>6} {'v1':>6} {'v2':>6} {'noMem':>6}  {'v2-v1':>7} {'v2-noMem':>9}"
print(header)
print("-" * len(header))
for r in rows:
    g = r["group"][:40]
    print(f"{g:<42} {r['n']:>4}  "
          f"{r['baseline']:>6.4f} "
          f"{(r['v1'] if r['v1'] is not None else 0):>6.4f} "
          f"{(r['v2'] if r['v2'] is not None else 0):>6.4f} "
          f"{(r['no_memory'] if r['no_memory'] is not None else 0):>6.4f}  "
          f"{(r['v2_minus_v1'] if r['v2_minus_v1'] is not None else 0):>+7.4f} "
          f"{(r['v2_minus_no_mem'] if r['v2_minus_no_mem'] is not None else 0):>+9.4f}")

# Save JSON
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump({"split": SPLIT, "alpha": ALPHA, "rows": rows}, f, indent=2)
print(f"\nsaved → {OUT_JSON}")
