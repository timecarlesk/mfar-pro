"""Plot α-sweep curves for v1 / v2 / no_memory × val / test.

Run from project root:
  $PY failure_analysis/type_b_memory/analysis_scripts/plot_alpha_sweep.py
Outputs:
  failure_analysis/type_b_memory/analysis_scripts/alpha_sweep.pdf
"""

import json
import os
import matplotlib.pyplot as plt

RUNS_ROOT = "output/failure_analysis/type_b_memory/runs/rerank/qwen3_32b"
OUT_PDF = "failure_analysis/type_b_memory/analysis_scripts/alpha_sweep.pdf"

ALPHAS = [0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
CONFIGS = [
    ("memory_v1", "Memory v1", "tab:blue", "o"),
    ("memory_v2", "Memory v2", "tab:red", "s"),
    ("memory_v1_no_memory", "No memory", "tab:gray", "^"),
]


def load_mrr(mv, alpha, split):
    path = f"{RUNS_ROOT}/{mv}/alpha_{alpha}_top50/memory_evaluation.json"
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    r = data[split]["reroute:ALL"]
    return r["baseline"]["mrr"], r["memory"]["mrr"]


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, split, title in [(axes[0], "val", "Validation"),
                           (axes[1], "test", "Test")]:
    baseline = None
    for mv, label, color, marker in CONFIGS:
        xs, ys = [], []
        for a in ALPHAS:
            b, m = load_mrr(mv, a, split)
            if m is not None:
                xs.append(a)
                ys.append(m)
                if baseline is None and b is not None:
                    baseline = b
        ax.plot(xs, ys, marker=marker, color=color, label=label, lw=2, markersize=7)

    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", lw=1.2, alpha=0.6,
                   label=f"mFAR baseline ({baseline:.3f})")

    ax.set_xlabel(r"Interpolation weight $\alpha$ (LLM vs mFAR)")
    if split == "val":
        ax.set_ylabel("Rerouted-query MRR")
    n = "N≈350" if split == "val" else "N=415"
    ax.set_title(f"{title} ({n})")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels([f"{a:.2f}" for a in ALPHAS], rotation=45, fontsize=8)
    ax.legend(loc="lower center", fontsize=9, framealpha=0.9)

fig.suptitle("Qwen3-32B Re-Ranking: α sweep on rerouted (negation) queries",
             fontsize=12, y=1.02)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight", format="pdf")
print(f"saved → {OUT_PDF}")
