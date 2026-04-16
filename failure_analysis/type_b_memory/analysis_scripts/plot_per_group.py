"""Per-group bar chart: ΔMRR over baseline for v1 / v2 / no_memory.

Groups sorted by N (descending). Small groups (N<5) collapsed to an
"other" bucket so the chart stays readable.

Run from project root:
  $PY failure_analysis/type_b_memory/analysis_scripts/plot_per_group.py
Outputs:
  failure_analysis/type_b_memory/analysis_scripts/per_group_delta.pdf
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

IN_JSON = "failure_analysis/type_b_memory/analysis_scripts/per_group_stats.json"
OUT_PDF = "failure_analysis/type_b_memory/analysis_scripts/per_group_delta.pdf"
MIN_N = 5  # groups with N < MIN_N are hidden (too noisy)

with open(IN_JSON) as f:
    data = json.load(f)

rows = [r for r in data["rows"] if r["n"] >= MIN_N]
rows.sort(key=lambda r: -r["n"])

labels = [f'{r["group"]}\n(N={r["n"]})' for r in rows]
delta_v1 = [r["v1"] - r["baseline"] for r in rows]
delta_v2 = [r["v2"] - r["baseline"] for r in rows]
delta_nm = [r["no_memory"] - r["baseline"] for r in rows]

x = np.arange(len(labels))
w = 0.27

fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5.5))
b1 = ax.bar(x - w, delta_nm, w, label="No memory", color="tab:gray", edgecolor="white")
b2 = ax.bar(x, delta_v1, w, label="Memory v1", color="tab:blue", edgecolor="white")
b3 = ax.bar(x + w, delta_v2, w, label="Memory v2", color="tab:red", edgecolor="white")

ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel(r"$\Delta$ MRR vs mFAR baseline")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_title(
    f"Per-(answer_type, negation_pattern) $\\Delta$MRR on test  "
    f"($\\alpha=$"
    f"{data['alpha']}, Qwen3-32B, N$\\geq${MIN_N} shown)",
    fontsize=11,
)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper right", framealpha=0.9)

# Annotate bars with Δ value
for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        if abs(h) < 0.003:
            continue
        va = "bottom" if h >= 0 else "top"
        ax.text(rect.get_x() + rect.get_width() / 2, h,
                f"{h:+.3f}", ha="center", va=va, fontsize=7)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight", format="pdf")
print(f"saved → {OUT_PDF}")
