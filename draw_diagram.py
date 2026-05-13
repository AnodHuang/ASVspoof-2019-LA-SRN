# ojsp_figs_fp_fn_acc_f1.py
# Generate IEEE OJSP-ready figures (single-column) for:
# (1) Accuracy vs F1-score (%)  (2) False Positive vs False Negative (counts)
# Exports: PDF (vector) + EPS (vector) + PNG (600 dpi line-art backup)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) Output directory
# =========================
OUT_DIR = "figures_srn"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 1) IEEE/OJSP-friendly style
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,   # embed TrueType fonts
    "ps.fonttype": 42,
})

def save_ieee(fig, basename: str):
    """Save as PDF/EPS (vector preferred) + 600 dpi PNG (line art backup)."""
    fig.savefig(os.path.join(OUT_DIR, f"{basename}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{basename}.eps"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{basename}.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

# =========================
# 2) Data (order fixed by user)
# =========================
models = ["wav2vec large", "mit-ast", "hubert", "matty", "wav2vec base"]

# NOTE:
# - These numbers are copied from your screenshots.
# - If wav2vec base differs from wav2vec large, modify the last row accordingly.
df = pd.DataFrame({
    "Model":    models,
    "Accuracy": [0.896753, 0.922498, 0.975097, 0.898845, 0.896753],
    "F1":       [0.945567, 0.954974, 0.985947, 0.940637, 0.945567],
    "FP":       [7355,     188,      121,      415,      7355],
    "FN":       [0,        5333,     1653,     6791,     0],
})

x = np.arange(len(df))

# =========================
# 3) Figure 1: Accuracy vs F1-score (%)
# =========================
fig, ax = plt.subplots(figsize=(3.5, 2.2))  # ~single-column width
w = 0.35

ax.bar(x - w/2, df["Accuracy"] * 100, width=w,
       edgecolor="black", facecolor="white", hatch="///", label="Accuracy")
ax.bar(x + w/2, df["F1"] * 100, width=w,
       edgecolor="black", facecolor="white", hatch="...", label="F1-score")

ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(df["Model"], rotation=25, ha="right")
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle=":", linewidth=0.6)

# Put legend outside (prevents overlap)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.tight_layout(rect=[0, 0, 0.85, 1])  # reserve space on the right for legend

save_ieee(fig, "ojsp_acc_f1")

# =========================
# 4) Figure 2: False Positive vs False Negative (counts)
# =========================
fig, ax = plt.subplots(figsize=(3.5, 2.2))
w = 0.35

ax.bar(x - w/2, df["FP"], width=w,
       edgecolor="black", facecolor="white", hatch="///", label="False Positive (FP)")
ax.bar(x + w/2, df["FN"], width=w,
       edgecolor="black", facecolor="white", hatch="...", label="False Negative (FN)")

ax.set_ylabel("Count")
ax.set_xticks(x)
ax.set_xticklabels(df["Model"], rotation=25, ha="right")
ax.grid(axis="y", linestyle=":", linewidth=0.6)

# Put legend outside (prevents overlap)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.tight_layout(rect=[0, 0, 0.85, 1])

save_ieee(fig, "ojsp_fp_fn")

# =========================
# 5) Export CSV for reproducibility
# =========================
df.to_csv(os.path.join(OUT_DIR, "ojsp_metrics_fp_fn_acc_f1.csv"), index=False)

print("Saved to:", os.path.abspath(OUT_DIR))
print("Figures:")
print(" - ojsp_acc_f1.(pdf/eps/png)")
print(" - ojsp_fp_fn.(pdf/eps/png)")
print("Data:")
print(" - ojsp_metrics_fp_fn_acc_f1.csv")
