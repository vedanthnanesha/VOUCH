"""Regenerate fig_heterogeneity.pdf from the four Hateful Memes summary JSONs
(same-family from main_evaluation.py, cross-family from cross_family_eval.py)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "fig_heterogeneity.pdf"

FILES = {
    ("clean", "same"):  RESULTS / "summary_hateful_memes_clean.json",
    ("clean", "cross"): RESULTS / "summary_hateful_memes_clean.cross_family.json",
    ("adv",   "same"):  RESULTS / "summary_hateful_memes_adversarial.json",
    ("adv",   "cross"): RESULTS / "summary_hateful_memes_adversarial.cross_family.json",
}

CONDITIONS = [
    ("independent",             "Indep."),
    ("ensemble",                "Ensemble"),
    ("centralized_single",      "Central.\nsingle"),
    ("centralized_coordinator", "Central.\ncoord."),
    ("contracts",               "VOUCH\ncontracts"),
    ("contracts_random",        "VOUCH\nrandom"),
    ("contracts_adaptive",      "VOUCH\nadaptive"),
]

C_SAME  = "#D6A878"  # tan
C_CROSS = "#8A8DD0"  # lavender


def load(regime, family):
    with open(FILES[(regime, family)]) as f:
        d = json.load(f)
    return [d[k]["accuracy"] * 100 for k, _ in CONDITIONS], d["independent"]["n"]


def plot_panel(ax, regime, title):
    same,  n_same  = load(regime, "same")
    cross, n_cross = load(regime, "cross")
    x = np.arange(len(CONDITIONS))
    w = 0.4
    bars_s = ax.bar(x - w / 2, same,  w, color=C_SAME,  label="Same-family (Qwen+Qwen)")
    bars_c = ax.bar(x + w / 2, cross, w, color=C_CROSS, label="Cross-family (Llama+Qwen)")
    for bars in (bars_s, bars_c):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CONDITIONS], fontsize=9)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.set_title(f"{title}   (same-family n={n_same}, cross-family n={n_cross})",
                 loc="left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
plot_panel(axes[0], "clean", "HM \u00b7 clean")
plot_panel(axes[1], "adv",   "HM \u00b7 adv")
axes[0].set_ylabel("Accuracy")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=10)

fig.tight_layout(rect=(0, 0, 1, 0.95))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
