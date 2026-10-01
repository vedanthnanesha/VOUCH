"""Figures derived from results/derived/*.csv.

Outputs into figures/:
  fig_abstention_disentangled.pdf  -- action rate (nearly identical) vs hurt|act (big drop)
  fig_ci_hurt_reduction.pdf        -- forest plot of bootstrap 95% CIs
  fig_heterogeneity.pdf            -- same-family vs cross-family on HM
  fig_intervention_change.pdf      -- per-intervention change rate (math only)

Design language matches experiments/render_figures.py:
  - Muted indigo PRIMARY + warm orange WARM as the two contrast colors
  - constrained_layout, generous figsize, no in-figure titles (LaTeX captions)
  - Bold value annotations with white halo
  - Soft grid, hairline spines
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "results" / "derived"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---- Palette (matches experiments/render_figures.py) ----
PRIMARY      = "#7E8FE0"  # softer indigo (VOUCH / dual-side)
PRIMARY_DARK = "#5763C3"
PRIMARY_FILL = "#F1F4FB"
ACCENT       = "#D88BB4"
WARM         = "#E9A87E"  # muted orange (single-side / baseline)
WARM_FILL    = "#FDF5EE"
GOOD         = "#7FB995"  # muted emerald
BAD          = "#D88080"
BAD_FILL     = "#FAEDED"
INK          = "#1F2937"
MUTED        = "#6B7280"
HAIRLINE     = "#D1D5DB"
PAPER        = "#FFFFFF"


PLT_STYLE = {
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans", "Arial", "Helvetica"],
    "font.weight":          "regular",
    "axes.titlesize":       11.0,
    "axes.titleweight":     "semibold",
    "axes.labelsize":       11,
    "axes.labelweight":     "regular",
    "axes.labelcolor":      INK,
    "axes.titlecolor":      INK,
    "axes.edgecolor":       HAIRLINE,
    "axes.linewidth":       0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.labelpad":        8,
    "axes.titlepad":        10,
    "xtick.labelsize":      9.5,
    "ytick.labelsize":      9.5,
    "xtick.color":          MUTED,
    "ytick.color":          MUTED,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.major.pad":      5,
    "ytick.major.pad":      5,
    "legend.fontsize":      10.0,
    "legend.frameon":       False,
    "legend.handlelength":  1.3,
    "legend.handletextpad": 0.6,
    "legend.borderpad":     0.2,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.10,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}
for k, v in PLT_STYLE.items():
    mpl.rcParams[k] = v


def _bold(ax, x, y, text, color=INK, fontsize=9.5, ha="center", va="bottom"):
    t = ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
                color=color, fontweight="bold")
    t.set_path_effects([
        path_effects.Stroke(linewidth=2.0, foreground=PAPER),
        path_effects.Normal(),
    ])


def _load_csv(name: str) -> list[dict]:
    return list(csv.DictReader((DERIVED / name).open()))


CELLS = [
    ("GSM8K", "clean"), ("GSM8K", "adv"),
    ("SVAMP", "clean"), ("SVAMP", "adv"),
    ("MATH",  "clean"), ("MATH",  "adv"),
    ("MBPP",  "clean"), ("MBPP",  "adv"),
    ("HumanEval", "clean"), ("HumanEval", "adv"),
]

DATASET_KEY = {
    "GSM8K-alg": "GSM8K", "GSM8K": "GSM8K",
    "SVAMP": "SVAMP",
    "MATH-alg": "MATH", "MATH": "MATH",
    "MBPP": "MBPP", "HumanEval": "HumanEval",
}

DATASET_DISPLAY = {
    "GSM8K": "GSM8K", "SVAMP": "SVAMP", "MATH": "MATH",
    "MBPP": "MBPP", "HumanEval": "Human\nEval",
}


# =========================================================================
# Figure 1: Abstention disentanglement
# Top    — action rate per cell (single value; both constructions match)
# Bottom — hurt | acted-on-wrong, single vs dual paired bars
# =========================================================================


def fig_abstention() -> Path:
    rows = _load_csv("abstention_disentangled.csv")
    by_cell = {(DATASET_KEY[r["dataset"]], r["regime"]): r for r in rows}
    cells = [c for c in CELLS if c in by_cell]

    labels = [f"{DATASET_DISPLAY[d]}\n{r}" for (d, r) in cells]
    act    = [float(by_cell[c]["action_rate"]) for c in cells]
    s_hurt = [float(by_cell[c]["single_hurt_per_acted_wrong"]) for c in cells]
    d_hurt = [float(by_cell[c]["dual_hurt_per_acted_wrong"])   for c in cells]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.6, 6.4),
        gridspec_kw=dict(height_ratios=[1.0, 1.55], hspace=0.45),
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.10)
    x = np.arange(len(cells))

    # Top: action rate per cell.
    ax_top.bar(x, act, width=0.55, color=PRIMARY, alpha=0.95,
               edgecolor=PAPER, linewidth=1.2, zorder=3)
    for i, v in enumerate(act):
        _bold(ax_top, x[i], v + 0.012, f"{int(round(v*100))}%",
              color=INK, fontsize=10)
    ax_top.set_ylim(0.60, 1.05)
    ax_top.set_yticks([0.60, 0.70, 0.80, 0.90, 1.00])
    ax_top.set_yticklabels(["60%", "70%", "80%", "90%", "100%"])
    ax_top.set_ylabel("Action rate", fontsize=11.5)
    ax_top.set_title("(a) Action rate is essentially identical across constructions",
                     loc="left", fontsize=11.0)
    ax_top.set_xticks(x); ax_top.set_xticklabels([""] * len(cells))
    ax_top.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.5,
                color=HAIRLINE, alpha=0.8, zorder=1)
    ax_top.tick_params(axis="both", which="both", length=0)
    ax_top.spines["left"].set_color(HAIRLINE)
    ax_top.spines["bottom"].set_color(HAIRLINE)
    ax_top.set_xlim(-0.7, len(cells) - 0.3)

    # Bottom: hurt|acted-on-wrong, single vs dual.
    w = 0.36
    b1 = ax_bot.bar(x - w/2, s_hurt, w, label="Single-side",
                    color=WARM, alpha=0.97, edgecolor=PAPER, linewidth=1.2,
                    zorder=3)
    b2 = ax_bot.bar(x + w/2, d_hurt, w, label="Dual-side (VOUCH)",
                    color=PRIMARY, alpha=0.98, edgecolor=PAPER, linewidth=1.2,
                    zorder=3)
    for bars, vals in [(b1, s_hurt), (b2, d_hurt)]:
        for b, v in zip(bars, vals):
            _bold(ax_bot, b.get_x() + b.get_width()/2, v + 0.018,
                  f"{int(round(v*100))}%", color=INK, fontsize=9.0)

    ymax = max(max(s_hurt), max(d_hurt)) * 1.22
    ax_bot.set_ylim(0, ymax)
    yt = np.arange(0.0, ymax, 0.20)
    ax_bot.set_yticks(yt)
    ax_bot.set_yticklabels([f"{int(t*100)}%" for t in yt])
    ax_bot.set_ylabel("Hurt | acted on wrong proposer\n(lower is better)",
                      fontsize=11.5)
    ax_bot.set_title("(b) Dual-side cuts harmful endorsements on every cell — "
                     "this is discrimination, not abstention",
                     loc="left", fontsize=11.0)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax_bot.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.5,
                color=HAIRLINE, alpha=0.8, zorder=1)
    ax_bot.tick_params(axis="both", which="both", length=0)
    ax_bot.spines["left"].set_color(HAIRLINE)
    ax_bot.spines["bottom"].set_color(HAIRLINE)
    ax_bot.legend(loc="upper left", frameon=True, fancybox=True,
                  framealpha=0.96, edgecolor="none", facecolor=PAPER,
                  borderpad=0.6, handlelength=1.4, handletextpad=0.7,
                  fontsize=10.5)
    ax_bot.set_xlim(-0.7, len(cells) - 0.3)

    out = FIG_DIR / "fig_abstention_disentangled.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# =========================================================================
# Figure 2: Forest plot of bootstrap 95% CIs.
# =========================================================================


def fig_ci_hurt() -> Path:
    rows = [
        row for row in _load_csv("ci_bootstrap.csv")
        if not row["cell"].startswith("HumanEval")
    ]

    def _display(name: str) -> str:
        name = name.replace("-alg", "")
        if " " in name:
            ds, reg = name.split(" ", 1)
            return f"{ds} · {reg}"
        return name

    cells = [_display(r["cell"]) for r in rows]
    point = np.array([float(r["rel_reduction_point"]) for r in rows]) * 100
    lo    = np.array([float(r["rel_reduction_lo"])    for r in rows]) * 100
    hi    = np.array([float(r["rel_reduction_hi"])    for r in rows]) * 100

    order = np.argsort(-point)
    cells = [cells[i] for i in order]
    point = point[order]; lo = lo[order]; hi = hi[order]
    err = np.vstack([point - lo, hi - point])

    n = len(cells)
    fig, ax = plt.subplots(figsize=(7.8, 0.55 * n + 1.5),
                           constrained_layout=True)
    y = np.arange(n)
    ax.axvline(0, color=HAIRLINE, linewidth=0.9, zorder=1)

    for i, p in enumerate(point):
        color = GOOD if p > 0 else BAD
        ax.plot([0, p], [y[i], y[i]], color=color, linewidth=2.6,
                alpha=0.85, zorder=2)
    ax.errorbar(point, y, xerr=err, fmt="none", ecolor=MUTED,
                elinewidth=1.0, capsize=4, capthick=1.0, zorder=3)
    ax.scatter(point, y, s=70, color=PRIMARY, edgecolor=PAPER,
               linewidth=1.4, zorder=4)

    # Right edge: leave room for the value label after the longest whisker.
    rmax = hi.max()
    label_x = rmax + 8
    xlim_right = label_x + 26
    for i in range(n):
        _bold(ax, label_x, y[i],
              f"{point[i]:.1f}%  [{lo[i]:.1f}, {hi[i]:.1f}]",
              color=INK, fontsize=9.5, ha="left", va="center")

    ax.set_yticks(y); ax.set_yticklabels(cells, fontsize=10.5, color=INK)
    ax.set_xlim(-12, xlim_right)
    # Only tick the data region, not the label gutter.
    xticks = [t for t in np.arange(0, rmax + 1, 20) if t <= rmax + 2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(t)}" for t in xticks])
    ax.set_xlabel("Relative reduction in Hurt|Act  "
                  r"($1 - r_{\mathrm{dual}}/r_{\mathrm{single}}$, %)",
                  fontsize=11.5)
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["left"].set_color(HAIRLINE)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.grid(axis="x", linestyle=(0, (3, 3)), linewidth=0.5,
            color=HAIRLINE, alpha=0.7, zorder=1)
    # Direction hint above the top row, well clear of x-axis label.
    ax.text(rmax / 2, -0.85, "favors dual-side →",
            ha="center", va="center", fontsize=9.5,
            color=MUTED, style="italic")

    out = FIG_DIR / "fig_ci_hurt_reduction.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# =========================================================================
# Figure 3: Heterogeneity — same vs cross family on HM, both regimes.
# =========================================================================


def fig_heterogeneity() -> Path:
    rows = _load_csv("heterogeneity_panel.csv")
    rows = [r for r in rows if r.get("samefamily_accuracy")
            and r.get("crossfamily_accuracy")]
    if not rows:
        print("heterogeneity: no rows; skipping")
        return None

    cond_display = {
        "independent":             "Indep.",
        "ensemble":                "Ensemble",
        "centralized_single":      "Central.\nsingle",
        "centralized_coordinator": "Central.\ncoord.",
        "contracts":               "VOUCH\ncontracts",
        "contracts_random":        "VOUCH\nrandom",
        "contracts_adaptive":      "VOUCH\nadaptive",
    }
    cond_order = [
        "independent", "ensemble",
        "centralized_single", "centralized_coordinator",
        "contracts", "contracts_random", "contracts_adaptive",
    ]
    regimes = ["clean", "adv"]
    by_key = {(r["dataset"], r["regime"], r["condition"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4),
                             sharey=True, constrained_layout=True)
    handles_for_legend = None

    for ax, reg in zip(axes, regimes):
        conds = [c for c in cond_order if ("HM", reg, c) in by_key]
        if not conds:
            ax.set_visible(False); continue
        x = np.arange(len(conds))
        w = 0.36
        same  = [float(by_key[("HM", reg, c)]["samefamily_accuracy"])  for c in conds]
        cross = [float(by_key[("HM", reg, c)]["crossfamily_accuracy"]) for c in conds]
        same_n  = int(by_key[("HM", reg, conds[0])]["samefamily_n_total"])
        cross_n = int(by_key[("HM", reg, conds[0])]["crossfamily_n_total"])

        b1 = ax.bar(x - w/2, same, w, color=WARM, alpha=0.97,
                    edgecolor=PAPER, linewidth=1.2,
                    label="Same-family (Qwen+Qwen)")
        b2 = ax.bar(x + w/2, cross, w, color=PRIMARY, alpha=0.97,
                    edgecolor=PAPER, linewidth=1.2,
                    label="Cross-family (Llama+Qwen)")
        if handles_for_legend is None:
            handles_for_legend = [b1, b2]
        for b, v in zip(b1, same):
            _bold(ax, b.get_x() + b.get_width()/2, v + 0.010,
                  f"{int(round(v*100))}", color=INK, fontsize=8.5)
        for b, v in zip(b2, cross):
            _bold(ax, b.get_x() + b.get_width()/2, v + 0.010,
                  f"{int(round(v*100))}", color=INK, fontsize=8.5)

        ax.set_title(f"HM · {reg}    "
                     f"(same-family n={same_n}, cross-family n={cross_n})",
                     loc="left", fontsize=11.0)
        ax.set_xticks(x)
        ax.set_xticklabels([cond_display[c] for c in conds],
                           fontsize=9.0, color=INK)
        ax.set_ylim(0, 0.82)
        ax.set_yticks(np.arange(0, 0.81, 0.20))
        ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0, 0.81, 0.20)])
        ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.5,
                color=HAIRLINE, alpha=0.8, zorder=0)
        ax.tick_params(axis="both", which="both", length=0)
        ax.spines["left"].set_color(HAIRLINE)
        ax.spines["bottom"].set_color(HAIRLINE)

    axes[0].set_ylabel("Accuracy", fontsize=11.5)
    if handles_for_legend is not None:
        fig.legend(handles_for_legend,
                   ["Same-family (Qwen+Qwen)", "Cross-family (Llama+Qwen)"],
                   loc="upper center", bbox_to_anchor=(0.5, 1.09),
                   ncol=2, frameon=False, fontsize=10.5)

    out = FIG_DIR / "fig_heterogeneity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# =========================================================================
# Figure 4: Per-intervention change rate (math; code is degenerate).
# =========================================================================


INTERV_DISPLAY = {
    "perturb_number_*10":  "perturb number  ($\\times$10)",
    "perturb_number_+1":   "perturb number  (+1)",
    "swap_problem":        "swap problem",
    "remove_clause_1":     "remove clause 1",
    "add_distractor":      "add distractor",
    "paraphrase":          "paraphrase",
}


def fig_intervention_change() -> Path:
    rows = _load_csv("intervention_informativeness.csv")
    rows = [r for r in rows if r["domain"] == "math"
            and r["intervention"] != "<none>"
            and r["mean_change_rate"]]
    if not rows:
        print("intervention_change: no rows; skipping")
        return None

    datasets = ["GSM8K-alg", "MATH-alg", "SVAMP"]
    dataset_display = {"GSM8K-alg": "GSM8K", "MATH-alg": "MATH", "SVAMP": "SVAMP"}

    # Global ordering by mean change rate (largest at top).
    global_mean = defaultdict(list)
    for r in rows:
        global_mean[r["intervention"]].append(float(r["mean_change_rate"]))
    interv_order = sorted(global_mean.keys(),
                          key=lambda k: -sum(global_mean[k]) / len(global_mean[k]))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8),
                             sharex=True, sharey=True, constrained_layout=True)

    for ax, ds in zip(axes, datasets):
        ds_rows = {r["intervention"]: r for r in rows if r["dataset"] == ds}
        vals = [float(ds_rows[iv]["mean_change_rate"]) if iv in ds_rows else 0
                for iv in interv_order]
        ns   = [int(ds_rows[iv]["n"]) if iv in ds_rows else 0
                for iv in interv_order]
        y = np.arange(len(interv_order))
        bars = ax.barh(y, vals, color=PRIMARY, alpha=0.95,
                       edgecolor=PAPER, linewidth=1.2, height=0.62, zorder=3)
        for b, v in zip(bars, vals):
            if v > 0:
                _bold(ax, v + 0.02, b.get_y() + b.get_height()/2,
                      f"{int(round(v*100))}%",
                      color=INK, fontsize=9.5, ha="left", va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([INTERV_DISPLAY[i] for i in interv_order],
                           fontsize=10.0, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_xticklabels([f"{int(t*100)}%" for t in np.arange(0, 1.01, 0.25)])
        ax.set_title(f"{dataset_display[ds]}   (n={ns[0] if ns[0] else 100} / intervention)",
                     loc="left", fontsize=11.0)
        ax.grid(axis="x", linestyle=(0, (3, 3)), linewidth=0.5,
                color=HAIRLINE, alpha=0.7, zorder=0)
        ax.tick_params(axis="both", which="both", length=0)
        ax.spines["left"].set_color(HAIRLINE)
        ax.spines["bottom"].set_color(HAIRLINE)

    axes[1].set_xlabel(r"P(answer changes $\mid$ intervention)   "
                       "— higher means more informative",
                       fontsize=11.5)

    out = FIG_DIR / "fig_intervention_change.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    fig_abstention()
    fig_ci_hurt()
    fig_heterogeneity()
    fig_intervention_change()


if __name__ == "__main__":
    main()
