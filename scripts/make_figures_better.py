"""Regenerate the paper figures in an alternative house style.

Style source: https://github.com/ChenLiu-1996/figures4papers
  (scientific-figure-making guidelines and references)

House-style rules:
  - PALETTE (semantic): blue_main = proposed, red_strong = baselines,
    green_3 = improvements, neutral = reference, highlight = single call-out.
  - apply_publication_style: top/right spines off, frameless legend,
    font.size = 22-24 for bar panels (15-16 for compact), axes.linewidth = 3
    for big bars (2 for compact), pdf.fonttype=42, svg.fonttype='none'.
  - finalize_figure: save .png AND .pdf, dpi=300 (600 for dense bars),
    tight_layout(pad=2), parents created.
  - Ultra-wide aspect ratios (width 3-4x height) for multi-metric panels.
  - Dedicated legend panel: last subplot with set_axis_off() carries the
    legend handles harvested from data panels.
  - Categorical bars without x-tick labels (legend names categories).
  - Dynamic y-axis scaling: tight limits so differences are visible.
  - Print-safe: bar edgecolor='black', linewidth=1.5-3, optional hatch.

Outputs into figures_better/, next to the default-style figures that
experiments/render_figures.py and scripts/make_revision_figures.py write
into figures/. The paper uses figures_better/ for the per-intervention
panels (fig_intervention_I, fig_intervention_change).

All numbers come from results/derived/*.csv (which are deterministically
derived from results/*.jsonl) and from results/analysis/summary.json
(a deterministic summary of the same JSONLs). The calibration panel is
skipped unless results/analysis/calibration.json exists.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# House style (figures4papers conventions)
# ============================================================

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1":        "#DDF3DE",
    "green_2":        "#AADCA9",
    "green_3":        "#8BCF8B",
    "red_1":          "#F6CFCB",
    "red_2":          "#E9A6A1",
    "red_strong":     "#B64342",
    "neutral":        "#CFCECE",
    "highlight":      "#FFD700",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
}

# figures4papers demos lean on the LIGHT half of the palette (green_2,
# red_2, blue_secondary, plus accent teal / violet). We do the same and
# additionally lighten the cobalt blue to keep the page airy.
PROPOSED      = "#6E9AD6"                   # soft cobalt (proposed / VOUCH)
PROPOSED_DEEP = PALETTE["blue_secondary"]   # #3775BA — code emphasis only
PROPOSED_PALE = "#B6CCE8"                   # very light blue
GOOD          = PALETTE["green_2"]          # #AADCA9
GOOD_DEEP     = PALETTE["green_3"]          # #8BCF8B
BASELINE      = "#F4B68A"                   # light peach (was red_2)
BASELINE_2    = "#D27A66"                   # warm terracotta (lighter than red_strong)
TEAL          = PALETTE["teal"]             # #42949E (code accent)
TEAL_LIGHT    = "#8FC0C7"
VIOLET        = PALETTE["violet"]           # #9A4D8E
VIOLET_LIGHT  = "#C49AB7"
GOLD          = PALETTE["highlight"]        # #FFD700
NEUTRAL     = PALETTE["neutral"]
HIGHLIGHT   = PALETTE["highlight"]
INK         = "#1F2937"
MUTED       = "#6B7280"
HAIRLINE    = "#D1D5DB"
PAPER       = "#FFFFFF"


def apply_publication_style(font_size: int = 26, axes_linewidth: float = 2.8):
    mpl.rcParams.update({
        "font.family":          ["DejaVu Sans", "Helvetica", "Arial",
                                 "sans-serif"],
        "font.size":            font_size,
        "axes.titlesize":       font_size + 2,
        "axes.labelsize":       font_size + 2,
        "axes.labelweight":     "semibold",
        "axes.labelcolor":      INK,
        "axes.edgecolor":       INK,
        "axes.linewidth":       axes_linewidth,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.titlepad":        12,
        "xtick.color":          INK,
        "ytick.color":          INK,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.width":    axes_linewidth * 0.6,
        "ytick.major.width":    axes_linewidth * 0.6,
        "xtick.labelsize":      font_size - 1,
        "ytick.labelsize":      font_size - 1,
        "legend.frameon":       False,
        "legend.fontsize":      font_size - 1,
        "legend.handlelength":  1.4,
        "legend.handletextpad": 0.5,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
        "text.usetex":          False,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
        "svg.fonttype":         "none",
    })


def finalize(fig, out_basename: Path, dpi: int = 300):
    """Save the figure as both .png and .pdf. Figures are created with
    constrained_layout=True so we do not call tight_layout here (which
    would warn about the dedicated legend panel using set_axis_off)."""
    out_basename.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_basename.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out_basename.name}.{{png,pdf}}")


def _halo(text_artist):
    text_artist.set_path_effects([
        path_effects.Stroke(linewidth=2.5, foreground=PAPER),
        path_effects.Normal(),
    ])


# ============================================================
# Data loading
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "results" / "derived"
ANALYSIS_DIR = ROOT / "results" / "analysis"
OUT_DIR = ROOT / "figures_better"


def _load_csv(name: str) -> list[dict]:
    return list(csv.DictReader((DERIVED / name).open()))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.load(path.open())


CELLS = [
    ("GSM8K", "clean"), ("GSM8K", "adv"),
    ("SVAMP", "clean"), ("SVAMP", "adv"),
    ("MATH",  "clean"), ("MATH",  "adv"),
    ("MBPP",  "clean"), ("MBPP",  "adv"),
    ("HumanEval", "clean"), ("HumanEval", "adv"),
]

DATASET_KEY = {
    "GSM8K-alg": "GSM8K", "GSM8K": "GSM8K", "SVAMP": "SVAMP",
    "MATH-alg": "MATH", "MATH": "MATH",
    "MBPP": "MBPP", "HumanEval": "HumanEval",
}

LABEL = {"GSM8K": "GSM8K", "SVAMP": "SVAMP", "MATH": "MATH",
         "MBPP": "MBPP", "HumanEval": "HumanEval"}


# ============================================================
# Figure 1 — Abstention disentangled
#
# Two stacked panels (Catch | Hurt). Action rate is folded into the
# x-tick label of each cell as a small percentage badge — no third panel
# is needed. Adversarial regime is shown with a hatch overlay so the
# clean / adv distinction is print-safe without doubling the hue count.
#
# Palette is intentionally pale (washed peach + washed cobalt + washed
# teal) — the contract panels in figures4papers (e.g. brainteaser) use a
# similar low-saturation register. No bold titles, no in-panel italic
# annotations, no math/code vertical separator — let the data breathe.
# ============================================================


# Light, washed-out palette specific to this figure
ABS_PEACH      = "#F4C8A8"   # math single-side
ABS_COBALT     = "#A8C8E8"   # math dual-side (proposed)
ABS_TERRACOTTA = "#E69C82"   # code single-side
ABS_TEAL       = "#85B9C2"   # code dual-side


def fig_abstention(out_dir: Path) -> None:
    rows = _load_csv("abstention_disentangled.csv")
    by_cell = {(DATASET_KEY[r["dataset"]], r["regime"]): r for r in rows}
    cells = [c for c in CELLS if c in by_cell]

    act    = np.array([float(by_cell[c]["action_rate"]) for c in cells])
    s_hurt = np.array([float(by_cell[c]["single_hurt_per_acted_wrong"]) for c in cells])
    d_hurt = np.array([float(by_cell[c]["dual_hurt_per_acted_wrong"])   for c in cells])
    s_catch = np.array([float(by_cell[c]["single_catch_per_acted_wrong"]) for c in cells])
    d_catch = np.array([float(by_cell[c]["dual_catch_per_acted_wrong"])   for c in cells])

    code_idx = np.array([d in ("MBPP", "HumanEval") for (d, _) in cells])
    adv_idx  = np.array([r == "adv"               for (_, r) in cells])

    single_face = [ABS_TERRACOTTA if c else ABS_PEACH  for c in code_idx]
    dual_face   = [ABS_TEAL       if c else ABS_COBALT for c in code_idx]
    hatch_for   = ["////"         if a else ""         for a in adv_idx]

    # Single-line x-tick label, rotated, plus action-rate badge.
    tick_labels = [
        f"{LABEL[d]} {r}\nact {int(round(a*100))}%"
        for (d, r), a in zip(cells, act)
    ]

    apply_publication_style(font_size=20, axes_linewidth=2)
    fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.6, 4.0, 4.0],
                          hspace=0.45)

    x = np.arange(len(cells))
    w = 0.36

    # --- legend strip (top) ----------------------------------------------
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_leg.set_axis_off()
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ABS_PEACH,     edgecolor="0.4", linewidth=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor=ABS_COBALT,    edgecolor="0.4", linewidth=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor=ABS_TERRACOTTA, edgecolor="0.4", linewidth=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor=ABS_TEAL,      edgecolor="0.4", linewidth=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor="white",       edgecolor="0.4", linewidth=1.0, hatch="////"),
    ]
    labels_leg = [
        "single-side (math)",
        "dual-side / VOUCH (math)",
        "single-side (code)",
        "dual-side / VOUCH (code)",
        "adversarial regime",
    ]
    ax_leg.legend(handles, labels_leg, loc="center", ncol=5,
                  fontsize=16, frameon=False, columnspacing=1.8,
                  handlelength=1.4, handletextpad=0.7, borderaxespad=0)

    def _paired(ax, single, dual, ylim, yticks, ytl, ylabel, title,
                show_xlabels):
        ax.bar(x - w/2, single, w, color=single_face,
               edgecolor="0.3", linewidth=1.0, hatch=hatch_for)
        ax.bar(x + w/2, dual, w, color=dual_face,
               edgecolor="0.3", linewidth=1.0, hatch=hatch_for)
        span = ylim[1] - ylim[0]
        for xi, v in zip(x - w/2, single):
            ax.text(xi, v + span * 0.022, f"{int(round(v*100))}",
                    ha="center", va="bottom", fontsize=11, color=INK)
        for xi, v in zip(x + w/2, dual):
            ax.text(xi, v + span * 0.022, f"{int(round(v*100))}",
                    ha="center", va="bottom", fontsize=11, color=INK)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks); ax.set_yticklabels(ytl)
        ax.set_ylabel(ylabel, fontsize=18)
        # plain, non-bold, lowercase title at left
        ax.set_title(title, loc="left", fontsize=16,
                     fontweight="regular", color=MUTED, pad=6)
        ax.set_xticks(x)
        if show_xlabels:
            ax.set_xticklabels(tick_labels, fontsize=11, color=INK,
                               rotation=0)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=4, width=1.5)

    ax_catch = fig.add_subplot(gs[1, 0])
    _paired(ax_catch, s_catch, d_catch,
            ylim=(0.55, 1.02),
            yticks=[0.60, 0.70, 0.80, 0.90, 1.00],
            ytl=["60", "70", "80", "90", "100"],
            ylabel="catch | acted on wrong  (%)",
            title="catch rate — higher is better",
            show_xlabels=False)

    ymax = max(s_hurt.max(), d_hurt.max()) * 1.25
    yt = np.arange(0.0, ymax, 0.10)
    ax_hurt = fig.add_subplot(gs[2, 0])
    _paired(ax_hurt, s_hurt, d_hurt,
            ylim=(0, ymax),
            yticks=yt, ytl=[f"{int(t*100)}" for t in yt],
            ylabel="hurt | acted on wrong  (%)",
            title="hurt rate — lower is better",
            show_xlabels=True)

    finalize(fig, out_dir / "fig_abstention_disentangled", dpi=600)


# ============================================================
# Figure 2 — Bootstrap CI forest plot
# ============================================================


def fig_ci_hurt(out_dir: Path) -> None:
    rows = _load_csv("ci_bootstrap.csv")

    def _disp(name: str) -> str:
        name = name.replace("-alg", "")
        if " " in name:
            ds, reg = name.split(" ", 1)
            return f"{ds}  ·  {reg}"
        return name

    cells = [_disp(r["cell"]) for r in rows]
    point = np.array([float(r["rel_reduction_point"]) for r in rows]) * 100
    lo    = np.array([float(r["rel_reduction_lo"])    for r in rows]) * 100
    hi    = np.array([float(r["rel_reduction_hi"])    for r in rows]) * 100

    order = np.argsort(-point)
    cells = [cells[i] for i in order]
    point = point[order]; lo = lo[order]; hi = hi[order]
    err = np.vstack([point - lo, hi - point])

    apply_publication_style(font_size=22, axes_linewidth=3)
    n = len(cells)
    fig, ax = plt.subplots(figsize=(14, 0.70 * n + 2.4),
                           constrained_layout=True)
    y = np.arange(n)

    ax.axvline(0, color="black", linewidth=1.5, zorder=1)
    for i, p in enumerate(point):
        color = GOOD if p > 0 else BASELINE_2
        ax.plot([0, p], [y[i], y[i]], color=color, linewidth=6.0,
                solid_capstyle="round", alpha=0.95, zorder=2)
    ax.errorbar(point, y, xerr=err, fmt="none", ecolor="black",
                elinewidth=2.0, capsize=8, capthick=2.0, zorder=3)
    ax.scatter(point, y, s=240, color=PROPOSED, edgecolor="black",
               linewidth=2.0, zorder=4)
    rmax = hi.max()
    label_x = rmax + 8
    xlim_right = label_x + 38
    for i in range(n):
        ax.text(label_x, y[i],
                f"{point[i]:.1f}%   [{lo[i]:.1f}, {hi[i]:.1f}]",
                ha="left", va="center", fontsize=20, color=INK,
                fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(cells, fontsize=22, color=INK)
    ax.set_xlim(-12, xlim_right)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0", "20", "40", "60", "80"])
    ax.set_xlabel("Relative reduction in Hurt | acted on wrong  (%)",
                  fontsize=24)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=5, width=2)
    ax.text(rmax / 2, -0.9, "favors dual-side  →",
            ha="center", va="center", fontsize=20, color=PROPOSED,
            fontweight="bold")

    finalize(fig, out_dir / "fig_ci_hurt_reduction", dpi=600)


# ============================================================
# Figure 3 — Heterogeneity (HM same-family vs cross-family)
# ============================================================


def fig_heterogeneity(out_dir: Path) -> None:
    rows = _load_csv("heterogeneity_panel.csv")
    rows = [r for r in rows if r.get("samefamily_accuracy")
            and r.get("crossfamily_accuracy")]
    if not rows:
        print("heterogeneity: no rows; skipping"); return

    cond_disp = {
        "independent":             "Indep.",
        "ensemble":                "Ensem.",
        "centralized_single":      "Cent.\nsingle",
        "centralized_coordinator": "Cent.\ncoord.",
        "contracts":               "VOUCH\nfixed",
        "contracts_random":        "VOUCH\nrand.",
        "contracts_adaptive":      "VOUCH\nadap.",
    }
    cond_order = [
        "independent", "ensemble",
        "centralized_single", "centralized_coordinator",
        "contracts", "contracts_random", "contracts_adaptive",
    ]
    by_key = {(r["dataset"], r["regime"], r["condition"]): r for r in rows}

    apply_publication_style(font_size=22, axes_linewidth=3)
    fig = plt.figure(figsize=(28, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.42], wspace=0.22)

    for col, reg in enumerate(["clean", "adv"]):
        ax = fig.add_subplot(gs[0, col])
        conds = [c for c in cond_order if ("HM", reg, c) in by_key]
        x = np.arange(len(conds))
        w = 0.38
        same  = [float(by_key[("HM", reg, c)]["samefamily_accuracy"])  for c in conds]
        cross = [float(by_key[("HM", reg, c)]["crossfamily_accuracy"]) for c in conds]
        same_n  = int(by_key[("HM", reg, conds[0])]["samefamily_n_total"])
        cross_n = int(by_key[("HM", reg, conds[0])]["crossfamily_n_total"])

        ax.bar(x - w/2, same, w, color=BASELINE, edgecolor="black",
               linewidth=1.6)
        ax.bar(x + w/2, cross, w, color=TEAL, edgecolor="black",
               linewidth=1.6)
        for xi, v in zip(x - w/2, same):
            ax.text(xi, v + 0.012, f"{int(round(v*100))}",
                    ha="center", va="bottom", fontsize=15, color=INK)
        for xi, v in zip(x + w/2, cross):
            ax.text(xi, v + 0.012, f"{int(round(v*100))}",
                    ha="center", va="bottom", fontsize=15, color=INK,
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([cond_disp[c] for c in conds],
                           fontsize=16, color=INK)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 0.85)
        ax.set_yticks([0.0, 0.20, 0.40, 0.60, 0.80])
        ax.set_yticklabels(["0", "20", "40", "60", "80"])
        if col == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=24)
        ax.set_title(
            f"HM  ·  {reg}\n"
            f"same-family n={same_n} · cross-family n={cross_n}",
            loc="left", fontsize=20, fontweight="bold", color=INK)

    ax_leg = fig.add_subplot(gs[0, 2])
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE, edgecolor="black", linewidth=1.6),
        plt.Rectangle((0, 0), 1, 1, facecolor=TEAL,     edgecolor="black", linewidth=1.6),
    ]
    ax_leg.legend(handles,
                  ["Same-family\n(Qwen + Qwen)",
                   "Cross-family\n(Llama + Qwen)"],
                  loc="center left", fontsize=22, frameon=False,
                  handlelength=1.4, handletextpad=0.8, borderaxespad=0)
    ax_leg.set_axis_off()

    finalize(fig, out_dir / "fig_heterogeneity", dpi=600)


# ============================================================
# Figure 4 — Per-intervention change rate (math; code is degenerate)
# ============================================================


INTERV_DISP = {
    "perturb_number_*10":  "perturb number  (×10)",
    "perturb_number_+1":   "perturb number  (+1)",
    "swap_problem":        "swap problem",
    "remove_clause_1":     "remove clause 1",
    "add_distractor":      "add distractor",
    "paraphrase":          "paraphrase",
}


def fig_intervention_change(out_dir: Path) -> None:
    rows = _load_csv("intervention_informativeness.csv")
    rows = [r for r in rows if r["domain"] == "math"
            and r["intervention"] != "<none>"
            and r["mean_change_rate"]]
    if not rows:
        print("intervention_change: no rows; skipping"); return

    datasets = ["GSM8K-alg", "MATH-alg", "SVAMP"]
    ds_disp  = {"GSM8K-alg": "GSM8K", "MATH-alg": "MATH", "SVAMP": "SVAMP"}

    global_mean = defaultdict(list)
    for r in rows:
        global_mean[r["intervention"]].append(float(r["mean_change_rate"]))
    interv_order = sorted(global_mean.keys(),
                          key=lambda k: -sum(global_mean[k]) / len(global_mean[k]))

    alphas = np.linspace(1.0, 0.35, len(interv_order))
    # Alpha-graded violet for ordered ranking — distinct from the blue
    # used everywhere else for the proposed method.
    violet_rgb = (154/255, 77/255, 142/255)
    colors = [(*violet_rgb, a) for a in alphas]

    apply_publication_style(font_size=22, axes_linewidth=3)
    fig, axes = plt.subplots(1, 3, figsize=(28, 6.0),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    for ax, ds in zip(axes, datasets):
        ds_rows = {r["intervention"]: r for r in rows if r["dataset"] == ds}
        vals = [float(ds_rows[iv]["mean_change_rate"]) if iv in ds_rows else 0.0
                for iv in interv_order]
        n0 = int(ds_rows[interv_order[0]]["n"]) if interv_order[0] in ds_rows else 100
        y = np.arange(len(interv_order))
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=1.6,
                height=0.66)
        for yi, v in zip(y, vals):
            if v > 0:
                ax.text(v + 0.014, yi, f"{int(round(v*100))}",
                        ha="left", va="center", fontsize=20, color=INK,
                        fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([INTERV_DISP[i] for i in interv_order],
                           fontsize=20, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        ax.set_title(f"{ds_disp[ds]}   (n={n0} / intervention)",
                     loc="left", fontsize=24, fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=5, width=2)

    axes[1].set_xlabel("P(answer changes | intervention)   (%)",
                       fontsize=24)
    finalize(fig, out_dir / "fig_intervention_change", dpi=600)


# ============================================================
# Figure 5 — Hurt:caught ratio (headline)
# ============================================================


def fig_hurt_caught(out_dir: Path) -> None:
    benchmarks  = ["MBPP\nclean", "MBPP\nadv.",
                   "HumanEval\nclean", "HumanEval\nadv."]
    single_side = np.array([0.575, 0.640, 2.067, 1.625])
    dual_side   = np.array([0.145, 0.093, 1.875, 1.520])

    apply_publication_style(font_size=22, axes_linewidth=3)
    fig = plt.figure(figsize=(16, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.32], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(benchmarks))
    w = 0.36
    ax.bar(x - w/2, single_side, w, color=BASELINE, edgecolor="black",
           linewidth=1.8)
    ax.bar(x + w/2, dual_side, w, color=PROPOSED, edgecolor="black",
           linewidth=1.8)
    for xi, v in zip(x - w/2, single_side):
        ax.text(xi, v + 0.04, f"{v:.2f}",
                ha="center", va="bottom", fontsize=18, color=INK)
    for xi, v in zip(x + w/2, dual_side):
        ax.text(xi, v + 0.04, f"{v:.2f}",
                ha="center", va="bottom", fontsize=18, color=INK,
                fontweight="bold")

    ax.axhline(1.0, color=BASELINE_2, linewidth=2.0,
               linestyle=(0, (5, 3)), alpha=0.9)
    t = ax.text(-0.45, 1.05, "break-even  (hurt : caught = 1)",
                ha="left", va="bottom", fontsize=18,
                color=BASELINE_2, fontweight="bold", style="italic")
    _halo(t)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=20, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 2.70)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylabel("Hurt : caught ratio  (lower is better)", fontsize=22)
    ax.set_xlim(-0.6, len(benchmarks) - 0.4)

    ax_leg = fig.add_subplot(gs[0, 1])
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE, edgecolor="black", linewidth=1.6),
        plt.Rectangle((0, 0), 1, 1, facecolor=PROPOSED, edgecolor="black", linewidth=1.6),
    ]
    ax_leg.legend(handles,
                  ["Single-side\n(self-form CC)",
                   "Dual-side (VOUCH)"],
                  loc="center left", fontsize=22, frameon=False,
                  handlelength=1.4, handletextpad=0.8, borderaxespad=0)
    ax_leg.set_axis_off()

    finalize(fig, out_dir / "fig_hurt_caught", dpi=300)


# ============================================================
# Figure 6 — Calibration reliability diagram (self-CC vs confidence)
# ============================================================


def fig_calibration(out_dir: Path) -> None:
    cal = _load_json(ANALYSIS_DIR / "calibration.json")
    if not cal:
        print("calibration: no calibration.json; skipping"); return

    bins = np.linspace(0.05, 0.95, 10)
    conf  = cal["confidence"]
    ccse  = cal["cc_self"]

    def _xy(blob):
        acc = blob["bin_accuracy"]; cnt = blob["bin_count"]
        xs, ys, ns = [], [], []
        for b, a, n in zip(bins, acc, cnt):
            if a is not None and n > 0:
                xs.append(b); ys.append(a); ns.append(n)
        return np.array(xs), np.array(ys), np.array(ns)

    cx, cy, cn = _xy(conf)
    sx, sy, sn = _xy(ccse)

    # -- colours for this figure --
    CONF_COLOR = "#C0392B"       # strong red for confidence
    CC_COLOR   = "#1A5276"       # deep navy-teal for CC_self
    DIAG_COLOR = "#7F8C8D"       # medium gray for diagonal
    SHADE_CONF = "#E74C3C"       # lighter red for overconfidence fill

    apply_publication_style(font_size=16, axes_linewidth=1.8)
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 5.2),
        gridspec_kw={"width_ratios": [1.0, 0.42], "wspace": 0.35},
    )

    # ---- left panel: reliability diagram ----
    # perfect-calibration diagonal
    ax.plot([0, 1], [0, 1], color=DIAG_COLOR, linewidth=1.8,
            linestyle="--", zorder=1, label="_nolegend_")
    t = ax.text(0.68, 0.74, "perfect\ncalibration", ha="center", va="bottom",
                fontsize=11, color=DIAG_COLOR, style="italic", rotation=38)
    _halo(t)

    # confidence: bar-style reliability diagram (width = bin width)
    bin_width = 0.10
    # shade overconfidence region between bin centre and accuracy
    for xi, yi in zip(cx, cy):
        if xi > yi:  # overconfident
            ax.fill_between([xi - bin_width/2, xi + bin_width/2],
                            yi, xi, color=SHADE_CONF, alpha=0.12, zorder=1)

    ax.bar(cx, cy, width=bin_width * 0.85, color=CONF_COLOR, alpha=0.55,
           edgecolor=CONF_COLOR, linewidth=1.0, zorder=2,
           label="Self-confidence")

    # CC_self: single point (all mass in one bin)
    ax.scatter(sx, sy, s=280, color=CC_COLOR, edgecolor="white",
               linewidth=2.0, zorder=5, marker="D",
               label=r"$\widehat{CC}_{\mathrm{self}}$")

    # annotate the CC point
    cc_annot_x, cc_annot_y = sx[0], sy[0]
    ax.annotate(
        f"all {cal['n_predictions']:,}\npredictions\nin one bin",
        xy=(cc_annot_x, cc_annot_y),
        xytext=(cc_annot_x + 0.18, cc_annot_y + 0.22),
        fontsize=10, color=CC_COLOR, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=CC_COLOR,
                        lw=1.5, connectionstyle="arc3,rad=-0.15"),
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=CC_COLOR,
                  lw=1.0, alpha=0.92),
        zorder=6,
    )

    # overconfidence annotation
    # find the bin with greatest gap (cx[i] - cy[i])
    gaps = cx - cy
    worst_idx = int(np.argmax(gaps))
    mid_gap_y = (cx[worst_idx] + cy[worst_idx]) / 2
    t2 = ax.text(cx[worst_idx] + 0.06, mid_gap_y, "overconfident",
                 fontsize=10, color=SHADE_CONF, fontstyle="italic",
                 ha="left", va="center", fontweight="bold")
    _halo(t2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("Predicted probability (trust signal)", fontsize=15)
    ax.set_ylabel("Empirical accuracy", fontsize=15)
    ax.set_aspect("equal", adjustable="box")

    leg = ax.legend(loc="upper left", fontsize=12, frameon=True,
                    fancybox=True, framealpha=0.9, edgecolor=HAIRLINE,
                    handlelength=1.6, handletextpad=0.5, borderpad=0.5)
    leg.get_frame().set_linewidth(0.8)

    # ---- right panel: ECE and Brier bar comparison ----
    metrics = ["ECE", "Brier"]
    conf_vals = [conf["ece"], conf["brier"]]
    cc_vals   = [ccse["ece"], ccse["brier"]]

    x_pos = np.arange(len(metrics))
    bar_w = 0.32

    bars_conf = ax_bar.bar(x_pos - bar_w/2, conf_vals, bar_w,
                           color=CONF_COLOR, alpha=0.65,
                           edgecolor=CONF_COLOR, linewidth=0.8,
                           label="Self-confidence", zorder=2)
    bars_cc = ax_bar.bar(x_pos + bar_w/2, cc_vals, bar_w,
                         color=CC_COLOR, alpha=0.80,
                         edgecolor=CC_COLOR, linewidth=0.8,
                         label=r"$\widehat{CC}_{\mathrm{self}}$", zorder=2)

    # value labels on bars
    for bar, val in zip(bars_conf, conf_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=CONF_COLOR)
    for bar, val in zip(bars_cc, cc_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=CC_COLOR)

    # arrows showing improvement
    for i in range(len(metrics)):
        diff = conf_vals[i] - cc_vals[i]
        pct = diff / conf_vals[i] * 100
        mid_x = x_pos[i]
        mid_y = max(conf_vals[i], cc_vals[i]) + 0.10
        ax_bar.annotate("", xy=(x_pos[i] + bar_w/2, cc_vals[i] + 0.04),
                         xytext=(x_pos[i] - bar_w/2, conf_vals[i] + 0.04),
                         arrowprops=dict(arrowstyle="-|>", color="#27AE60",
                                         lw=2.0))
        ax_bar.text(mid_x, mid_y, f"−{pct:.0f}%",
                    ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color="#27AE60")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metrics, fontsize=14)
    ax_bar.set_ylabel("Score (lower is better)", fontsize=13)
    ax_bar.set_ylim(0, 0.82)
    ax_bar.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    leg2 = ax_bar.legend(loc="upper right", fontsize=11, frameon=True,
                         fancybox=True, framealpha=0.9, edgecolor=HAIRLINE,
                         borderpad=0.4)
    leg2.get_frame().set_linewidth(0.8)

    # panel labels
    ax.text(-0.08, 1.04, "(a)", transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=INK, va="top")
    ax_bar.text(-0.12, 1.04, "(b)", transform=ax_bar.transAxes,
                fontsize=16, fontweight="bold", color=INK, va="top")

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.95)
    finalize(fig, out_dir / "fig_calibration", dpi=300)


# ============================================================
# Figure 7 — Cross-domain pathology scatter
# ============================================================


def fig_cross_domain_signature(out_dir: Path) -> None:
    points = [
        ("MBPP",            -5.4,  3.1),
        ("HumanEval",       -7.3,  2.0),
        ("GSM8K",          -28.0,  4.2),
        ("MATH (algebra)",   0.6, 11.6),
        ("SVAMP",          -21.0,  5.5),
    ]
    label_offsets = {
        "MBPP":            ( 1.0,  0.55),
        "HumanEval":       ( 1.0, -1.25),
        "GSM8K":           ( 1.4,  0.55),
        "MATH (algebra)":  (-8.6,  0.70),
        "SVAMP":           ( 1.4,  0.55),
    }

    apply_publication_style(font_size=22, axes_linewidth=3)
    fig, ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
    ax.set_xlim(-34, 8)
    ax.set_ylim(-2, 15)

    ax.axhspan(0, 15, xmin=0, xmax=34/42, color=PALETTE["red_1"],
               alpha=0.55, zorder=0)
    ax.axhline(0, color=NEUTRAL, linewidth=1.2, zorder=1)
    ax.axvline(0, color=NEUTRAL, linewidth=1.2, zorder=1)

    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])

    ax.scatter(xs, ys, s=520, color=PROPOSED, alpha=0.18, zorder=2,
               edgecolor="none")
    ax.scatter(xs, ys, s=190, color=PROPOSED, zorder=3,
               edgecolor="black", linewidth=2.0)

    for (lab, x, y) in points:
        dx, dy = label_offsets[lab]
        t = ax.text(x + dx, y + dy, lab, fontsize=20, color=INK,
                    fontweight="bold")
        _halo(t)

    t = ax.text(-32, 13.7, "pathology quadrant",
                fontsize=20, color=BASELINE_2, fontweight="bold")
    _halo(t)
    t = ax.text(-32, 12.5, "capability falls, self-coherence rises",
                fontsize=16, color=BASELINE_2, style="italic")
    _halo(t)

    ax.set_xlabel(r"$\Delta$ accuracy  (pp): adversarial $-$ clean",
                  fontsize=22)
    ax.set_ylabel(r"$\Delta$ verification rate  (pp)",
                  fontsize=22)
    ax.tick_params(axis="both", length=5, width=2)

    finalize(fig, out_dir / "fig_cross_domain_signature", dpi=300)


# ============================================================
# Figure 8 — Per-agent self-CC vs cross-CC
# ============================================================


def fig_per_agent_cc(out_dir: Path) -> None:
    summary = _load_json(ANALYSIS_DIR / "summary.json")
    if not summary:
        print("per_agent_cc: no summary; skipping"); return

    per_agent = {}
    for regs in summary.get("per_dataset", {}).values():
        for blob in regs.values():
            for agent, stats in (blob.get("cc_per_agent") or {}).items():
                self_n = stats.get("n", 0)
                if self_n == 0:
                    continue
                slot = per_agent.setdefault(agent, {
                    "self_w": 0.0, "self_n": 0,
                    "self_lo": [], "self_hi": [],
                    "cross_w": 0.0, "cross_n": 0,
                    "cross_lo": [], "cross_hi": [],
                })
                slot["self_w"] += stats["mean"] * self_n
                slot["self_n"] += self_n
                slot["self_lo"].append((stats["ci_lo"], self_n))
                slot["self_hi"].append((stats["ci_hi"], self_n))
                cn = stats.get("cross_n", 0)
                if cn:
                    slot["cross_w"] += stats["cross_mean"] * cn
                    slot["cross_n"] += cn
                    slot["cross_lo"].append((stats["cross_ci_lo"], cn))
                    slot["cross_hi"].append((stats["cross_ci_hi"], cn))
    if not per_agent:
        print("per_agent_cc: empty; skipping"); return

    def _wavg(pairs):
        num = sum(v * n for v, n in pairs)
        den = sum(n for _, n in pairs)
        return num / den if den else 0.0

    order = ["qwen_image", "qwen_multimodal", "llama_text"]
    agents = [a for a in order if a in per_agent]

    self_mean  = np.array([per_agent[a]["self_w"] / per_agent[a]["self_n"] for a in agents])
    self_lo    = np.array([_wavg(per_agent[a]["self_lo"]) for a in agents])
    self_hi    = np.array([_wavg(per_agent[a]["self_hi"]) for a in agents])
    cross_mean = np.array([
        per_agent[a]["cross_w"] / per_agent[a]["cross_n"]
        if per_agent[a]["cross_n"] else 0.0 for a in agents])
    cross_lo   = np.array([_wavg(per_agent[a]["cross_lo"]) for a in agents])
    cross_hi   = np.array([_wavg(per_agent[a]["cross_hi"]) for a in agents])

    self_err  = np.vstack([self_mean - self_lo, self_hi - self_mean])
    cross_err = np.vstack([cross_mean - cross_lo, cross_hi - cross_mean])

    pretty = {
        "qwen_image":      "image agent",
        "qwen_multimodal": "multimodal agent",
        "llama_text":      "text agent",
    }
    ytick_labels = [
        f"{pretty[a]}\n(n = {per_agent[a]['self_n']:,} contracts)"
        for a in agents
    ]

    apply_publication_style(font_size=22, axes_linewidth=3)
    fig = plt.figure(figsize=(16, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.32], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])

    y = np.arange(len(agents))
    h = 0.36
    ax.barh(y - h/2, self_mean, h, color=BASELINE, edgecolor="black",
            linewidth=1.6,
            xerr=self_err, error_kw=dict(ecolor="black", lw=1.5, capsize=6))
    ax.barh(y + h/2, cross_mean, h, color=PROPOSED, edgecolor="black",
            linewidth=1.6,
            xerr=cross_err, error_kw=dict(ecolor="black", lw=1.5, capsize=6))

    for yi, mean, hi in zip(y - h/2, self_mean, self_hi):
        t = ax.text(max(mean, hi) + 0.018, yi, f"{mean:.2f}",
                    va="center", fontsize=18, color=INK)
        _halo(t)
    for yi, mean, hi in zip(y + h/2, cross_mean, cross_hi):
        t = ax.text(max(mean, hi) + 0.018, yi, f"{mean:.2f}",
                    va="center", fontsize=18, color=INK, fontweight="bold")
        _halo(t)

    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels, fontsize=20, color=INK)
    # CC values for the 3 agents are in the 0.05-0.11 band, with CI tops
    # well under 0.25. A 0-to-1 x-axis would waste 70% of the panel and
    # hide all per-agent differences. Tighten to [0, 0.30].
    ax.set_xlim(0, 0.30)
    ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_xlabel(r"Counterfactual-consistency rate  $\widehat{CC}(a)$",
                  fontsize=24)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=5, width=2)

    ax_leg = fig.add_subplot(gs[0, 1])
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE, edgecolor="black", linewidth=1.6),
        plt.Rectangle((0, 0), 1, 1, facecolor=PROPOSED, edgecolor="black", linewidth=1.6),
    ]
    ax_leg.legend(handles,
                  ["Self-form\n" r"$\widehat{CC}_{\rm self}(a)$",
                   "Cross-form\n" r"$\widehat{CC}_{\rm cross}(a)$"],
                  loc="center left", fontsize=22, frameon=False,
                  handlelength=1.4, handletextpad=0.8,
                  labelspacing=1.4, borderaxespad=0)
    ax_leg.set_axis_off()

    finalize(fig, out_dir / "fig_per_agent_cc", dpi=300)


# ============================================================
# Figure 9 — Intervention informativeness I (multimodal)
# ============================================================


def fig_intervention_I(out_dir: Path) -> None:
    summary = _load_json(ANALYSIS_DIR / "summary.json")
    if not summary:
        print("intervention_I: no summary; skipping"); return

    agg = {}
    for regs in summary.get("per_dataset", {}).values():
        for blob in regs.values():
            for ivn, stats in (blob.get("informativeness") or {}).items():
                a = agg.setdefault(ivn, {"attempts": 0, "flagged": 0})
                a["attempts"] += stats.get("n_attempts", 0)
                a["flagged"]  += stats.get("n_flagged", 0)
    items = [(ivn.replace("_", " "), v["flagged"] / max(v["attempts"], 1),
              v["attempts"])
             for ivn, v in agg.items() if v["attempts"] >= 2]
    if not items:
        print("intervention_I: empty; skipping"); return
    items.sort(key=lambda x: x[1])

    labels = [f"{name}  (n={n:,})" for (name, _, n) in items]
    vals = np.array([v for (_, v, _) in items])
    XMIN, XMAX = 0.93, 1.02

    alphas = np.linspace(0.45, 1.0, len(items))
    # Alpha-graded teal — distinct from blue (used for VOUCH headline bars)
    # and from violet (used for intervention CHANGE rate panel).
    teal_rgb = (66/255, 148/255, 158/255)
    colors = [(*teal_rgb, a) for a in alphas]

    apply_publication_style(font_size=22, axes_linewidth=3)
    fig, ax = plt.subplots(figsize=(13, max(5.0, 0.7 * len(items) + 1.6)),
                           constrained_layout=True)
    y = np.arange(len(items))
    widths = vals - XMIN
    ax.barh(y, widths, left=XMIN, color=colors, edgecolor="black",
            linewidth=1.6, height=0.66)
    for yi, v in zip(y, vals):
        t = ax.text(v + 0.002, yi, f"{v:.3f}",
                    va="center", ha="left", fontsize=20, color=INK,
                    fontweight="bold")
        _halo(t)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=20, color=INK)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([0.94, 0.96, 0.98, 1.00])
    ax.set_xlabel(r"Intervention informativeness  $\widehat{I}(\iota)$",
                  fontsize=24)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=5, width=2)

    finalize(fig, out_dir / "fig_intervention_I", dpi=300)


# ============================================================
# Figure 10 — Sample-size stability (MBPP adversarial)
# ============================================================


def fig_sample_size_stability(out_dir: Path) -> None:
    path = ROOT / "results" / "mbpp_codegen_dual_adversarial_n200.jsonl"
    if not path.exists():
        print("sample_size_stability: missing jsonl; skipping"); return

    rows = [json.loads(l) for l in path.open() if l.strip()]
    rng = np.random.default_rng(0)
    ns = [50, 100, 150, 200]
    means, lows, highs = [], [], []
    for n in ns:
        boots = []
        for _ in range(1000):
            idx = rng.choice(len(rows), size=n, replace=True)
            sample = [rows[i] for i in idx]
            hurt = caught = 0
            for r in sample:
                v0 = r.get("v0_a", "")
                if not (v0 and "F" in v0):
                    continue
                if r.get("verified_dual"):
                    hurt += 1
                elif r.get("contract_ok"):
                    caught += 1
            boots.append(hurt / max(caught, 1))
        means.append(np.mean(boots))
        lows.append(np.percentile(boots, 2.5))
        highs.append(np.percentile(boots, 97.5))
    means = np.array(means); lows = np.array(lows); highs = np.array(highs)

    apply_publication_style(font_size=22, axes_linewidth=3)
    fig = plt.figure(figsize=(14, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.32], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])

    ax.axhline(0.640, color=BASELINE_2, linewidth=2.0,
               linestyle=(0, (5, 3)), alpha=0.9, zorder=1)
    t = ax.text(ns[-1] - 4, 0.660,
                "single-side baseline  (0.64)", ha="right", va="bottom",
                fontsize=18, color=BASELINE_2, style="italic",
                fontweight="bold")
    _halo(t)

    ax.fill_between(ns, lows, highs, color=PROPOSED, alpha=0.18,
                    linewidth=0, zorder=2)
    ax.plot(ns, means, color=PROPOSED, linewidth=3.0, zorder=3)
    ax.scatter(ns, means, s=180, color=PROPOSED, edgecolor="black",
               linewidth=2.0, zorder=4)
    for n, m in zip(ns, means):
        t = ax.text(n, m + 0.030, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=18, color=INK,
                    fontweight="bold")
        _halo(t)

    ax.set_xlabel(r"Sample size  $n$", fontsize=24)
    ax.set_ylabel("Hurt : caught  (MBPP adv.)", fontsize=24)
    ax.set_xticks(ns)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.0, 0.20, 0.40, 0.60, 0.80])
    ax.tick_params(axis="both", length=5, width=2)

    ax_leg = fig.add_subplot(gs[0, 1])
    handles = [
        plt.Line2D([0], [0], color=PROPOSED, linewidth=3,
                   marker="o", markersize=12, markeredgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PROPOSED, alpha=0.22,
                      edgecolor="none"),
        plt.Line2D([0], [0], color=BASELINE_2, linewidth=3,
                   linestyle=(0, (5, 3))),
    ]
    labels = [
        "Dual-side\nhurt : caught",
        "95% bootstrap CI\n(1000 resamples)",
        "Single-side\nbaseline (0.64)",
    ]
    ax_leg.legend(handles, labels, loc="center left",
                  fontsize=20, frameon=False,
                  handlelength=1.6, handletextpad=0.8,
                  labelspacing=1.4, borderaxespad=0)
    ax_leg.set_axis_off()

    finalize(fig, out_dir / "fig_sample_size_stability", dpi=300)


# ============================================================
# Figure 11 — Weight trajectory (Hateful Memes clean)
# ============================================================


def fig_weight_trajectory(out_dir: Path) -> None:
    summary = _load_json(ANALYSIS_DIR / "summary.json")
    if not summary:
        print("weight_trajectory: no summary; skipping"); return

    target = summary.get("per_dataset", {}).get("hateful_memes", {}).get("clean", {})
    wt = target.get("weight_trajectory", {})
    if not wt:
        for ds in ["crisismmd", "mm_imdb"]:
            target = summary.get("per_dataset", {}).get(ds, {}).get("clean", {})
            wt = target.get("weight_trajectory", {})
            if wt:
                break
    if not wt:
        print("weight_trajectory: no data; skipping"); return

    color_map = {
        "llama_text":      BASELINE_2,
        "qwen_image":      GOOD,
        "qwen_multimodal": PROPOSED,
    }
    label_map = {
        "llama_text":      "text agent",
        "qwen_image":      "image agent",
        "qwen_multimodal": "multimodal agent",
    }
    order = ["qwen_image", "llama_text", "qwen_multimodal"]
    items = [(a, wt[a]) for a in order if a in wt] + \
            [(a, traj) for a, traj in wt.items() if a not in order]

    apply_publication_style(font_size=22, axes_linewidth=3)
    fig = plt.figure(figsize=(15, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.30], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color=HAIRLINE,
            linewidth=0.8, alpha=0.7, zorder=0)

    n_examples = 0
    for agent, traj in items:
        ts = np.array([p[0] for p in traj], dtype=float)
        ws = np.array([max(p[1], 1e-4) for p in traj], dtype=float)
        c = color_map.get(agent, INK)
        n_examples = max(n_examples, int(ts[-1]))

        change_idx = [0]
        for i in range(1, len(ws)):
            if abs(ws[i] - ws[i - 1]) > 1e-9:
                change_idx.append(i)
        change_idx.append(len(ws) - 1)
        change_idx = sorted(set(change_idx))

        ax.step(ts, ws, where="post", color=c, alpha=0.18,
                linewidth=9.0, solid_capstyle="round", zorder=2)
        ax.step(ts, ws, where="post", color=c, alpha=0.95,
                linewidth=3.0, solid_capstyle="round", zorder=3,
                label=label_map.get(agent, agent))
        if len(change_idx) > 2:
            tx = ts[change_idx]; wx = ws[change_idx]
            ax.scatter(tx, wx, s=70, color=c, edgecolor=PAPER,
                       linewidth=1.5, zorder=4, clip_on=False)

    x_end = n_examples
    for agent, traj in items:
        final_w = max(traj[-1][1], 1e-4)
        c = color_map.get(agent, INK)
        ax.plot([x_end, x_end + 4], [final_w, final_w],
                color=c, alpha=0.6, linewidth=1.5,
                zorder=3, clip_on=False)
        label = f"{final_w:.2g}" if final_w >= 1e-3 else f"{final_w:.1e}"
        t = ax.text(x_end + 6, final_w, label,
                    fontsize=18, color=c, fontweight="bold",
                    ha="left", va="center")
        _halo(t)

    ax.set_xlabel(rf"Example index  $t$   (Hateful Memes clean, $n={n_examples}$)",
                  fontsize=22)
    ax.set_ylabel(r"Voting weight  $w_a^{(t)}$   (log scale)", fontsize=22)
    ax.set_yscale("log")
    ax.set_ylim(7e-4, 1.6)
    ax.set_xlim(0, n_examples + 60)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax.tick_params(axis="both", length=5, width=2)

    ax_leg = fig.add_subplot(gs[0, 1])
    handles = [
        plt.Line2D([0], [0], color=GOOD,       linewidth=4),
        plt.Line2D([0], [0], color=BASELINE_2, linewidth=4),
        plt.Line2D([0], [0], color=PROPOSED,   linewidth=4),
    ]
    ax_leg.legend(handles,
                  ["image agent", "text agent", "multimodal agent"],
                  loc="center left", fontsize=22, frameon=False,
                  handlelength=1.8, handletextpad=0.8,
                  labelspacing=1.4, borderaxespad=0)
    ax_leg.set_axis_off()

    finalize(fig, out_dir / "fig_weight_trajectory", dpi=300)


# ============================================================
# Driver
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[figures_better] writing to {args.out}")
    fig_abstention(args.out)
    fig_ci_hurt(args.out)
    fig_heterogeneity(args.out)
    fig_intervention_change(args.out)
    fig_hurt_caught(args.out)
    fig_calibration(args.out)
    fig_cross_domain_signature(args.out)
    fig_per_agent_cc(args.out)
    fig_intervention_I(args.out)
    fig_sample_size_stability(args.out)
    fig_weight_trajectory(args.out)
    print("[figures_better] done")


if __name__ == "__main__":
    main()
