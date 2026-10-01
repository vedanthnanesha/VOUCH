"""Figure renderer for the paper.

Reads results/analysis/summary.json (written by
``python -m experiments.format_results --analyses``) and results/*.jsonl,
and writes PDFs to figures/.

Design language:
  - Light, elegant indigo / violet palette with warm accents
  - Single accent color per concept; neutral grays for context
  - No chart titles (captions in LaTeX carry that)
  - Smooth lines, halo'd scatter markers, soft fills
  - Generous margins; constrained_layout so nothing gets clipped
  - Quiet typography: tight axis labels, bold value annotations
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- Palette: muted, paper-elegant (lower saturation than before) ----
PRIMARY      = "#7E8FE0"  # softer indigo (was #6366F1)
PRIMARY_DARK = "#5763C3"  # for emphasis only
PRIMARY_FILL = "#F1F4FB"  # very soft indigo wash
ACCENT       = "#D88BB4"  # dusty pink
WARM         = "#E9A87E"  # muted orange (was #FB923C)
WARM_FILL    = "#FDF5EE"  # warm wash
GOOD         = "#7FB995"  # muted emerald
BAD          = "#D88080"  # muted red
BAD_FILL     = "#FAEDED"  # very soft red wash
INK          = "#1F2937"  # gray-800 (primary text)
MUTED        = "#6B7280"  # gray-500 (axis ticks/labels)
HAIRLINE     = "#D1D5DB"  # gray-300 (spines)
PAPER        = "#FFFFFF"


PLT_STYLE = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
    "font.weight":        "regular",
    "axes.titlesize":     0.01,
    "axes.titleweight":   "regular",
    "axes.labelsize":     11,
    "axes.labelweight":   "regular",
    "axes.labelcolor":    INK,
    "axes.edgecolor":     HAIRLINE,
    "axes.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelpad":      8,
    "axes.titlepad":      0,
    "xtick.labelsize":    9.5,
    "ytick.labelsize":    9.5,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.major.pad":    5,
    "ytick.major.pad":    5,
    "legend.fontsize":    9.5,
    "legend.frameon":     False,
    "legend.handlelength": 1.3,
    "legend.handletextpad": 0.6,
    "legend.borderpad":   0.2,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
}


def setup_style():
    for k, v in PLT_STYLE.items():
        mpl.rcParams[k] = v


def _bold_value(ax, x, y, text, color, fontsize=9.5):
    txt = ax.text(x, y, text, ha="center", va="bottom",
                   fontsize=fontsize, color=color, fontweight="bold")
    txt.set_path_effects([
        path_effects.Stroke(linewidth=2.0, foreground=PAPER),
        path_effects.Normal(),
    ])


def _smooth_xy(xs, ys, n=200):
    """Simple linear-interpolated smoothing for trajectory aesthetics."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 4:
        return xs, ys
    xnew = np.linspace(xs.min(), xs.max(), n)
    ynew = np.interp(xnew, xs, ys)
    return xnew, ynew


# ============================================================
# Figure 3: hurt:caught comparison (headline result)
# ============================================================

def fig_hurt_caught_bars(out: Path) -> None:
    benchmarks  = ["MBPP\nclean", "MBPP\nadversarial",
                    "HumanEval\nclean", "HumanEval\nadversarial"]
    single_side = np.array([0.575, 0.640, 2.067, 1.625])
    dual_side   = np.array([0.145, 0.093, 1.875, 1.520])

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    x = np.arange(len(benchmarks))
    w = 0.34

    bars1 = ax.bar(x - w/2, single_side, w, label="single-side",
                    color=WARM, alpha=0.95, edgecolor=PAPER, linewidth=1.2,
                    zorder=3)
    bars2 = ax.bar(x + w/2, dual_side, w, label="dual-side",
                    color=PRIMARY, alpha=0.98, edgecolor=PAPER, linewidth=1.2,
                    zorder=3)

    for bars, vals in [(bars1, single_side), (bars2, dual_side)]:
        for b, v in zip(bars, vals):
            _bold_value(ax, b.get_x() + b.get_width()/2, v + 0.06,
                         f"{v:.2f}", INK, fontsize=10)

    # Break-even reference at h:c=1, with annotation on the LEFT to avoid bars
    ax.axhline(y=1.0, linestyle=(0, (4, 3)), linewidth=1.3,
                color=BAD, alpha=0.75, zorder=2)
    ax.text(-0.40, 1.08,
            "break-even (hurt : caught = 1)",
            ha="left", va="bottom", fontsize=10,
            color=BAD, style="italic", fontweight="semibold",
            path_effects=[path_effects.Stroke(linewidth=2.2, foreground=PAPER),
                          path_effects.Normal()])

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10.5, color=INK)
    ax.set_ylabel("hurt : caught ratio   (lower is better)",
                    fontsize=11.5)
    ax.set_xlim(-0.55, len(benchmarks) - 0.45)
    ax.set_ylim(0, 2.7)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.tick_params(axis='both', which='both', length=0)
    ax.spines['left'].set_color(HAIRLINE)

    ax.legend(loc="upper right", fontsize=10.5, frameon=True,
                fancybox=True, framealpha=0.96, edgecolor="none",
                facecolor=PAPER, borderpad=0.6,
                handlelength=1.4, handletextpad=0.7)
    fig.savefig(out / "fig_hurt_caught.pdf")
    plt.close(fig)


# ============================================================
# Figure 4: cross-domain pathology scatter
# ============================================================

def fig_cross_domain_signature(out: Path) -> None:
    points = [
        ("MBPP",            -5.4,  3.1),
        ("HumanEval",       -7.3,  2.0),
        ("GSM8K",          -28.0,  4.2),
        ("MATH (algebra)",   0.6, 11.6),
        ("SVAMP",          -21.0,  5.5),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.set_xlim(-34, 8)
    ax.set_ylim(-2, 15)

    ax.axhspan(0, 15, xmin=0, xmax=34/42, color=BAD_FILL, zorder=0)

    ax.axhline(y=0, color=HAIRLINE, linewidth=0.7, zorder=1)
    ax.axvline(x=0, color=HAIRLINE, linewidth=0.7, zorder=1)

    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])

    ax.scatter(xs, ys, s=320, color=PRIMARY, alpha=0.22, zorder=2,
                edgecolor="none")
    ax.scatter(xs, ys, s=90, color=PRIMARY, zorder=3,
                edgecolor=PAPER, linewidth=1.5)

    label_offsets = {
        "MBPP":            (1.0,  0.50),
        "HumanEval":       (1.0, -1.10),
        "GSM8K":           (1.2,  0.50),
        "MATH (algebra)": (-9.0,  0.50),
        "SVAMP":           (1.2,  0.50),
    }
    for lab, x, y in points:
        dx, dy = label_offsets.get(lab, (1.0, 0.5))
        t = ax.text(x + dx, y + dy, lab, fontsize=10.5, color=INK,
                     fontweight="semibold")
        t.set_path_effects([
            path_effects.Stroke(linewidth=2.5, foreground=PAPER),
            path_effects.Normal(),
        ])

    qx = -32
    ax.text(qx, 13.7, "pathology quadrant",
            fontsize=10, color=BAD, fontweight="bold",
            path_effects=[path_effects.Stroke(linewidth=2.5, foreground=PAPER),
                          path_effects.Normal()])
    ax.text(qx, 12.6, "capability falls, self-coherence rises",
            fontsize=9, color=BAD, style="italic", alpha=0.9,
            path_effects=[path_effects.Stroke(linewidth=2.5, foreground=PAPER),
                          path_effects.Normal()])

    ax.set_xlabel(r"$\Delta$ accuracy (pp): adversarial regime $-$ clean regime",
                    fontsize=11)
    ax.set_ylabel(r"$\Delta$ verification rate (pp): adversarial $-$ clean",
                    fontsize=11)
    ax.tick_params(axis='both', which='both', length=0)
    fig.savefig(out / "fig_cross_domain_signature.pdf")
    plt.close(fig)


# ============================================================
# Figure A1: per-agent CC
# ============================================================

def _full_dataset_label(ds: str) -> str:
    return {
        "hateful_memes": "Hateful Memes",
        "crisismmd":     "CrisisMMD",
        "mm_imdb":       "MM-IMDb",
    }.get(ds, ds)


def _full_regime_label(reg: str) -> str:
    return {"adv": "adversarial", "adversarial": "adversarial",
             "clean": "clean"}.get(reg, reg)


def _full_agent_label(agent: str) -> str:
    return {
        "llama_text":      "text",
        "qwen_image":      "image",
        "qwen_multimodal": "multimodal",
    }.get(agent, agent)


def fig_per_agent_cc(out: Path, summary: dict) -> None:
    # Pool all cells together; aggregate self-CC and cross-CC per agent.
    # The picture we want to show: self-CC is mostly low (agents fail their
    # own commitment); cross-CC measures whether an agent can correctly
    # predict another agent's behaviour under the same intervention.
    per_agent = {}
    for ds, regs in summary.get("per_dataset", {}).items():
        for reg, blob in regs.items():
            for agent, stats in (blob.get("cc_per_agent") or {}).items():
                self_n = stats.get("n", 0)
                if self_n == 0:
                    continue
                slot = per_agent.setdefault(agent, {
                    "self_w": 0.0, "self_n": 0, "self_lo": [], "self_hi": [],
                    "cross_w": 0.0, "cross_n": 0, "cross_lo": [], "cross_hi": [],
                })
                slot["self_w"]  += stats["mean"] * self_n
                slot["self_n"]  += self_n
                slot["self_lo"].append((stats["ci_lo"], self_n))
                slot["self_hi"].append((stats["ci_hi"], self_n))
                cn = stats.get("cross_n", 0)
                if cn:
                    slot["cross_w"] += stats["cross_mean"] * cn
                    slot["cross_n"] += cn
                    slot["cross_lo"].append((stats["cross_ci_lo"], cn))
                    slot["cross_hi"].append((stats["cross_ci_hi"], cn))
    if not per_agent:
        return

    def _wavg(pairs):
        num = sum(v * n for v, n in pairs)
        den = sum(n for _, n in pairs)
        return num / den if den else 0.0

    order = ["qwen_image", "qwen_multimodal", "llama_text"]
    agents = [a for a in order if a in per_agent]

    self_mean  = np.array([per_agent[a]["self_w"]  / per_agent[a]["self_n"]  for a in agents])
    self_lo    = np.array([_wavg(per_agent[a]["self_lo"])  for a in agents])
    self_hi    = np.array([_wavg(per_agent[a]["self_hi"])  for a in agents])
    self_n     = [per_agent[a]["self_n"] for a in agents]
    cross_mean = np.array([per_agent[a]["cross_w"] / per_agent[a]["cross_n"] if per_agent[a]["cross_n"] else 0.0 for a in agents])
    cross_lo   = np.array([_wavg(per_agent[a]["cross_lo"]) for a in agents])
    cross_hi   = np.array([_wavg(per_agent[a]["cross_hi"]) for a in agents])
    cross_n    = [per_agent[a]["cross_n"] for a in agents]

    self_err  = np.vstack([self_mean - self_lo, self_hi - self_mean])
    cross_err = np.vstack([cross_mean - cross_lo, cross_hi - cross_mean])

    pretty = {
        "qwen_image":      "image agent",
        "qwen_multimodal": "multimodal agent",
        "llama_text":      "text agent",
    }
    # Wider figure with a bit more height so labels and the legend
    # have room. We keep sample counts on the y-tick labels (not on the
    # bars) to avoid value-label collisions when self-CC and cross-CC
    # are both small (e.g. ~0.05).
    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)

    y = np.arange(len(agents))
    h = 0.36
    ax.barh(y - h/2, self_mean, h,
            color=PRIMARY, alpha=0.95, edgecolor=PAPER, linewidth=1.0,
            xerr=self_err,
            error_kw=dict(ecolor=INK, lw=0.9, capsize=3, alpha=0.55),
            label=r"self-form $\widehat{CC}_{\mathrm{self}}(a)$",
            zorder=3)
    ax.barh(y + h/2, cross_mean, h,
            color=GOOD, alpha=0.95, edgecolor=PAPER, linewidth=1.0,
            xerr=cross_err,
            error_kw=dict(ecolor=INK, lw=0.9, capsize=3, alpha=0.55),
            label=r"cross-form $\widehat{CC}_{\mathrm{cross}}(a)$",
            zorder=3)

    # Numeric labels: only the CC value, placed past the error-bar tip so
    # they never sit on top of each other.
    for yi, mean, hi in zip(y - h/2, self_mean, self_hi):
        t = ax.text(max(mean, hi) + 0.025, yi, f"{mean:.2f}",
                     va="center", fontsize=10, color=PRIMARY_DARK,
                     fontweight="semibold")
        t.set_path_effects([path_effects.Stroke(linewidth=2.0, foreground=PAPER),
                             path_effects.Normal()])
    for yi, mean, hi in zip(y + h/2, cross_mean, cross_hi):
        t = ax.text(max(mean, hi) + 0.025, yi, f"{mean:.2f}",
                     va="center", fontsize=10, color=INK,
                     fontweight="semibold")
        t.set_path_effects([path_effects.Stroke(linewidth=2.0, foreground=PAPER),
                             path_effects.Normal()])

    # y-tick label carries the agent name and per-agent contract count.
    ytick_labels = [
        f"{pretty[a]}\n(n={per_agent[a]['self_n']} contracts)"
        for a in agents
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels, fontsize=10.5, color=INK)
    # CC values are in the 0.05-0.11 band; do not stretch to 1.0.
    ax.set_xlim(0, 0.30)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.set_xlabel(r"counterfactual-consistency rate $\widehat{CC}(a)$",
                   fontsize=11, color=INK)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=HAIRLINE, linewidth=0.6, alpha=0.6, zorder=0)
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(HAIRLINE)
        ax.spines[spine].set_linewidth(0.8)
    ax.legend(loc="lower right", fontsize=10, frameon=True, fancybox=True,
              framealpha=0.96, edgecolor="none", facecolor=PAPER,
              borderpad=0.5, handlelength=1.4, handletextpad=0.6)
    ax.invert_yaxis()
    fig.savefig(out / "fig_per_agent_cc.pdf")
    plt.close(fig)


# ============================================================
# Figure A2: intervention informativeness
# ============================================================

def fig_intervention_I(out: Path, summary: dict) -> None:
    agg = {}
    for ds, regs in summary.get("per_dataset", {}).items():
        for reg, blob in regs.items():
            for ivn, stats in (blob.get("informativeness") or {}).items():
                if ivn not in agg:
                    agg[ivn] = {"attempts": 0, "flagged": 0}
                agg[ivn]["attempts"] += stats.get("n_attempts", 0)
                agg[ivn]["flagged"] += stats.get("n_flagged", 0)
    if not agg:
        return
    items = [(ivn.replace("_", " "),
               v["flagged"] / max(v["attempts"], 1), v["attempts"])
             for ivn, v in agg.items() if v["attempts"] >= 2]
    items.sort(key=lambda x: x[1])
    labels = [f"{x[0]}  (n={x[2]})" for x in items]
    vals = np.array([x[1] for x in items])

    # All values are near 1.0; zoom so differences are visible.
    XMIN = 0.93
    XMAX = 1.02
    fig_h = max(3.0, 0.48 * len(items))
    fig, ax = plt.subplots(figsize=(4.7, fig_h), constrained_layout=True)
    y = np.arange(len(items))
    # Draw bars on the zoomed axis: subtract XMIN so the bar starts at 0 visually
    widths = vals - XMIN
    ax.barh(y, widths, left=XMIN, color=PRIMARY, alpha=0.95,
             edgecolor=PAPER, linewidth=1.0, height=0.62, zorder=2)
    for yi, v in zip(y, vals):
        t = ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=9.5,
                     color=INK, fontweight="semibold")
        t.set_path_effects([
            path_effects.Stroke(linewidth=2.0, foreground=PAPER),
            path_effects.Normal(),
        ])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel(r"intervention informativeness $\hat{I}(\iota)$",
                    fontsize=10.5)
    ax.set_xlim(XMIN, XMAX)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax.tick_params(axis='both', which='both', length=0)
    fig.savefig(out / "fig_intervention_I.pdf")
    plt.close(fig)


# ============================================================
# Figure A3: sample-size stability
# ============================================================

def fig_sample_size_stability(out: Path) -> None:
    path = ROOT / "results" / "mbpp_codegen_dual_adversarial_n200.jsonl"
    if not path.exists():
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
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
                wrong_initial = v0 and "F" in v0
                if not wrong_initial: continue
                if r.get("verified_dual"): hurt += 1
                elif r.get("contract_ok"): caught += 1
            boots.append(hurt / max(caught, 1))
        means.append(np.mean(boots))
        lows.append(np.percentile(boots, 2.5))
        highs.append(np.percentile(boots, 97.5))
    means, lows, highs = map(np.array, (means, lows, highs))

    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    ns_s, lows_s = _smooth_xy(ns, lows)
    _, highs_s = _smooth_xy(ns, highs)
    _, means_s = _smooth_xy(ns, means)
    ax.fill_between(ns_s, lows_s, highs_s, color=PRIMARY, alpha=0.18,
                     linewidth=0, label=r"95\% bootstrap CI", zorder=2)
    ax.plot(ns_s, means_s, color=PRIMARY, linewidth=2.0, zorder=3,
             alpha=0.95, label="dual-side hurt : caught")
    ax.scatter(ns, means, s=70, color=PAPER, edgecolor=PRIMARY_DARK,
                linewidth=1.7, zorder=4)

    ax.axhline(y=0.640, linestyle=(0, (4, 3)), linewidth=1.3,
                color=WARM, alpha=0.95,
                label="single-side baseline ($0.64$)")

    ax.set_xlabel(r"sample size $n$", fontsize=10.5)
    ax.set_ylabel("hurt : caught   (MBPP adversarial)", fontsize=10.5)
    ax.set_xticks(ns)
    ax.set_ylim(0, 0.80)
    ax.tick_params(axis='both', which='both', length=0)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, fancybox=True,
               framealpha=0.95, edgecolor="none", facecolor=PAPER,
               borderpad=0.5, handlelength=1.4, handletextpad=0.6)
    fig.savefig(out / "fig_sample_size_stability.pdf")
    plt.close(fig)


# ============================================================
# Figure A4: weight trajectory
# ============================================================

def fig_weight_trajectory(out: Path, summary: dict) -> None:
    # Prefer HM/clean: shows real divergence (image stays at 1.0,
    # text drops to 0.25, multimodal drops to ~0.001). MM-IMDb collapses
    # everyone to ~0 and is uninformative.
    target = summary.get("per_dataset", {}).get("hateful_memes", {}).get("clean", {})
    wt = target.get("weight_trajectory", {})
    if not wt:
        for ds in ["crisismmd", "mm_imdb"]:
            target = summary.get("per_dataset", {}).get(ds, {}).get("clean", {})
            wt = target.get("weight_trajectory", {})
            if wt:
                break
    if not wt:
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    color_map = {
        "llama_text":      WARM,
        "qwen_image":      GOOD,
        "qwen_multimodal": PRIMARY,
    }
    label_map = {
        "llama_text":      "text agent",
        "qwen_image":      "image agent",
        "qwen_multimodal": "multimodal agent",
    }
    order = ["qwen_image", "llama_text", "qwen_multimodal"]
    items = [(a, wt[a]) for a in order if a in wt] + \
            [(a, traj) for a, traj in wt.items() if a not in order]

    # Subtle gridlines first so they sit behind everything.
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color=HAIRLINE,
            linewidth=0.6, alpha=0.7, zorder=0)

    for agent, traj in items:
        ts = np.array([p[0] for p in traj], dtype=float)
        ws = np.array([max(p[1], 1e-4) for p in traj], dtype=float)
        c = color_map.get(agent, INK)

        # Identify true transition points so we can mark them.
        change_idx = [0]
        for i in range(1, len(ws)):
            if abs(ws[i] - ws[i-1]) > 1e-9:
                change_idx.append(i)
        change_idx.append(len(ws) - 1)
        change_idx = sorted(set(change_idx))

        # Soft glow as a backing layer.
        ax.step(ts, ws, where="post", color=c, alpha=0.18,
                linewidth=6.5, solid_capstyle="round", zorder=2)
        # Crisp staircase.
        ax.step(ts, ws, where="post", color=c, alpha=0.98,
                linewidth=2.2, solid_capstyle="round", zorder=3,
                label=label_map.get(agent, agent))
        # Mark each transition with a small dot to highlight discrete updates.
        if len(change_idx) > 2:
            tx = ts[change_idx]
            wx = ws[change_idx]
            ax.scatter(tx, wx, s=24, color=c,
                       edgecolor=PAPER, linewidth=0.8,
                       zorder=4, clip_on=False)

    n_examples = int(max(ts))
    ax.set_xlabel(f"example index $t$ (Hateful Memes clean, $n={n_examples}$)",
                   fontsize=10.5)
    ax.set_ylabel(r"voting weight $w_a^{(t)}$ (log scale)", fontsize=10.5)
    ax.set_yscale("log")
    ax.set_ylim(7e-4, 1.6)
    ax.set_xlim(0, n_examples * 1.02)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(HAIRLINE)
        ax.spines[spine].set_linewidth(0.8)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True, fancybox=True,
              framealpha=0.95, edgecolor="none", facecolor=PAPER,
              borderpad=0.5, handlelength=1.6, handletextpad=0.6)
    fig.savefig(out / "fig_weight_trajectory.pdf")
    plt.close(fig)


def fig_weight_trajectory_v2(out: Path, summary: dict) -> None:
    # Alternate version: CrisisMMD clean shows all three agents moving
    # (not just one or two). Per-step values annotated so the staircase
    # is readable, not just "a straight line down".
    target = summary.get("per_dataset", {}).get("crisismmd", {}).get("clean", {})
    wt = target.get("weight_trajectory", {})
    if not wt:
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    color_map = {
        "llama_text":      WARM,
        "qwen_image":      GOOD,
        "qwen_multimodal": PRIMARY,
    }
    label_map = {
        "llama_text":      "text agent",
        "qwen_image":      "image agent",
        "qwen_multimodal": "multimodal agent",
    }
    order = ["qwen_image", "llama_text", "qwen_multimodal"]
    items = [(a, wt[a]) for a in order if a in wt] + \
            [(a, traj) for a, traj in wt.items() if a not in order]

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", color=HAIRLINE,
            linewidth=0.6, alpha=0.6, zorder=0)
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)),
                alpha=0.7, zorder=1)

    YMIN = 1e-5
    n_examples = 0

    for agent, traj in items:
        ts = np.array([p[0] for p in traj], dtype=float)
        ws = np.array([max(p[1], YMIN) for p in traj], dtype=float)
        c = color_map.get(agent, INK)
        n_examples = max(n_examples, int(ts[-1]))

        change_idx = [0]
        for i in range(1, len(ws)):
            if abs(ws[i] - ws[i-1]) > 1e-12:
                change_idx.append(i)
        change_idx.append(len(ws) - 1)
        change_idx = sorted(set(change_idx))

        ax.step(ts, ws, where="post", color=c, alpha=0.20,
                linewidth=7.0, solid_capstyle="round", zorder=2)
        ax.step(ts, ws, where="post", color=c, alpha=0.98,
                linewidth=2.3, solid_capstyle="round", zorder=3,
                label=label_map.get(agent, agent))

        if len(change_idx) > 2:
            tx = ts[change_idx]
            wx = ws[change_idx]
            ax.scatter(tx, wx, s=36, color=c,
                       edgecolor=PAPER, linewidth=1.0,
                       zorder=4, clip_on=False)

    # Annotate ONLY the final value per agent, placed to the right of the
    # last data point, vertically aligned with the end-of-trajectory weight.
    # This avoids the overlap that happened when we labelled mid-trajectory
    # transitions for all three agents.
    x_end = n_examples
    for agent, traj in items:
        final_w = max(traj[-1][1], YMIN)
        c = color_map.get(agent, INK)
        # short tick from data into label region
        ax.plot([x_end, x_end + 6], [final_w, final_w],
                 color=c, alpha=0.55, linewidth=1.0, zorder=3,
                 clip_on=False)
        label = f"{final_w:.2g}" if final_w >= 1e-3 else f"{final_w:.1e}"
        t = ax.text(x_end + 8, final_w, label,
                     fontsize=9, color=c, fontweight="semibold",
                     ha="left", va="center")
        t.set_path_effects([
            path_effects.Stroke(linewidth=2.0, foreground=PAPER),
            path_effects.Normal(),
        ])

    # Caption note about the reference line at w=1
    ax.text(2, 1.30, "starting weight $w_a^{(0)} = 1$",
             ha="left", va="bottom", fontsize=8.5, color=MUTED, style="italic")

    ax.set_xlabel(f"example index $t$ (CrisisMMD clean, $n={n_examples}$)",
                   fontsize=10.5)
    ax.set_ylabel(r"voting weight $w_a^{(t)}$ (log scale)", fontsize=10.5)
    ax.set_yscale("log")
    ax.set_ylim(YMIN * 0.7, 3.5)
    # Leave room on the right for the end-of-trajectory value labels.
    ax.set_xlim(0, n_examples + 40)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(HAIRLINE)
        ax.spines[spine].set_linewidth(0.8)
    ax.legend(loc="lower left", fontsize=10, frameon=True, fancybox=True,
              framealpha=0.96, edgecolor="none", facecolor=PAPER,
              borderpad=0.5, handlelength=1.6, handletextpad=0.6)
    fig.savefig(out / "fig_weight_trajectory_v2.pdf")
    plt.close(fig)


# ============================================================
# Driver
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "figures")
    ap.add_argument("--summary", type=Path,
                    default=ROOT / "results" / "analysis" / "summary.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    if args.summary.exists():
        summary = json.load(open(args.summary))
    else:
        summary = {"per_dataset": {}}

    print(f"[render] writing to {args.out}")
    fig_hurt_caught_bars(args.out);                print("  fig_hurt_caught.pdf")
    fig_cross_domain_signature(args.out);           print("  fig_cross_domain_signature.pdf")
    fig_per_agent_cc(args.out, summary);            print("  fig_per_agent_cc.pdf")
    fig_intervention_I(args.out, summary);          print("  fig_intervention_I.pdf")
    fig_sample_size_stability(args.out);            print("  fig_sample_size_stability.pdf")
    fig_weight_trajectory(args.out, summary);       print("  fig_weight_trajectory.pdf")
    fig_weight_trajectory_v2(args.out, summary);    print("  fig_weight_trajectory_v2.pdf")
    print("[render] done")


if __name__ == "__main__":
    main()
