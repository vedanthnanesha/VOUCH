"""Counterfactual Consistency (CC), Intervention Informativeness (I), and
calibration metrics (ECE, Brier) for the VOUCH experiments.

All functions take per-example records of the shape that
``main_evaluation.py`` writes to the JSONL output (``ProtocolResult``
serialized to JSON). They never touch the model; pure post-hoc analysis.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Conditions that run the contract protocol (VOUCH, VOUCH-Random, VOUCH-Adaptive).
VOUCH_CONDITIONS = ("contracts", "contracts_random", "contracts_adaptive")


# ----------------------- IO --------------------------

def load_jsonl(path: Path, condition: str | None = None) -> list[dict]:
    out: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if condition and r.get("condition") != condition:
                continue
            out.append(r)
    return out


# ---------------- Counterfactual Consistency ----------------

def cc_per_agent(records: list[dict]) -> dict[str, dict]:
    """Empirical CC(a) = fraction of verifications in which the agent's
    counterfactual prediction matched its claim, per proposer.

    Args:
      records: list of contracts-condition ProtocolResult dicts.
    Returns:
      dict[agent_id, {n_contracts, n_verified, cc, ci_low, ci_high}].
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "v": 0})
    for r in records:
        for c in r.get("contracts") or []:
            p = c.get("proposer", "?")
            counts[p]["n"] += 1
            if c.get("verified") is True:
                counts[p]["v"] += 1

    out: dict[str, dict] = {}
    for aid, d in counts.items():
        n, v = d["n"], d["v"]
        cc = v / n if n else float("nan")
        # Wilson 95% CI for a binomial proportion
        lo, hi = _wilson_ci(v, n)
        out[aid] = {"n_contracts": n, "n_verified": v,
                     "cc": cc, "ci_low": lo, "ci_high": hi}
    return out


def cc_self_cross_per_agent(records: list[dict], n_boot: int = 1000,
                            seed: int = 42) -> dict[str, dict]:
    """Self-form and cross-form CC per proposing agent.

    Self-form CC is the fraction of an agent's contracts in which its own
    post-intervention prediction matched its commitment
    (``proposer_verified``). Cross-form CC is the fraction in which its
    commitment about the acceptor's post-intervention label held
    (``acceptor_verified``). CIs are 95% percentile-bootstrap intervals
    with ``n_boot`` resamples and a fresh ``random.Random(seed)`` per series.

    Args:
      records: ProtocolResult dicts; pass the records of all three VOUCH
        conditions (see ``VOUCH_CONDITIONS``) to pool them.
    Returns:
      dict[agent_id, {mean, ci_lo, ci_hi, n,
                      cross_mean, cross_ci_lo, cross_ci_hi, cross_n}].
    """
    series: dict[str, tuple[list[int], list[int]]] = {}
    for r in records:
        for c in r.get("contracts") or []:
            self_ok, cross_ok = series.setdefault(c.get("proposer", "?"), ([], []))
            self_ok.append(1 if c.get("proposer_verified") is True else 0)
            cross_ok.append(1 if c.get("acceptor_verified") is True else 0)

    out: dict[str, dict] = {}
    for aid, (self_ok, cross_ok) in series.items():
        lo, hi = _bootstrap_mean_ci(self_ok, n_boot, seed)
        cross_lo, cross_hi = _bootstrap_mean_ci(cross_ok, n_boot, seed)
        out[aid] = {
            "mean": sum(self_ok) / len(self_ok), "ci_lo": lo, "ci_hi": hi,
            "n": len(self_ok),
            "cross_mean": sum(cross_ok) / len(cross_ok),
            "cross_ci_lo": cross_lo, "cross_ci_hi": cross_hi,
            "cross_n": len(cross_ok),
        }
    return out


def _bootstrap_mean_ci(values: list[int], n_boot: int,
                       seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------- Intervention Informativeness ----------------

def informativeness_per_intervention(records: list[dict]) -> dict[str, dict]:
    """Empirical I(iota) = Pr[verification flags an agent whose initial
    prediction was wrong | iota was applied].

    Among contracts proposed by an agent whose initial prediction
    *disagreed* with ground truth, what fraction does iota cause to fail?
    High I = diagnostic instrument; low I = signal-destroying.
    """
    iota_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "flagged": 0})
    for r in records:
        gt = _gt_str(r)
        initial_preds = r.get("initial_preds") or {}
        for c in r.get("contracts") or []:
            iota = c.get("intervention_type", "?")
            proposer = c.get("proposer")
            # Only consider contracts whose proposer was wrong on the example
            init = initial_preds.get(proposer, {}) if isinstance(initial_preds, dict) else {}
            proposer_label = init.get("label")
            if proposer_label is None or gt is None:
                continue
            if proposer_label == gt:
                continue  # proposer was correct; not a "shortcut-sensitive" candidate
            iota_counts[iota]["n"] += 1
            if c.get("verified") is False:
                iota_counts[iota]["flagged"] += 1

    out: dict[str, dict] = {}
    for iota, d in iota_counts.items():
        n, k = d["n"], d["flagged"]
        I = k / n if n else float("nan")
        lo, hi = _wilson_ci(k, n)
        out[iota] = {"n_attempts": n, "n_flagged": k,
                     "I": I, "ci_low": lo, "ci_high": hi}
    return out


# ---------------- Calibration: ECE & Brier ----------------

def ece(scores: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error over ``n_bins`` equal-width buckets."""
    if not scores or len(scores) != len(correct):
        return float("nan")
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    total = len(scores)
    ece_val = 0.0
    for lo, hi in bins:
        mask = [(lo <= s < hi or (hi == 1.0 and s == 1.0)) for s in scores]
        idxs = [i for i, m in enumerate(mask) if m]
        if not idxs:
            continue
        avg_conf = sum(scores[i] for i in idxs) / len(idxs)
        avg_acc = sum(1 for i in idxs if correct[i]) / len(idxs)
        ece_val += (len(idxs) / total) * abs(avg_conf - avg_acc)
    return ece_val


def brier_score(scores: list[float], correct: list[bool]) -> float:
    if not scores or len(scores) != len(correct):
        return float("nan")
    return sum((s - (1.0 if c else 0.0)) ** 2 for s, c in zip(scores, correct)) / len(scores)


def reliability_curve(scores: list[float], correct: list[bool],
                       n_bins: int = 10) -> list[tuple[float, float, int]]:
    """Returns list of (bin_center, accuracy, count) for plotting."""
    if not scores:
        return []
    out: list[tuple[float, float, int]] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idxs = [j for j, s in enumerate(scores)
                if lo <= s < hi or (hi == 1.0 and s == 1.0)]
        if not idxs:
            continue
        acc = sum(1 for j in idxs if correct[j]) / len(idxs)
        center = (lo + hi) / 2.0
        out.append((center, acc, len(idxs)))
    return out


# ---------------- multilabel F1 ----------------

def _to_set(v) -> set[str]:
    if v is None:
        return set()
    if isinstance(v, (list, tuple, set)):
        return {str(x).strip().lower() for x in v if str(x).strip()}
    return {p.strip().lower() for p in str(v).split(",") if p.strip()}


def multilabel_f1(records: list[dict], pred_key: str = "final_label",
                  gt_key: str = "ground_truth") -> dict[str, float]:
    """Per-record set-F1 then micro and macro aggregates.

    Returns ``{micro_f1, macro_f1, exact_match, n}``.
    Suitable for MM-IMDb genre-classification accuracy reporting.
    """
    tps = fps = fns = 0
    per_record_f1: list[float] = []
    n_exact = 0
    for r in records:
        pred = _to_set(r.get(pred_key))
        gt = _to_set(r.get(gt_key))
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        tps += tp
        fps += fp
        fns += fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_record_f1.append(f1)
        if pred == gt and pred:
            n_exact += 1
    micro_prec = tps / (tps + fps) if (tps + fps) else 0.0
    micro_rec = tps / (tps + fns) if (tps + fns) else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                if (micro_prec + micro_rec) else 0.0)
    macro_f1 = sum(per_record_f1) / max(len(per_record_f1), 1)
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "exact_match": n_exact / max(len(records), 1),
        "n": len(records),
    }


# ---------------- helpers ----------------

def _gt_str(r: dict) -> str | None:
    """Coerce ground_truth into the same string space as the predictions
    (handles the 0/1 integer labels used by Hateful Memes)."""
    gt = r.get("ground_truth")
    if gt is None:
        return None
    if isinstance(gt, int):
        # Hateful-Memes-style 0/1 → strings; multi-class datasets already
        # use strings.
        return "hateful" if gt == 1 else "not_hateful"
    return str(gt)
