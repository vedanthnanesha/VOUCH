"""Re-analysis script that computes multiple correctness metrics from a
completed JSONL.

Works on any dataset, but the metrics are most informative on multi-label
tasks (MM-IMDb) where strict subset equality is too brittle.

For each condition we report:
  - subset_accuracy:  pred_set == gt_set (current default in main_evaluation)
  - partial_accuracy: pred_set & gt_set != empty
  - jaccard:          |pred_set & gt_set| / |pred_set | gt_set|, mean over rows
  - macro_f1:         per-label F1 across the union of seen labels, mean
  - micro_f1:         F1 over pooled label predictions

For the contracts condition we also recompute the 6-cell contract quality
breakdown under three correctness definitions for the proposer's initial
prediction: strict, partial, and f1>=0.5.

Run:
  python -m experiments.reanalyze_multilabel \\
      --jsonl results/main_eval_mm_imdb_adversarial.jsonl \\
      --out results/tables/mm_imdb_adv_metrics.txt
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def to_set(label_value: Any) -> set[str]:
    """Coerce a label value (single str or list/tuple, ints for binary, etc.)
    into a normalized lowercase token set.
    """
    if label_value is None:
        return set()
    if isinstance(label_value, (list, tuple, set)):
        return {str(x).strip().lower() for x in label_value if str(x).strip()}
    if isinstance(label_value, int):
        # binary mapping used by HM
        return {"hateful"} if label_value == 1 else {"not_hateful"}
    s = str(label_value).strip().lower()
    if "," in s:
        return {p.strip() for p in s.split(",") if p.strip()}
    if not s or s == "unknown":
        return set()
    return {s}


def subset_eq(pred: set[str], gt: set[str]) -> bool:
    return bool(gt) and pred == gt


def partial_match(pred: set[str], gt: set[str]) -> bool:
    return bool(gt) and bool(pred & gt)


def jaccard(pred: set[str], gt: set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not (pred | gt):
        return 0.0
    return len(pred & gt) / len(pred | gt)


def example_f1(pred: set[str], gt: set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gt)
    return 2 * p * r / (p + r)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n) + (z * z / (4 * n * n))) ** 0.5 / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def per_condition_metrics(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-condition: subset_acc, partial_acc, mean Jaccard, mean F1.
    For single-label tasks subset_acc == partial_acc and Jaccard == F1.
    """
    per_cond: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {
        "subset": [], "partial": [], "jaccard": [], "f1": [], "n": 0,
    })
    for r in rows:
        c = r["condition"]
        gt = to_set(r.get("ground_truth"))
        pred = to_set(r.get("final_label"))
        per_cond[c]["n"] += 1
        per_cond[c]["subset"].append(int(subset_eq(pred, gt)))
        per_cond[c]["partial"].append(int(partial_match(pred, gt)))
        per_cond[c]["jaccard"].append(jaccard(pred, gt))
        per_cond[c]["f1"].append(example_f1(pred, gt))
    out: dict[str, dict[str, Any]] = {}
    for c, d in per_cond.items():
        n = d["n"]
        out[c] = {
            "n": n,
            "subset_acc": sum(d["subset"]) / max(n, 1),
            "subset_ci": wilson_ci(sum(d["subset"]), n),
            "partial_acc": sum(d["partial"]) / max(n, 1),
            "partial_ci": wilson_ci(sum(d["partial"]), n),
            "mean_jaccard": sum(d["jaccard"]) / max(n, 1),
            "mean_f1": sum(d["f1"]) / max(n, 1),
        }
    return out


def macro_micro_f1(rows: list[dict], target_condition: str) -> dict[str, float]:
    """Macro and micro F1 across labels for one condition.
    """
    seen_labels: set[str] = set()
    preds_by_ex: list[tuple[set, set]] = []
    for r in rows:
        if r["condition"] != target_condition:
            continue
        gt = to_set(r.get("ground_truth"))
        pred = to_set(r.get("final_label"))
        seen_labels.update(gt | pred)
        preds_by_ex.append((pred, gt))
    if not preds_by_ex or not seen_labels:
        return {"macro_f1": 0.0, "micro_f1": 0.0}
    per_label_tp: dict[str, int] = defaultdict(int)
    per_label_fp: dict[str, int] = defaultdict(int)
    per_label_fn: dict[str, int] = defaultdict(int)
    for pred, gt in preds_by_ex:
        for lab in seen_labels:
            if lab in pred and lab in gt: per_label_tp[lab] += 1
            elif lab in pred and lab not in gt: per_label_fp[lab] += 1
            elif lab not in pred and lab in gt: per_label_fn[lab] += 1
    per_label_f1: list[float] = []
    total_tp = total_fp = total_fn = 0
    for lab in seen_labels:
        tp = per_label_tp[lab]; fp = per_label_fp[lab]; fn = per_label_fn[lab]
        total_tp += tp; total_fp += fp; total_fn += fn
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        if p + r == 0:
            per_label_f1.append(0.0)
        else:
            per_label_f1.append(2 * p * r / (p + r))
    macro_f1 = sum(per_label_f1) / max(len(per_label_f1), 1)
    if total_tp + total_fp + total_fn == 0:
        micro_f1 = 0.0
    else:
        p_mi = total_tp / max(total_tp + total_fp, 1)
        r_mi = total_tp / max(total_tp + total_fn, 1)
        micro_f1 = 2 * p_mi * r_mi / max(p_mi + r_mi, 1e-9)
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


def contract_quality_with_definitions(rows: list[dict]) -> dict[str, dict[str, int]]:
    """For each correctness definition, return the 6-cell contract quality table."""
    cats_by_def: dict[str, Counter] = {
        "strict":  Counter(),
        "partial": Counter(),
        "f1_05":   Counter(),
    }
    for r in rows:
        if r["condition"] != "contracts":
            continue
        gt = to_set(r.get("ground_truth"))
        for c in r.get("contracts", []):
            proposer = c.get("proposer")
            proposer_label = (r.get("initial_preds") or {}).get(proposer, {}).get("label")
            pred_set = to_set(proposer_label)
            judgments = {
                "strict":  subset_eq(pred_set, gt),
                "partial": partial_match(pred_set, gt),
                "f1_05":   example_f1(pred_set, gt) >= 0.5,
            }
            for defn, correct in judgments.items():
                if c.get("acceptance") is False:
                    bucket = "correct_rejected" if correct else "wrong_rejected"
                elif c.get("verified") is True:
                    bucket = "correct_verified" if correct else "wrong_verified"
                elif c.get("verified") is False:
                    bucket = "correct_failed" if correct else "wrong_failed"
                else:
                    bucket = "other"
                cats_by_def[defn][bucket] += 1
    return {defn: dict(cats) for defn, cats in cats_by_def.items()}


def emit_text_report(metrics: dict, mf1: dict, contracts: dict, source: str,
                      out_path: Path) -> None:
    order = [
        "independent", "ensemble", "centralized_single",
        "centralized_coordinator", "contracts", "contracts_random",
        "contracts_adaptive",
    ]
    lines: list[str] = []
    lines.append(f"# Re-analysis report")
    lines.append(f"# source: {source}")
    lines.append("")
    lines.append("Per-condition accuracy with four metrics")
    lines.append(f"{'condition':<26}{'subset':>9}{'partial':>10}{'jaccard':>10}{'mean_f1':>10}{'n':>6}")
    for c in order:
        if c not in metrics: continue
        m = metrics[c]
        lines.append(
            f"{c:<26}"
            f"{m['subset_acc']:>9.3f}{m['partial_acc']:>10.3f}"
            f"{m['mean_jaccard']:>10.3f}{m['mean_f1']:>10.3f}{m['n']:>6d}"
        )
    lines.append("")
    lines.append("Macro and micro F1 for contracts condition")
    lines.append(f"  macro_f1 = {mf1.get('contracts', {}).get('macro_f1', 0):.3f}")
    lines.append(f"  micro_f1 = {mf1.get('contracts', {}).get('micro_f1', 0):.3f}")
    lines.append("")
    lines.append("Contract quality breakdown under three correctness definitions")
    lines.append("(strict = pred_set == gt_set; partial = pred_set & gt_set != empty; f1_05 = per-example F1 >= 0.5)")
    for defn in ("strict", "partial", "f1_05"):
        cats = contracts.get(defn, {})
        total = sum(cats.values())
        lines.append("")
        lines.append(f"  definition: {defn}  (total contracts: {total})")
        for k in ("correct_verified", "correct_failed", "correct_rejected",
                   "wrong_verified", "wrong_failed", "wrong_rejected"):
            lines.append(f"    {k:<22} {cats.get(k, 0):>4}")
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    metrics = per_condition_metrics(rows)
    mf1 = {}
    for c in metrics:
        mf1[c] = macro_micro_f1(rows, c)
    contracts = contract_quality_with_definitions(rows)
    out = args.out or (Path("results/tables") / (args.jsonl.stem + "_metrics.txt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    emit_text_report(metrics, mf1, contracts, str(args.jsonl), out)
    print(f"\n[reanalyze] report -> {out}")


if __name__ == "__main__":
    main()
