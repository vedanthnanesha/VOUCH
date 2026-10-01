"""Format evaluation summary JSON files into LaTeX cells, and run the
post-hoc analyses over the classification JSONLs.

Default mode reads results/summary_<dataset>_<mode>.json files and prints:
  - LaTeX-ready cell strings for the main results table
  - Contract analysis numbers for the Contract Analysis subsection

``--analyses`` writes results/analysis/summary.json (per-agent self/cross
CC, per-intervention informativeness, cost, multi-label F1, paired tests,
and weight trajectories), which the figure scripts read.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def cell(d: dict | None) -> str:
    if d is None:
        return "---"
    acc = d.get("accuracy")
    if acc is None:
        return "---"
    lo = d.get("ci_low", acc)
    hi = d.get("ci_high", acc)
    return f"{acc*100:.1f}\\,\\scriptsize{{[{lo*100:.0f},{hi*100:.0f}]}}"


def load(dataset: str, mode: str) -> dict | None:
    p = RES / f"summary_{dataset}_{mode}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def get(s: dict | None, *keys):
    """Navigate nested dict; return None if any key missing."""
    if s is None:
        return None
    for k in keys:
        if not isinstance(s, dict) or k not in s:
            return None
        s = s[k]
    return s


def main():
    hm_c = load("hateful_memes", "clean")
    hm_a = load("hateful_memes", "adversarial")
    cm_c = load("crisismmd", "clean")
    cm_a = load("crisismmd", "adversarial")

    # ---- per-cell values for the table ----
    rows = [
        ("Independent (text)",
         get(hm_c, "independent", "per_agent", "llama_text"),
         get(hm_a, "independent", "per_agent", "llama_text"),
         get(cm_c, "independent", "per_agent", "llama_text"),
         get(cm_a, "independent", "per_agent", "llama_text")),
        ("Independent (image)",
         get(hm_c, "independent", "per_agent", "qwen_image"),
         get(hm_a, "independent", "per_agent", "qwen_image"),
         get(cm_c, "independent", "per_agent", "qwen_image"),
         get(cm_a, "independent", "per_agent", "qwen_image")),
        ("Independent (multimodal)",
         get(hm_c, "independent", "per_agent", "qwen_multimodal"),
         get(hm_a, "independent", "per_agent", "qwen_multimodal"),
         get(cm_c, "independent", "per_agent", "qwen_multimodal"),
         get(cm_a, "independent", "per_agent", "qwen_multimodal")),
        ("Ensemble",
         get(hm_c, "ensemble"), get(hm_a, "ensemble"),
         get(cm_c, "ensemble"), get(cm_a, "ensemble")),
        ("Free Deliberation",
         get(hm_c, "free_deliberation"), get(hm_a, "free_deliberation"),
         get(cm_c, "free_deliberation"), get(cm_a, "free_deliberation")),
        ("Centralized-Single",
         get(hm_c, "centralized_single"), get(hm_a, "centralized_single"),
         get(cm_c, "centralized_single"), get(cm_a, "centralized_single")),
        ("Centralized-Coord.",
         get(hm_c, "centralized_coordinator"), get(hm_a, "centralized_coordinator"),
         get(cm_c, "centralized_coordinator"), get(cm_a, "centralized_coordinator")),
        ("VOUCH",
         get(hm_c, "contracts"), get(hm_a, "contracts"),
         get(cm_c, "contracts"), get(cm_a, "contracts")),
    ]

    print("=" * 80)
    print("LaTeX cells for results table:")
    print("=" * 80)
    for name, *cells in rows:
        latex_row = " & ".join([name] + [cell(c) for c in cells]) + " \\\\"
        print(latex_row)

    # ---- contract analysis numbers ----
    print()
    print("=" * 80)
    print("Contract analysis (HM clean):")
    print("=" * 80)
    for label, s in [("HM clean", hm_c), ("HM adversarial", hm_a),
                     ("CMMD clean", cm_c), ("CMMD adversarial", cm_a)]:
        cstats = get(s, "contracts", "contracts_stats")
        cqual = get(s, "contracts", "contract_quality")
        if not cstats:
            continue
        n = get(s, "contracts", "n")
        print(f"\n--- {label} (n={n}) ---")
        print(f"  total_proposed: {cstats['total_proposed']}")
        print(f"  verified: {cstats['verified']}  "
              f"failed: {cstats['failed']}  "
              f"rejected: {cstats['rejected']}")
        v_rate = cstats['verified'] / max(cstats['total_proposed'], 1)
        f_rate = cstats['failed'] / max(cstats['total_proposed'], 1)
        print(f"  verification rate: {v_rate:.0%}; failure rate: {f_rate:.0%}")
        print(f"  agreed_immediately: {cstats['agreed_immediately']}/{n}")
        print(f"  interventions: {cstats['interventions_used']}")
        if cqual:
            print(f"  quality:")
            print(f"    correct+verified={cqual['correct_proposer_verified']} "
                  f"correct+failed={cqual['correct_proposer_failed']} "
                  f"correct+rejected={cqual['correct_proposer_rejected']}")
            print(f"    wrong+verified={cqual['wrong_proposer_verified']} "
                  f"wrong+failed={cqual['wrong_proposer_failed']} "
                  f"wrong+rejected={cqual['wrong_proposer_rejected']}")


def run_analyses(datasets: list[str] | None = None,
                 regimes: list[str] | None = None) -> None:
    """Run the post-hoc classification analyses end-to-end and dump
    everything to results/analysis/summary.json. Imports are lazy so the
    default ``main`` keeps working even if matplotlib is unavailable."""
    from experiments.analysis import metrics, tests, cost as cost_mod, longitudinal

    datasets = datasets or ["hateful_memes", "crisismmd", "mm_imdb"]
    regimes = regimes or ["clean", "adversarial"]

    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    big: dict = {"per_dataset": {}}
    for ds in datasets:
        per_regime: dict = {}
        for reg in regimes:
            jsonl = ROOT / "results" / f"main_eval_{ds}_{reg}.jsonl"
            if not jsonl.exists():
                continue
            recs_contracts = metrics.load_jsonl(jsonl, condition="contracts")
            recs_vouch = [r for r in metrics.load_jsonl(jsonl)
                          if r.get("condition") in metrics.VOUCH_CONDITIONS]
            per_regime[reg] = {
                "n_contracts_records": len(recs_contracts),
                "cc_per_agent": metrics.cc_self_cross_per_agent(recs_vouch),
                "informativeness": metrics.informativeness_per_intervention(recs_contracts),
                "cost": cost_mod.cost_table(jsonl),
            }
            # Multi-label F1 per condition (only meaningful for MM-IMDb but
            # safe to compute everywhere — collapses to identity for single-
            # label tasks).
            f1_by_cond: dict = {}
            for cond_name in ["independent", "ensemble", "contracts",
                              "contracts_random", "contracts_adaptive",
                              "centralized_single", "centralized_coordinator"]:
                recs = metrics.load_jsonl(jsonl, condition=cond_name)
                if recs:
                    f1_by_cond[cond_name] = metrics.multilabel_f1(recs)
            per_regime[reg]["f1_by_condition"] = f1_by_cond
            # paired McNemar between Ensemble and Contracts (where both ran)
            try:
                a, b = tests.paired_decisions(jsonl, jsonl,
                                              condition_a="ensemble",
                                              condition_b="contracts")
                if a and b:
                    per_regime[reg]["mcnemar_ens_vs_contracts"] = tests.mcnemar_exact(a, b)
                    per_regime[reg]["permutation_ens_vs_contracts"] = tests.permutation_test(a, b)
            except Exception as e:  # noqa: BLE001
                per_regime[reg]["pairwise_error"] = repr(e)
            # longitudinal weight trajectories
            try:
                per_regime[reg]["weight_trajectory"] = longitudinal.weight_trajectory(jsonl)
            except Exception as e:  # noqa: BLE001
                per_regime[reg]["weight_trajectory_error"] = repr(e)
        if per_regime:
            big["per_dataset"][ds] = per_regime

    out_json = out / "summary.json"
    out_json.write_text(json.dumps(big, indent=2, default=str))
    print(f"[analyses] summary -> {out_json}")


if __name__ == "__main__":
    import argparse as _argparse
    ap = _argparse.ArgumentParser()
    ap.add_argument("--analyses", action="store_true",
                    help="Run the post-hoc analyses (CC, I, cost, F1, paired "
                         "tests, weight trajectories) -> results/analysis/summary.json")
    args, _ = ap.parse_known_args()
    if args.analyses:
        run_analyses()
    else:
        main()
