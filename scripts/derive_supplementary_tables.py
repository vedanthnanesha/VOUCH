"""Derive supplementary tables used in appendix C.18 and C.19.

Outputs (deterministic from results/*.jsonl + results/baselines/summary_*.json):

  results/derived/judge_baseline_table.csv
    Per-cell judge baseline (standard + adv-judge) — basis for Tab.~C.12.

  results/derived/intervention_fire_rate.csv
    Per-intervention dual-side pass rate across the math cells — basis for
    Tab.~C.19.

Re-run after any change to results/baselines/summary_judge_*.json or the
math dual-side JSONLs.
"""
from __future__ import annotations
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "results" / "derived"


def write_judge_baseline_table() -> Path:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/baselines/summary_judge_*.json"))):
        s = json.load(open(f))
        base = os.path.basename(f).replace("summary_judge_", "").replace(".json", "")
        ds = base.split("_")[0].upper()
        regime = "adv" if "adversarial" in base else "clean"
        judge_kind = "adv-judge" if base.endswith("_advjudge") else "standard"
        rows.append({
            "dataset":      ds,
            "regime":       regime,
            "judge_kind":   judge_kind,
            "n":            s["n"],
            "n_disagree":   s["n_disagreement_cases"],
            "judge_acc":    round(s["judge_accuracy"], 4),
            "agent_a_acc":  round(s["agent_a_accuracy"], 4),
            "agent_b_acc":  round(s["agent_b_accuracy"], 4),
            "pick_a_rate":  round(s["pick_a_rate"], 4),
            "pick_b_rate":  round(s["pick_b_rate"], 4),
            "abstain_rate": round(s["abstain_rate"], 4),
        })
    order = {"GSM8K": 0, "MATH": 1, "SVAMP": 2}
    rows.sort(key=lambda r: (order.get(r["dataset"], 99),
                              r["regime"], r["judge_kind"]))
    out = DERIVED / "judge_baseline_table.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    return out


# (cell display name, dual-side JSONL filename)
MATH_DUAL_FILES = [
    ("GSM8K",      "gsm8k_math_dual_n300.jsonl"),
    ("GSM8K-adv",  "gsm8k_math_dual_adversarial_n300.jsonl"),
    ("SVAMP",      "svamp_math_dual_n300.jsonl"),
    ("SVAMP-adv",  "svamp_math_dual_adversarial_n300.jsonl"),
    ("MATH",       "math_math_dual_n300.jsonl"),
    ("MATH-adv",   "math_math_dual_adversarial_n300.jsonl"),
]


def write_intervention_fire_rate() -> Path:
    rows_out = []
    for cell, fname in MATH_DUAL_FILES:
        path = ROOT / "results" / fname
        if not path.exists():
            print(f"  skip {cell}: missing {fname}")
            continue
        by_iv: dict[str, dict] = defaultdict(
            lambda: {"n_attempted": 0, "n_emitted": 0,
                     "n_dual_pass": 0, "n_single_pass": 0})
        n_total = n_emitted = 0
        for line in path.open():
            r = json.loads(line)
            iv = r.get("intervention", "<none>")
            slot = by_iv[iv]
            slot["n_attempted"] += 1
            n_total += 1
            if r.get("contract_ok"):
                slot["n_emitted"] += 1
                n_emitted += 1
                if r.get("verified_dual"):
                    slot["n_dual_pass"] += 1
                if r.get("verified_single_side"):
                    slot["n_single_pass"] += 1
        for iv, s in sorted(by_iv.items(), key=lambda kv: -kv[1]["n_emitted"]):
            if s["n_attempted"] == 0:
                continue
            rows_out.append({
                "cell": cell,
                "intervention": iv,
                "n_attempted": s["n_attempted"],
                "n_emitted": s["n_emitted"],
                "fire_rate":               round(s["n_emitted"] / max(n_emitted, 1), 4),
                "attempt_rate":            round(s["n_attempted"] / max(n_total, 1), 4),
                "dual_pass_per_emitted":   (round(s["n_dual_pass"] / max(s["n_emitted"], 1), 4)
                                            if s["n_emitted"] else None),
                "single_pass_per_emitted": (round(s["n_single_pass"] / max(s["n_emitted"], 1), 4)
                                            if s["n_emitted"] else None),
            })
    out = DERIVED / "intervention_fire_rate.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"wrote {out}  ({len(rows_out)} rows)")
    return out


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    write_judge_baseline_table()
    write_intervention_fire_rate()


if __name__ == "__main__":
    main()
