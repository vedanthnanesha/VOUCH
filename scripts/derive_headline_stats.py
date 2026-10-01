"""Derive canonical (dataset, condition, model_pair) headline statistics from
every results JSONL in ``results/`` so the paper's headline numbers
can be regenerated from one place.

Writes:
  results/derived/headline_table.csv          -- one row per (dataset, condition, pair)
  results/derived/abstention_disentangled.csv -- action-rate vs catch-rate
  results/derived/heterogeneity_panel.csv     -- same-family vs cross-family
  results/derived/intervention_informativeness.csv  -- per-intervention flip
  results/derived/calibration_metrics.csv     -- Brier + ECE for confidence vs CC

Reads only existing JSONLs. No model calls.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DERIVED = RESULTS / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)


# -- file -> (domain, dataset, condition, model_pair) --------------------
# model_pair encodes the (proposer/text agent, acceptor/multimodal agent)
# family combination used to produce the file.

MULTIMODAL_FILES = {
    "main_eval_hateful_memes_clean.jsonl":
        ("multimodal", "HM",        "clean",    "samefamily_qwen+qwen"),
    "main_eval_hateful_memes_adversarial.jsonl":
        ("multimodal", "HM",        "adv",      "samefamily_qwen+qwen"),
    "cross_family_eval_hateful_memes_clean.jsonl":
        ("multimodal", "HM",        "clean",    "crossfamily_llama+qwen"),
    "cross_family_eval_hateful_memes_adversarial.jsonl":
        ("multimodal", "HM",        "adv",      "crossfamily_llama+qwen"),
    "main_eval_crisismmd_clean.jsonl":
        ("multimodal", "CrisisMMD", "clean",    "samefamily_qwen+qwen"),
    "main_eval_crisismmd_adversarial.jsonl":
        ("multimodal", "CrisisMMD", "adv",      "samefamily_qwen+qwen"),
    "main_eval_mm_imdb_clean.jsonl":
        ("multimodal", "MM-IMDb",   "clean",    "samefamily_qwen+qwen"),
    "main_eval_mm_imdb_adversarial.jsonl":
        ("multimodal", "MM-IMDb",   "adv",      "samefamily_qwen+qwen"),
}

# Math dual-side: Qwen2.5-7B proposer + Llama-3.1-8B acceptor, always cross-family.
MATH_DUAL_FILES = {
    "gsm8k_math_dual_n300.jsonl":
        ("math", "GSM8K-alg", "clean", "crossfamily_qwen+llama"),
    "gsm8k_math_dual_adversarial_n300.jsonl":
        ("math", "GSM8K-alg", "adv",   "crossfamily_qwen+llama"),
    "svamp_math_dual_n300.jsonl":
        ("math", "SVAMP",      "clean", "crossfamily_qwen+llama"),
    "svamp_math_dual_adversarial_n300.jsonl":
        ("math", "SVAMP",      "adv",   "crossfamily_qwen+llama"),
    "math_math_dual_n300.jsonl":
        ("math", "MATH-alg",   "clean", "crossfamily_qwen+llama"),
    "math_math_dual_adversarial_n300.jsonl":
        ("math", "MATH-alg",   "adv",   "crossfamily_qwen+llama"),
}

CODEGEN_DUAL_FILES = {
    "mbpp_codegen_dual_n200.jsonl":
        ("code", "MBPP",      "clean", "crossfamily_qwen+llama"),
    "mbpp_codegen_dual_adversarial_n200.jsonl":
        ("code", "MBPP",      "adv",   "crossfamily_qwen+llama"),
    "humaneval_codegen_dual_n164.jsonl":
        ("code", "HumanEval", "clean", "crossfamily_qwen+llama"),
    "humaneval_codegen_dual_adversarial_n164.jsonl":
        ("code", "HumanEval", "adv",   "crossfamily_qwen+llama"),
}

# Single-side (self-form CC) files used as baselines for the dual-side
# comparison. These ALSO show the self-coherence pathology directly.
MATH_SINGLE_FILES = {
    "gsm8k_math.jsonl":             ("math", "GSM8K-alg", "clean", "selfform_qwen"),
    "gsm8k_math_adversarial.jsonl": ("math", "GSM8K-alg", "adv",   "selfform_qwen"),
    "svamp_math.jsonl":             ("math", "SVAMP",      "clean", "selfform_qwen"),
    "svamp_math_adversarial.jsonl": ("math", "SVAMP",      "adv",   "selfform_qwen"),
    "math_math.jsonl":              ("math", "MATH-alg",   "clean", "selfform_qwen"),
    "math_math_adversarial.jsonl":  ("math", "MATH-alg",   "adv",   "selfform_qwen"),
}

CODEGEN_SINGLE_FILES = {
    "mbpp_codegen_clean_n500.jsonl":
        ("code", "MBPP",      "clean", "selfform_qwen"),
    "mbpp_codegen_adv_n500.jsonl":
        ("code", "MBPP",      "adv",   "selfform_qwen"),
    "humaneval_codegen_clean_n164.jsonl":
        ("code", "HumanEval", "clean", "selfform_qwen"),
    "humaneval_codegen_adv_n164.jsonl":
        ("code", "HumanEval", "adv",   "selfform_qwen"),
}


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ===================================================================
# 1. Headline table -- one row per (dataset, condition, pair)
# ===================================================================


def stats_dual_math_or_code(path: Path) -> dict:
    """Compute headline stats for a dual-side math or code JSONL.

    Definitions used throughout the paper:
      n_total          : rows in the file
      n_emitted        : rows where contract_ok is True (proposer produced a
                          parseable dual contract); rows with
                          contract_parse_failed are NOT emitted
      n_verified_single: emitted and verified_single_side
      n_verified_dual  : emitted and verified_dual
      n_initial_correct_a: proposer was correct before any contract
      hurt_single / hurt_dual: VERIFIED contract on a row where proposer was
                          initially wrong (i.e. the contract endorsed a wrong
                          answer)
      caught_single / caught_dual: NOT-verified contract on a row where
                          proposer was initially wrong (i.e. the contract
                          successfully flagged a wrong answer)
      helped_single / helped_dual: VERIFIED contract on a row where proposer
                          was initially correct
      missed_single / missed_dual: NOT-verified contract on a row where
                          proposer was initially correct (over-rejection)
    """
    n_total = n_emitted = 0
    n_initial_correct_a = n_initial_correct_b = 0
    n_verified_single = n_verified_dual = 0
    hurt_s = caught_s = helped_s = missed_s = 0
    hurt_d = caught_d = helped_d = missed_d = 0

    for rec in _iter_jsonl(path):
        n_total += 1
        if "initial_correct_a" in rec:
            a_correct = bool(rec["initial_correct_a"])
        else:
            v0_a = str(rec.get("v0_a", "")).upper()
            a_correct = bool(v0_a) and "F" not in v0_a
        b_correct = bool(rec.get("initial_correct_b"))
        if a_correct:
            n_initial_correct_a += 1
        if b_correct:
            n_initial_correct_b += 1
        if not rec.get("contract_ok"):
            continue
        n_emitted += 1
        v_s = bool(rec.get("verified_single_side"))
        v_d = bool(rec.get("verified_dual"))
        if v_s:
            n_verified_single += 1
        if v_d:
            n_verified_dual += 1
        if v_s:
            if a_correct:
                helped_s += 1
            else:
                hurt_s += 1
        else:
            if a_correct:
                missed_s += 1
            else:
                caught_s += 1
        if v_d:
            if a_correct:
                helped_d += 1
            else:
                hurt_d += 1
        else:
            if a_correct:
                missed_d += 1
            else:
                caught_d += 1

    return {
        "n_total": n_total,
        "n_emitted": n_emitted,
        "n_initial_correct_a": n_initial_correct_a,
        "n_initial_correct_b": n_initial_correct_b,
        "n_verified_single": n_verified_single,
        "n_verified_dual":   n_verified_dual,
        "single_helped": helped_s, "single_hurt": hurt_s,
        "single_caught": caught_s, "single_missed": missed_s,
        "dual_helped":   helped_d, "dual_hurt":   hurt_d,
        "dual_caught":   caught_d, "dual_missed": missed_d,
    }


def stats_single_math_or_code(path: Path) -> dict:
    """Stats for a self-form (single-side) math or code JSONL.

    Single-side runs have only ``side_a_verified`` (or ``verified``) and
    ``initial_correct_a``. We re-use the same hurt/caught semantics so
    self-form and dual-form numbers are directly comparable.
    """
    n_total = n_emitted = 0
    n_initial_correct = 0
    n_verified = 0
    hurt = caught = helped = missed = 0

    for rec in _iter_jsonl(path):
        n_total += 1
        a_correct = bool(rec.get("initial_correct_a")
                          or rec.get("initial_correct"))
        if a_correct:
            n_initial_correct += 1
        if not rec.get("contract_ok"):
            continue
        n_emitted += 1
        v = bool(rec.get("verified")
                 or rec.get("verified_single_side")
                 or rec.get("side_a_verified"))
        if v:
            n_verified += 1
            if a_correct:
                helped += 1
            else:
                hurt += 1
        else:
            if a_correct:
                missed += 1
            else:
                caught += 1

    return {
        "n_total": n_total,
        "n_emitted": n_emitted,
        "n_initial_correct_a": n_initial_correct,
        "n_initial_correct_b": None,
        "n_verified_single": n_verified,
        "n_verified_dual":   None,
        "single_helped": helped, "single_hurt": hurt,
        "single_caught": caught, "single_missed": missed,
        "dual_helped":   None,   "dual_hurt":   None,
        "dual_caught":   None,   "dual_missed": None,
    }


def stats_multimodal(path: Path) -> dict:
    """Stats for a multimodal classification JSONL.

    Multimodal records do NOT carry verified_dual / verified_single -- they
    carry final accuracy under each ``condition`` (independent, ensemble,
    centralized_*, contracts*). We persist per-condition accuracy and
    contract-emission rate (rows where len(contracts)>0 under the
    ``contracts*`` conditions).
    """
    per_cond_total: Counter[str] = Counter()
    per_cond_correct: Counter[str] = Counter()
    per_cond_emitted: Counter[str] = Counter()   # contract emitted at all
    per_cond_n_contracts: Counter[str] = Counter()
    for rec in _iter_jsonl(path):
        cond = rec.get("condition", "")
        per_cond_total[cond] += 1
        if rec.get("correct"):
            per_cond_correct[cond] += 1
        if cond.startswith("contracts"):
            contracts = rec.get("contracts") or []
            if isinstance(contracts, list) and len(contracts) > 0:
                per_cond_emitted[cond] += 1
                per_cond_n_contracts[cond] += len(contracts)

    rows = {}
    for cond, n in per_cond_total.items():
        rows[cond] = {
            "n_total": n,
            "n_correct": per_cond_correct[cond],
            "n_contracts_emitted_rows": per_cond_emitted[cond],
            "n_contracts_total":        per_cond_n_contracts[cond],
        }
    return rows


def write_headline_table() -> Path:
    out = DERIVED / "headline_table.csv"
    rows: list[dict] = []

    # multimodal: one row per (file, condition)
    for fname, (domain, dataset, regime, pair) in MULTIMODAL_FILES.items():
        p = RESULTS / fname
        if not p.exists():
            continue
        per_cond = stats_multimodal(p)
        for cond, st in per_cond.items():
            rows.append({
                "source_file": fname,
                "domain": domain,
                "dataset": dataset,
                "regime": regime,
                "model_pair": pair,
                "condition": cond,
                "n_total": st["n_total"],
                "n_correct": st["n_correct"],
                "n_emitted": st["n_contracts_emitted_rows"]
                              if cond.startswith("contracts") else "",
                "n_contracts_total": st["n_contracts_total"]
                                       if cond.startswith("contracts") else "",
                "n_initial_correct_a": "",
                "n_verified_single": "",
                "n_verified_dual":   "",
                "single_helped": "", "single_hurt": "",
                "single_caught": "", "single_missed": "",
                "dual_helped":   "", "dual_hurt":   "",
                "dual_caught":   "", "dual_missed": "",
            })

    for fname, (domain, dataset, regime, pair) in {
        **MATH_DUAL_FILES, **CODEGEN_DUAL_FILES,
    }.items():
        p = RESULTS / fname
        if not p.exists():
            continue
        st = stats_dual_math_or_code(p)
        rows.append({
            "source_file": fname,
            "domain": domain, "dataset": dataset, "regime": regime,
            "model_pair": pair, "condition": "dual_vouch",
            "n_total": st["n_total"], "n_correct": "",
            "n_emitted": st["n_emitted"],
            "n_contracts_total": st["n_emitted"],
            "n_initial_correct_a": st["n_initial_correct_a"],
            "n_verified_single":  st["n_verified_single"],
            "n_verified_dual":    st["n_verified_dual"],
            "single_helped": st["single_helped"],
            "single_hurt":   st["single_hurt"],
            "single_caught": st["single_caught"],
            "single_missed": st["single_missed"],
            "dual_helped":   st["dual_helped"],
            "dual_hurt":     st["dual_hurt"],
            "dual_caught":   st["dual_caught"],
            "dual_missed":   st["dual_missed"],
        })

    for fname, (domain, dataset, regime, pair) in {
        **MATH_SINGLE_FILES, **CODEGEN_SINGLE_FILES,
    }.items():
        p = RESULTS / fname
        if not p.exists():
            continue
        st = stats_single_math_or_code(p)
        rows.append({
            "source_file": fname,
            "domain": domain, "dataset": dataset, "regime": regime,
            "model_pair": pair, "condition": "single_vouch",
            "n_total": st["n_total"], "n_correct": "",
            "n_emitted": st["n_emitted"],
            "n_contracts_total": st["n_emitted"],
            "n_initial_correct_a": st["n_initial_correct_a"],
            "n_verified_single":  st["n_verified_single"],
            "n_verified_dual":    "",
            "single_helped": st["single_helped"],
            "single_hurt":   st["single_hurt"],
            "single_caught": st["single_caught"],
            "single_missed": st["single_missed"],
            "dual_helped":   "", "dual_hurt":   "",
            "dual_caught":   "", "dual_missed": "",
        })

    cols = ["domain", "dataset", "regime", "model_pair", "condition",
            "n_total", "n_correct", "n_emitted", "n_contracts_total",
            "n_initial_correct_a",
            "n_verified_single", "n_verified_dual",
            "single_helped", "single_hurt", "single_caught", "single_missed",
            "dual_helped",   "dual_hurt",   "dual_caught",   "dual_missed",
            "source_file"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    return out


# ===================================================================
# 2. Abstention disentanglement
# ===================================================================


def write_abstention_table() -> Path:
    """For each dual-side (math+code) cell, report:
       action_rate     = n_emitted / n_total
       catch_per_acted = caught_dual / (caught_dual + hurt_dual)
                          [i.e. precision-of-catch among acted-on-wrong cases]
       catch_per_total = caught_dual / n_total
       hurt_per_acted  = hurt_dual / (caught_dual + hurt_dual)
       and the analogous quantities for single-side.
    """
    out = DERIVED / "abstention_disentangled.csv"
    rows: list[dict] = []
    for fname, (domain, dataset, regime, pair) in {
        **MATH_DUAL_FILES, **CODEGEN_DUAL_FILES,
    }.items():
        p = RESULTS / fname
        if not p.exists():
            continue
        st = stats_dual_math_or_code(p)

        def _ratio(num, den):
            return (num / den) if den else None

        s_acted_wrong = st["single_caught"] + st["single_hurt"]
        d_acted_wrong = st["dual_caught"]   + st["dual_hurt"]
        rows.append({
            "domain": domain, "dataset": dataset, "regime": regime,
            "model_pair": pair,
            "n_total":   st["n_total"],
            "n_emitted": st["n_emitted"],
            "action_rate": _ratio(st["n_emitted"], st["n_total"]),

            "single_catch_per_acted_wrong":
                _ratio(st["single_caught"], s_acted_wrong),
            "single_hurt_per_acted_wrong":
                _ratio(st["single_hurt"],   s_acted_wrong),
            "single_catch_per_total":
                _ratio(st["single_caught"], st["n_total"]),
            "single_hurt_per_total":
                _ratio(st["single_hurt"],   st["n_total"]),

            "dual_catch_per_acted_wrong":
                _ratio(st["dual_caught"], d_acted_wrong),
            "dual_hurt_per_acted_wrong":
                _ratio(st["dual_hurt"],   d_acted_wrong),
            "dual_catch_per_total":
                _ratio(st["dual_caught"], st["n_total"]),
            "dual_hurt_per_total":
                _ratio(st["dual_hurt"],   st["n_total"]),

            "n_single_helped": st["single_helped"], "n_single_hurt": st["single_hurt"],
            "n_single_caught": st["single_caught"], "n_single_missed": st["single_missed"],
            "n_dual_helped":   st["dual_helped"],   "n_dual_hurt":   st["dual_hurt"],
            "n_dual_caught":   st["dual_caught"],   "n_dual_missed": st["dual_missed"],
            "source_file": fname,
        })
    cols = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    return out


# ===================================================================
# 3. Heterogeneity panel
# ===================================================================


def write_heterogeneity_panel() -> Path:
    """For each multimodal dataset that has both same-family and cross-family
    files, side-by-side accuracy and contract-emission for the ``contracts``
    condition.
    """
    out = DERIVED / "heterogeneity_panel.csv"
    by_key: dict[tuple[str, str, str], dict] = {}
    for fname, (domain, dataset, regime, pair) in MULTIMODAL_FILES.items():
        p = RESULTS / fname
        if not p.exists():
            continue
        per_cond = stats_multimodal(p)
        family = "samefamily" if pair.startswith("samefamily") else "crossfamily"
        for cond, st in per_cond.items():
            k = (dataset, regime, cond)
            row = by_key.setdefault(k, {
                "dataset": dataset, "regime": regime, "condition": cond,
            })
            row[f"{family}_n_total"]   = st["n_total"]
            row[f"{family}_n_correct"] = st["n_correct"]
            row[f"{family}_accuracy"]  = (st["n_correct"] / st["n_total"]
                                            if st["n_total"] else None)
            if cond.startswith("contracts"):
                row[f"{family}_n_emitted"] = st["n_contracts_emitted_rows"]
                row[f"{family}_n_contracts_total"] = st["n_contracts_total"]

    rows = list(by_key.values())
    # only keep keys that have both families OR both regimes for direct comparison
    cols = ["dataset", "regime", "condition",
            "samefamily_n_total", "samefamily_n_correct", "samefamily_accuracy",
            "samefamily_n_emitted", "samefamily_n_contracts_total",
            "crossfamily_n_total", "crossfamily_n_correct", "crossfamily_accuracy",
            "crossfamily_n_emitted", "crossfamily_n_contracts_total"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out}  ({len(rows)} rows)")
    return out


# ===================================================================
# 4. Intervention informativeness (math + code dual)
# ===================================================================


def write_intervention_informativeness() -> Path:
    """For each intervention type, P(answer changes after intervention).

    Uses the dual-side math + code JSONLs because they record both the
    pre-intervention and post-intervention answer for both agents.
    Informativeness = P(a1_a_actual differs from a0_a) -- this is the
    "post-intervention behavioural shift" used in the paper.
    """
    out = DERIVED / "intervention_informativeness.csv"
    counters: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"n": 0, "a_changed": 0, "b_changed": 0})
    for fname, (domain, dataset, regime, pair) in {
        **MATH_DUAL_FILES, **CODEGEN_DUAL_FILES,
    }.items():
        p = RESULTS / fname
        if not p.exists():
            continue
        for rec in _iter_jsonl(p):
            iv = rec.get("intervention") or "<none>"
            k = (domain, dataset, iv)
            c = counters[k]
            c["n"] += 1
            if rec.get("a0_a") is not None and rec.get("a1_a_actual") is not None:
                if str(rec["a0_a"]) != str(rec["a1_a_actual"]):
                    c["a_changed"] += 1
            if rec.get("a0_b") is not None and rec.get("a1_b_actual") is not None:
                if str(rec["a0_b"]) != str(rec["a1_b_actual"]):
                    c["b_changed"] += 1

    rows = []
    for (domain, dataset, iv), c in sorted(counters.items()):
        rows.append({
            "domain": domain, "dataset": dataset, "intervention": iv,
            "n": c["n"],
            "a_change_rate": (c["a_changed"] / c["n"]) if c["n"] else None,
            "b_change_rate": (c["b_changed"] / c["n"]) if c["n"] else None,
            "mean_change_rate": ((c["a_changed"] + c["b_changed"]) /
                                   (2 * c["n"])) if c["n"] else None,
        })
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh,
            fieldnames=["domain", "dataset", "intervention", "n",
                        "a_change_rate", "b_change_rate", "mean_change_rate"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    return out


# ===================================================================
# 5. Calibration metrics: Brier + ECE for confidence vs CC.
# ===================================================================


def _brier(probs, ys) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(probs) if probs else float("nan")


def _ece(probs, ys, n_bins: int = 10) -> float:
    if not probs:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, ys):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append((p, y))
    n = len(probs)
    s = 0.0
    for b in bins:
        if not b:
            continue
        mp = sum(p for p, _ in b) / len(b)
        my = sum(y for _, y in b) / len(b)
        s += (len(b) / n) * abs(mp - my)
    return s


def write_calibration_metrics() -> Path:
    """Compute Brier / ECE per agent per multimodal dataset for confidence."""
    out = DERIVED / "calibration_metrics.csv"
    rows = []
    for fname, (domain, dataset, regime, pair) in MULTIMODAL_FILES.items():
        p = RESULTS / fname
        if not p.exists():
            continue
        agent_probs: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for rec in _iter_jsonl(p):
            if rec.get("condition") != "independent":
                continue   # use each agent's own prediction, not aggregated
            gt = rec.get("ground_truth")
            if gt is None:
                continue
            initial = rec.get("initial_preds") or {}
            for agent_id, pred in initial.items():
                if not isinstance(pred, dict):
                    continue
                conf = pred.get("confidence")
                label = pred.get("label")
                if conf is None or label is None:
                    continue
                # binary correctness on this agent's modality
                label_int = 1 if label == "hateful" else (0 if label == "not_hateful" else None)
                if label_int is None:
                    # for non-binary datasets we record matched-class probability instead
                    correct = 1 if str(label) == str(rec.get("final_label")) else 0
                    agent_probs[agent_id].append((float(conf), correct))
                else:
                    correct = 1 if label_int == int(gt) else 0
                    # confidence is the model's stated confidence in its predicted label;
                    # convert to P(positive class)
                    if label_int == 1:
                        p_pos = float(conf)
                    else:
                        p_pos = 1.0 - float(conf)
                    agent_probs[agent_id].append((p_pos, int(gt)))

        for agent_id, lst in agent_probs.items():
            probs = [p for p, _ in lst]
            ys    = [y for _, y in lst]
            rows.append({
                "dataset": dataset, "regime": regime, "model_pair": pair,
                "agent": agent_id, "n": len(lst),
                "brier": round(_brier(probs, ys), 4),
                "ece":   round(_ece(probs, ys),   4),
            })

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "regime", "model_pair",
                                             "agent", "n", "brier", "ece"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    return out


# ===================================================================


def main() -> None:
    write_headline_table()
    write_abstention_table()
    write_heterogeneity_panel()
    write_intervention_informativeness()
    write_calibration_metrics()


if __name__ == "__main__":
    main()
