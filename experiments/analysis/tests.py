"""Paired statistical tests used by the analysis scripts.

  - mcnemar_exact: McNemar's exact (binomial) test for two paired binary
                    decisions (e.g. VOUCH vs Ensemble correct/incorrect).
  - permutation_test: paired permutation test for accuracy differences.
  - paired_decisions: load two JSONLs and produce aligned correct-vectors
                       keyed on example_id.

All functions are stdlib only (no scipy dependency).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


# -------- aligning two runs by example_id --------

def paired_decisions(jsonl_a: Path, jsonl_b: Path,
                     condition_a: str, condition_b: str
                     ) -> tuple[list[bool], list[bool]]:
    """Align two JSONLs on example_id; return paired correct-vectors.

    Only keeps examples present (under the requested condition) in BOTH
    JSONLs.
    """
    def _load(p: Path, cond: str) -> dict[str, bool]:
        out: dict[str, bool] = {}
        with Path(p).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("condition") != cond:
                    continue
                eid = r.get("example_id")
                if eid is None:
                    continue
                out[str(eid)] = bool(r.get("correct"))
        return out

    a = _load(jsonl_a, condition_a)
    b = _load(jsonl_b, condition_b)
    keys = sorted(set(a) & set(b))
    return [a[k] for k in keys], [b[k] for k in keys]


# -------- McNemar's exact test --------

def mcnemar_exact(correct_a: list[bool],
                  correct_b: list[bool]) -> dict[str, float | int]:
    """Two-sided McNemar exact (binomial) test.

    Counts discordant pairs:
      b = #(A correct, B incorrect)
      c = #(A incorrect, B correct)
    Under H0 the two methods agree on average, b ~ Binomial(b+c, 0.5).
    p-value is twice the lower-tail probability for the more extreme count.
    """
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, x in zip(correct_a, correct_b) if a and not x)
    c = sum(1 for a, x in zip(correct_a, correct_b) if not a and x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0,
                "acc_a": _mean(correct_a), "acc_b": _mean(correct_b)}

    k = min(b, c)
    # two-sided p = 2 * P(X <= k) under Binomial(n, 0.5), clamped to 1.0
    p = 2.0 * sum(_binom_pmf(i, n, 0.5) for i in range(k + 1))
    p = min(1.0, p)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p,
            "acc_a": _mean(correct_a), "acc_b": _mean(correct_b)}


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def _mean(xs: list[bool]) -> float:
    return sum(1 for x in xs if x) / len(xs) if xs else float("nan")


# -------- paired permutation test --------

def permutation_test(correct_a: list[bool], correct_b: list[bool],
                     n_perm: int = 10000, seed: int = 0) -> dict:
    """Two-sided paired permutation test on the difference in accuracy.

    Under H0, on each example the labels (correct_a, correct_b) are
    exchangeable; we randomly swap each pair and recompute the diff.
    """
    assert len(correct_a) == len(correct_b)
    n = len(correct_a)
    if n == 0:
        return {"observed_diff": 0.0, "p_value": 1.0, "n_perm": 0}

    pairs = list(zip(correct_a, correct_b))
    observed = (_mean(correct_a) - _mean(correct_b))
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        swapped = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]
        a_perm = [p[0] for p in swapped]
        b_perm = [p[1] for p in swapped]
        d = _mean(a_perm) - _mean(b_perm)
        if abs(d) >= abs(observed):
            count += 1
    p = count / n_perm
    return {"observed_diff": observed, "p_value": p, "n_perm": n_perm}
