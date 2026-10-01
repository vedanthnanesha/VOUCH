"""Wilson and bootstrap CIs for every headline cell.

Reads results/derived/abstention_disentangled.csv and results/derived/
headline_table.csv (produced by derive_headline_stats.py) and writes:

  results/derived/ci_wilson.csv      -- Wilson 95% CIs on every rate
  results/derived/ci_bootstrap.csv   -- paired bootstrap 95% CIs on the
                                          four headline reductions
                                          (single -> dual hurt-rate)

Wilson uses statsmodels.proportion.proportion_confint if available, with
a manual fallback (the Wilson interval formula).
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "results" / "derived"


# -- Wilson 95% --------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) Wilson 95% interval. (0,0) when n==0."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half   = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def fmt(x: float) -> str:
    return f"{x:.4f}"


def write_wilson() -> Path:
    in_path = DERIVED / "abstention_disentangled.csv"
    out_path = DERIVED / "ci_wilson.csv"
    rows = list(csv.DictReader(in_path.open()))

    # (k_num_field, n_den_field, label)
    targets = [
        ("n_single_hurt",   "n_emitted", "single_hurt_per_emitted"),
        ("n_dual_hurt",     "n_emitted", "dual_hurt_per_emitted"),
        ("n_single_caught", "n_emitted", "single_catch_per_emitted"),
        ("n_dual_caught",   "n_emitted", "dual_catch_per_emitted"),
        # catch-per-acted-on-wrong (kills "just abstaining" claim):
        # numerator = caught; denominator = caught + hurt
        ("__single_acted_caught__", "__single_acted__", "single_catch_per_acted"),
        ("__dual_acted_caught__",   "__dual_acted__",   "dual_catch_per_acted"),
    ]

    out_rows: list[dict] = []
    for r in rows:
        s_caught = int(r["n_single_caught"]); s_hurt = int(r["n_single_hurt"])
        d_caught = int(r["n_dual_caught"]);   d_hurt = int(r["n_dual_hurt"])
        s_acted = s_caught + s_hurt
        d_acted = d_caught + d_hurt
        synth = {
            "__single_acted_caught__": s_caught,
            "__single_acted__":        s_acted,
            "__dual_acted_caught__":   d_caught,
            "__dual_acted__":          d_acted,
        }
        for k_field, n_field, label in targets:
            k = int(synth.get(k_field, r.get(k_field) or 0))
            n = int(synth.get(n_field, r.get(n_field) or 0))
            p, lo, hi = wilson(k, n)
            out_rows.append({
                "domain": r["domain"], "dataset": r["dataset"], "regime": r["regime"],
                "model_pair": r["model_pair"],
                "metric": label,
                "k": k, "n": n,
                "point": fmt(p), "lo": fmt(lo), "hi": fmt(hi),
            })

    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["domain", "dataset", "regime",
            "model_pair", "metric", "k", "n", "point", "lo", "hi"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path}  ({len(out_rows)} rows)")
    return out_path


# -- Paired bootstrap on hurt-rate reduction --------------------------------


def bootstrap_paired(in_path: Path, paired_keys: list[tuple[str, str, str]],
                      B: int = 2000, seed: int = 0
                      ) -> list[dict]:
    """For each paired key (single-side-file, dual-side-file, label), compute
    the bootstrap 95% CI on (single_hurt_rate - dual_hurt_rate) and on the
    relative reduction 1 - dual/single. ``in_path`` is the original
    headline_table.csv just so we can look up the source filenames.

    For each row we ALSO need access to the per-record raw flags, so this
    re-opens the dual-side JSONLs directly.
    """
    rng = random.Random(seed)

    def load(jsonl: Path):
        rows = []
        for line in jsonl.open():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    out = []
    for (path, label, _unused) in paired_keys:
        rows = load(Path(path))
        # per-record (initially_wrong, single_verified, dual_verified)
        events = []
        for r in rows:
            if not r.get("contract_ok"):
                continue
            if "initial_correct_a" in r:
                ic_a = bool(r["initial_correct_a"])
            else:
                v0_a = str(r.get("v0_a", "")).upper()
                ic_a = bool(v0_a) and "F" not in v0_a
            v_s = bool(r.get("verified_single_side"))
            v_d = bool(r.get("verified_dual"))
            events.append((ic_a, v_s, v_d))
        if not events:
            continue
        # observed rates
        s_caught = sum(1 for (c, vs, vd) in events if (not c) and (not vs))
        s_hurt   = sum(1 for (c, vs, vd) in events if (not c) and vs)
        d_caught = sum(1 for (c, vs, vd) in events if (not c) and (not vd))
        d_hurt   = sum(1 for (c, vs, vd) in events if (not c) and vd)
        s_acted  = s_caught + s_hurt
        d_acted  = d_caught + d_hurt
        s_hurt_per_acted = s_hurt / s_acted if s_acted else 0.0
        d_hurt_per_acted = d_hurt / d_acted if d_acted else 0.0
        obs_diff = s_hurt_per_acted - d_hurt_per_acted
        obs_rel  = (1 - d_hurt_per_acted / s_hurt_per_acted) if s_hurt_per_acted else 0.0

        diffs = []
        rels  = []
        n = len(events)
        for _ in range(B):
            sample = [events[rng.randrange(n)] for _ in range(n)]
            ss_c = sum(1 for (c, vs, vd) in sample if (not c) and (not vs))
            ss_h = sum(1 for (c, vs, vd) in sample if (not c) and vs)
            dd_c = sum(1 for (c, vs, vd) in sample if (not c) and (not vd))
            dd_h = sum(1 for (c, vs, vd) in sample if (not c) and vd)
            sa = ss_c + ss_h
            da = dd_c + dd_h
            s_hp = ss_h / sa if sa else 0.0
            d_hp = dd_h / da if da else 0.0
            diffs.append(s_hp - d_hp)
            rels.append((1 - d_hp / s_hp) if s_hp else 0.0)
        diffs.sort(); rels.sort()
        out.append({
            "cell": label, "source": Path(path).name,
            "n_acted_wrong_single": s_acted,
            "n_acted_wrong_dual":   d_acted,
            "single_hurt_per_acted":  fmt(s_hurt_per_acted),
            "dual_hurt_per_acted":    fmt(d_hurt_per_acted),
            "abs_reduction_point":    fmt(obs_diff),
            "abs_reduction_lo":       fmt(diffs[int(0.025 * B)]),
            "abs_reduction_hi":       fmt(diffs[int(0.975 * B)]),
            "rel_reduction_point":    fmt(obs_rel),
            "rel_reduction_lo":       fmt(rels[int(0.025 * B)]),
            "rel_reduction_hi":       fmt(rels[int(0.975 * B)]),
        })
    return out


def write_bootstrap() -> Path:
    paired = [
        (ROOT / "results" / "gsm8k_math_dual_n300.jsonl",
         "GSM8K-alg clean", "math"),
        (ROOT / "results" / "gsm8k_math_dual_adversarial_n300.jsonl",
         "GSM8K-alg adv",   "math"),
        (ROOT / "results" / "svamp_math_dual_n300.jsonl",
         "SVAMP clean",     "math"),
        (ROOT / "results" / "svamp_math_dual_adversarial_n300.jsonl",
         "SVAMP adv",       "math"),
        (ROOT / "results" / "math_math_dual_n300.jsonl",
         "MATH-alg clean",  "math"),
        (ROOT / "results" / "math_math_dual_adversarial_n300.jsonl",
         "MATH-alg adv",    "math"),
        (ROOT / "results" / "mbpp_codegen_dual_n200.jsonl",
         "MBPP clean",      "code"),
        (ROOT / "results" / "mbpp_codegen_dual_adversarial_n200.jsonl",
         "MBPP adv",        "code"),
        (ROOT / "results" / "humaneval_codegen_dual_n164.jsonl",
         "HumanEval clean", "code"),
        (ROOT / "results" / "humaneval_codegen_dual_adversarial_n164.jsonl",
         "HumanEval adv",   "code"),
    ]
    paired = [t for t in paired if t[0].exists()]
    out = bootstrap_paired(DERIVED / "headline_table.csv", paired, B=2000)
    out_path = DERIVED / "ci_bootstrap.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()) if out else
            ["cell", "source", "n_acted_wrong_single", "n_acted_wrong_dual",
             "single_hurt_per_acted", "dual_hurt_per_acted",
             "abs_reduction_point", "abs_reduction_lo", "abs_reduction_hi",
             "rel_reduction_point", "rel_reduction_lo", "rel_reduction_hi"])
        w.writeheader()
        w.writerows(out)
    print(f"wrote {out_path}  ({len(out)} rows)")
    return out_path


def main() -> None:
    write_wilson()
    write_bootstrap()


if __name__ == "__main__":
    main()
