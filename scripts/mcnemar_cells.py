"""Per-cell paired McNemar tests, VOUCH vs Ensemble, on the multimodal cells.

For each cell, b counts examples on which only VOUCH (``contracts``) is
correct and c counts examples on which only Ensemble is correct; p is the
exact two-sided binomial p-value on the discordant pairs. The pooled test
uses the paired decisions of all cells together.

Writes results/derived/per_cell_mcnemar.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.analysis import tests  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "derived" / "per_cell_mcnemar.json"

# (display name, results JSONL)
CELLS = [
    ("Hateful Memes clean (same)",  "main_eval_hateful_memes_clean.jsonl"),
    ("CrisisMMD clean (same)",      "main_eval_crisismmd_clean.jsonl"),
    ("CrisisMMD adv (same)",        "main_eval_crisismmd_adversarial.jsonl"),
    ("MM-IMDb clean (same)",        "main_eval_mm_imdb_clean.jsonl"),
    ("MM-IMDb adv (same)",          "main_eval_mm_imdb_adversarial.jsonl"),
    ("Hateful Memes adv (cross)",   "cross_family_eval_hateful_memes_adversarial.jsonl"),
    ("Hateful Memes clean (cross)", "cross_family_eval_hateful_memes_clean.jsonl"),
]


def main() -> None:
    per_cell = []
    pooled_vouch: list[bool] = []
    pooled_ensemble: list[bool] = []
    for name, fname in CELLS:
        path = RESULTS / fname
        if not path.exists():
            print(f"  skip {name}: missing {fname}")
            continue
        vouch, ensemble = tests.paired_decisions(path, path,
                                                 condition_a="contracts",
                                                 condition_b="ensemble")
        res = tests.mcnemar_exact(vouch, ensemble)
        per_cell.append({"name": name, "b": res["b"], "c": res["c"],
                         "same": len(vouch) - res["b"] - res["c"],
                         "p": res["p_value"]})
        pooled_vouch += vouch
        pooled_ensemble += ensemble

    pooled = tests.mcnemar_exact(pooled_vouch, pooled_ensemble)
    out = {"per_cell": per_cell,
           "pooled": {"b": pooled["b"], "c": pooled["c"], "p": pooled["p_value"]}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for row in per_cell:
        print(f"  {row['name']:<30} b={row['b']:>3} c={row['c']:>3} p={row['p']:.3f}")
    print(f"  {'pooled':<30} b={pooled['b']:>3} c={pooled['c']:>3} p={pooled['p_value']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
