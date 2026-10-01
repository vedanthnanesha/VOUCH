"""Compute the intervention-informativeness ranking I(ι) per dataset.

Reads completed classification JSONLs, calls
``experiments.analysis.metrics.informativeness_per_intervention``, and emits
both a JSON summary AND a Python snippet that can be pasted into
``ccontracts/task_config.py`` to set ``intervention_ranking`` for each
dataset's TaskConfig.

This is the offline ranking step used by VOUCH-Adaptive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.analysis import metrics


def rank_dataset(jsonls: list[Path]) -> list[tuple[str, float]]:
    recs: list[dict] = []
    for p in jsonls:
        recs += metrics.load_jsonl(p, condition="contracts")
    info = metrics.informativeness_per_intervention(recs)
    ranked = sorted(info.items(),
                    key=lambda kv: kv[1].get("I", 0.0),
                    reverse=True)
    return [(name, float(d.get("I", 0.0))) for name, d in ranked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["hateful_memes", "crisismmd", "mm_imdb"])
    ap.add_argument("--regimes", nargs="+",
                    default=["clean", "adversarial"])
    ap.add_argument("--out", default=str(ROOT / "results" /
                                          "intervention_rankings.json"))
    args = ap.parse_args()

    rankings: dict[str, list[tuple[str, float]]] = {}
    for ds in args.datasets:
        jsonls = []
        for reg in args.regimes:
            p = ROOT / "results" / f"main_eval_{ds}_{reg}.jsonl"
            if p.exists():
                jsonls.append(p)
        if not jsonls:
            print(f"[{ds}] no input JSONLs found; skipping")
            continue
        rankings[ds] = rank_dataset(jsonls)
        print(f"[{ds}]")
        for name, score in rankings[ds]:
            print(f"  {name:<25s} I={score:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rankings, indent=2))
    print(f"\n[rankings] saved -> {out_path}")

    # Print Python snippet for task_config
    print("\n# --- paste into task_config.py per dataset ---")
    for ds, ranked in rankings.items():
        ranked_names = [n for n, _ in ranked]
        print(f"# {ds}: intervention_ranking={ranked_names}")


if __name__ == "__main__":
    main()
