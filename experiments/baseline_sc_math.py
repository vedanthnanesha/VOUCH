"""Self-Consistency (SC) baseline (Wang et al. 2022).

For each problem we sample N reasoning chains from the proposer (Qwen2.5-7B
by default) at temperature 0.7, parse the final answer from each, and pick
the majority. This serves as a within-agent "coordination without a
contract" baseline.

Output JSONL schema (one record per example):
  id, ground_truth_answer, problem,
  greedy_answer, greedy_correct,
  sampled_answers (list of N parsed answers),
  sampled_raw     (list of N truncated raw outputs),
  sc_answer       (majority-vote winner),
  sc_correct,
  sc_overturned_wrong  (True iff greedy wrong but sc correct),
  sc_overturned_right  (True iff greedy correct but sc wrong),
  sc_agreement         (max-vote count / N),
  wall_clock_s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ccontracts.agents._worker_pool import get_worker, shutdown_all
from ccontracts.contracts.math_interventions import (
    answers_match, parse_final_answer)
from ccontracts.datasets import gsm8k, math_bench, svamp


SOLVE_PROMPT = """Solve the following math problem. Show your reasoning
step by step. End your response with the final numeric answer on a
new line in the form: #### <answer>

Problem:
{problem}"""

ADVERSARIAL_SOLVE_PROMPT = """You are a fast solver under a deadline.
Skip detailed reasoning. Use the most obvious heuristic. Trust the
first computation that looks right.

End your response with the final numeric answer on a new line in the
form: #### <answer>

Problem:
{problem}"""


DATASET_LOADERS = {
    "gsm8k": gsm8k,
    "math":  math_bench,
    "svamp": svamp,
}


def run_sc_one(worker, prompt: str, n_samples: int,
               temperature: float, max_new_tokens: int = 384
               ) -> tuple[list[str], list[str]]:
    """Sample n_samples greedy-decoded continuations with different seeds
    via the persistent worker, returning (parsed_answers, raw_outputs)."""
    raws: list[str] = []
    parsed: list[str] = []
    for k in range(n_samples):
        out = worker.call(prompt, image_path=None,
                          sample=True, temperature=temperature,
                          seed=1 + k, max_new_tokens=max_new_tokens)
        raws.append(out)
        parsed.append(parse_final_answer(out) or "")
    return parsed, raws


def majority(answers: list[str]) -> tuple[str, int]:
    """Return (majority_answer, vote_count). Empty answers excluded from voting."""
    nonempty = [a for a in answers if a != ""]
    if not nonempty:
        return "", 0
    c = Counter(nonempty)
    ans, cnt = c.most_common(1)[0]
    return ans, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()),
                    default="gsm8k")
    ap.add_argument("--n", type=int, default=300,
                    help="number of examples")
    ap.add_argument("--n-samples", type=int, default=5,
                    help="number of SC samples per example")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--model",
                    default="Qwen/Qwen2.5-7B-Instruct",
                    help="proposer model id")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    suffix = "_adversarial" if args.adversarial else ""
    if args.out is None:
        args.out = str(ROOT / "results" / "baselines"
                        / f"sc_{args.dataset}{suffix}_n{args.n}_k{args.n_samples}.jsonl")
    if args.summary is None:
        args.summary = str(ROOT / "results" / "baselines"
                            / f"summary_sc_{args.dataset}{suffix}_n{args.n}_k{args.n_samples}.json")

    examples = DATASET_LOADERS[args.dataset].load(n=args.n)
    print(f"[sc-{args.dataset}{suffix}] loaded {len(examples)} examples; "
          f"k={args.n_samples} temp={args.temperature} model={args.model}")

    worker = get_worker("text", args.model, dtype="fp32")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_tpl = ADVERSARIAL_SOLVE_PROMPT if args.adversarial else SOLVE_PROMPT
    records: list[dict] = []
    t0 = time.time()
    with out_path.open("w") as fh:
        for i, ex in enumerate(examples, 1):
            tx = time.time()
            prompt = prompt_tpl.format(problem=ex.text)
            # greedy proposer (matches single-side VOUCH baseline)
            greedy_raw = worker.call(prompt, image_path=None,
                                       sample=False, max_new_tokens=384)
            greedy_ans = parse_final_answer(greedy_raw) or ""
            greedy_correct = answers_match(greedy_ans, str(ex.label))
            # SC sampling
            sampled, sampled_raws = run_sc_one(
                worker, prompt,
                n_samples=args.n_samples,
                temperature=args.temperature)
            sc_ans, sc_votes = majority(sampled)
            sc_correct = answers_match(sc_ans, str(ex.label))
            rec = {
                "id": ex.id,
                "ground_truth_answer": ex.label,
                "problem": ex.text[:1000],
                "greedy_answer": greedy_ans,
                "greedy_correct": bool(greedy_correct),
                "sampled_answers": sampled,
                "sampled_raw": [s[:600] for s in sampled_raws],
                "sc_answer": sc_ans,
                "sc_votes": sc_votes,
                "sc_correct": bool(sc_correct),
                "sc_overturned_wrong": bool((not greedy_correct) and sc_correct),
                "sc_overturned_right": bool(greedy_correct and (not sc_correct)),
                "sc_agreement": sc_votes / max(args.n_samples, 1),
                "wall_clock_s": round(time.time() - tx, 2),
            }
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i:3d}/{len(examples)}] {ex.id} "
                  f"greedy={greedy_ans} sc={sc_ans} ({sc_votes}/{args.n_samples}) "
                  f"gt={ex.label} g={greedy_correct} sc={sc_correct} "
                  f"({rec['wall_clock_s']}s)")

    n_total = len(records)
    n_greedy_correct = sum(1 for r in records if r["greedy_correct"])
    n_sc_correct     = sum(1 for r in records if r["sc_correct"])
    n_overturn_wrong = sum(1 for r in records if r["sc_overturned_wrong"])
    n_overturn_right = sum(1 for r in records if r["sc_overturned_right"])
    summary = {
        "dataset": args.dataset, "regime": "adversarial" if args.adversarial else "clean",
        "model": args.model,
        "n_samples": args.n_samples, "temperature": args.temperature,
        "n_examples": n_total,
        "greedy_accuracy": n_greedy_correct / max(n_total, 1),
        "sc_accuracy":     n_sc_correct     / max(n_total, 1),
        "n_greedy_correct": n_greedy_correct,
        "n_sc_correct":     n_sc_correct,
        "n_sc_overturned_wrong_to_right": n_overturn_wrong,
        "n_sc_overturned_right_to_wrong": n_overturn_right,
        "mean_sc_agreement": (sum(r["sc_agreement"] for r in records) /
                                max(n_total, 1)),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    shutdown_all()


if __name__ == "__main__":
    main()
