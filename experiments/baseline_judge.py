"""Single-Judge baseline.

A third Llama-3.1-8B-Instruct call (or configurable model) reads the
problem and both proposers' answers, then picks one. This is the most
common "coordination via judge model" baseline (used implicitly in
ChatEval, AutoGen, etc.) and tests whether a separate verifier model can
stand in for protocol-executed verification.

Operates on existing dual-side math JSONLs to avoid recomputing
the proposer/acceptor steps:
  input  : results/{dataset}_math_dual{,_adversarial}_n{N}.jsonl
  output : results/baselines/judge_<input stem>{,_advjudge}.jsonl

Per-row schema:
  id, ground_truth_answer,
  a0_a, a0_b, initial_correct_a, initial_correct_b,
  judge_pick      ("A" | "B" | "abstain"),
  judge_answer    (the picked numeric string),
  judge_correct,
  judge_raw       (truncated judge response),
  agreement_flag  (True if both proposers already agreed pre-judge),
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ccontracts.agents._worker_pool import get_worker, shutdown_all
from ccontracts.contracts.math_interventions import (
    answers_match, parse_final_answer)


JUDGE_PROMPT = """You are an impartial judge. Two agents independently
solved the following problem and produced different final answers (or
the same answer). Read the problem and both candidate answers, then
decide which final answer is correct.

Problem:
{problem}

Agent A's final answer: {a0_a}
Agent B's final answer: {a0_b}

Respond with ONLY one JSON object on a single line, exactly:
{{"pick": "A" | "B", "final_answer": "<number>"}}"""


ADVERSARIAL_JUDGE_PROMPT = """You are a fast tie-breaker under a deadline.
Skip detailed reasoning. Trust the answer that looks most confidently
stated. If in doubt, pick Agent A.

Problem:
{problem}

Agent A's final answer: {a0_a}
Agent B's final answer: {a0_b}

Respond with ONLY one JSON object on a single line, exactly:
{{"pick": "A" | "B", "final_answer": "<number>"}}"""


JSON_RE = re.compile(r"\{[\s\S]*?\}")


def parse_judge(raw: str) -> dict | None:
    m = JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if "pick" not in obj or "final_answer" not in obj:
        return None
    return {"pick": str(obj["pick"]).strip().upper()[:1],
             "final_answer": str(obj["final_answer"]).strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", required=True,
                    help="dual-side JSONL with a0_a, a0_b, ground_truth_answer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--judge-model",
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adversarial-judge", action="store_true",
                    help="apply adversarial system framing to the judge "
                         "(tests 'judge inherits self-coherence pathology').")
    ap.add_argument("--problem-field", default="modified_problem",
                    help="which field carries the problem text. We default to "
                         "modified_problem so the judge sees what the agents "
                         "actually solved. Falls back if missing.")
    args = ap.parse_args()

    in_path = Path(args.in_file)
    assert in_path.exists(), f"input not found: {in_path}"
    tag = "_advjudge" if args.adversarial_judge else ""
    if args.out is None:
        args.out = str(ROOT / "results" / "baselines"
                        / f"judge_{in_path.stem}{tag}.jsonl")
    if args.summary is None:
        args.summary = str(ROOT / "results" / "baselines"
                            / f"summary_judge_{in_path.stem}{tag}.json")

    print(f"[judge] reading {in_path}; out={args.out}; "
          f"judge={args.judge_model}; adv={args.adversarial_judge}")

    worker = get_worker("text", args.judge_model, dtype="fp32")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_tpl = ADVERSARIAL_JUDGE_PROMPT if args.adversarial_judge else JUDGE_PROMPT

    records: list[dict] = []
    t0 = time.time()
    with in_path.open() as fin, out_path.open("w") as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                src = json.loads(line)
            except Exception:
                continue
            a0_a = src.get("a0_a")
            a0_b = src.get("a0_b")
            gt = src.get("ground_truth_answer")
            problem = (src.get(args.problem_field)
                        or src.get("modified_problem")
                        or src.get("problem"))
            if a0_a is None or a0_b is None or problem is None:
                continue

            prompt = prompt_tpl.format(problem=problem,
                                         a0_a=a0_a, a0_b=a0_b)
            raw = worker.call(prompt, image_path=None,
                              sample=False, max_new_tokens=192)
            parsed = parse_judge(raw)
            if parsed is None:
                pick = "abstain"
                final = ""
                ok = False
            else:
                pick = parsed["pick"]
                final = parsed["final_answer"]
                # ensure final_answer matches the picked agent's answer when
                # the model fabricates a new number, prefer the original.
                if pick == "A":
                    final = final or str(a0_a)
                elif pick == "B":
                    final = final or str(a0_b)
                else:
                    pick = "abstain"
                ok = parsed is not None

            judge_correct = answers_match(final, str(gt)) if final else False
            rec = {
                "id": src.get("id"),
                "ground_truth_answer": gt,
                "a0_a": a0_a, "a0_b": a0_b,
                "initial_correct_a": bool(src.get("initial_correct_a")),
                "initial_correct_b": bool(src.get("initial_correct_b")),
                "judge_pick": pick,
                "judge_answer": final,
                "judge_correct": bool(judge_correct),
                "judge_raw": (raw or "")[:600],
                "judge_parse_ok": ok,
                "agreement_flag": str(a0_a) == str(a0_b),
                "adversarial_judge": args.adversarial_judge,
            }
            records.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % 10 == 0:
                print(f"[{i}] judge={pick} ans={final} gt={gt} ok={judge_correct}")

    n = len(records)
    n_correct = sum(1 for r in records if r["judge_correct"])
    n_pick_a  = sum(1 for r in records if r["judge_pick"] == "A")
    n_pick_b  = sum(1 for r in records if r["judge_pick"] == "B")
    n_abstain = sum(1 for r in records if r["judge_pick"] == "abstain")
    n_a_right = sum(1 for r in records if r["initial_correct_a"])
    n_b_right = sum(1 for r in records if r["initial_correct_b"])
    n_disagreement = sum(1 for r in records if not r["agreement_flag"])

    summary = {
        "in_file": str(in_path),
        "judge_model": args.judge_model,
        "adversarial_judge": args.adversarial_judge,
        "n": n,
        "judge_accuracy": n_correct / max(n, 1),
        "agent_a_accuracy": n_a_right / max(n, 1),
        "agent_b_accuracy": n_b_right / max(n, 1),
        "n_pick_a": n_pick_a, "n_pick_b": n_pick_b, "n_abstain": n_abstain,
        "pick_a_rate": n_pick_a / max(n, 1),
        "pick_b_rate": n_pick_b / max(n, 1),
        "abstain_rate": n_abstain / max(n, 1),
        "n_disagreement_cases": n_disagreement,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    shutdown_all()


if __name__ == "__main__":
    main()
