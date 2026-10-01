"""Math-reasoning evaluation harness for VOUCH.

For each problem the agent generates a CoT solution and a numeric final
answer. The agent then commits to a counterfactual contract: pick one
intervention from the menu, predict the new answer after the
intervention is applied. The protocol applies the intervention
deterministically and re-runs the agent on the intervened problem; the
contract is verified iff the new answer matches the commitment.

Supports GSM8K, MATH, and SVAMP. The intervention menu is the same
across datasets; the verification operator (numeric answer match) is
the same.

Output: a JSONL with one record per example containing the candidate
chain-of-thought, the parsed initial answer, the chosen intervention,
the committed counterfactual answer, the actual post-intervention
answer, and a `verified` boolean.
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

from ccontracts.agents.llama_text_subprocess import LlamaTextSubprocessAgent
from ccontracts.contracts.math_interventions import (
    INTERVENTION_NAMES,
    apply_intervention,
    answers_match,
    parse_final_answer,
)
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


CONTRACT_PROMPT = """You just solved this math problem and got the
answer A_0 = {a0}. Now consider a small modification to the problem.

Modified problem:
{modified_problem}

Predict the new numeric answer A_1 that you would compute for the
modified problem. Respond with ONLY one JSON object on a single line,
exactly:
{{"intervention": "{intervention}", "a1_prediction": "<number>"}}"""


JSON_RE = re.compile(r"\{[\s\S]*?\}")

MENU = INTERVENTION_NAMES

DATASET_LOADERS = {
    "gsm8k": gsm8k,
    "math":  math_bench,
    "svamp": svamp,
}


def parse_contract(raw: str) -> dict | None:
    m = JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if "a1_prediction" not in obj:
        return None
    obj["a1_prediction"] = str(obj["a1_prediction"])
    return obj


def evaluate_example(agent, ex, out_lines, adversarial: bool, menu_idx: int) -> None:
    prompt_template = ADVERSARIAL_SOLVE_PROMPT if adversarial else SOLVE_PROMPT
    solve_prompt = prompt_template.format(problem=ex.text)
    raw_solution = agent._generate_batch([solve_prompt], [None], seed=0, temperature=0.0)[0]
    a0 = parse_final_answer(raw_solution)

    # Pick the next intervention in the menu (round-robin, deterministic).
    intervention = MENU[menu_idx % len(MENU)]
    modified_text, tag = apply_intervention(ex.text, intervention)
    if tag is None:
        out_lines.append({
            "id": ex.id,
            "a0_initial": a0,
            "intervention": intervention,
            "verified": False,
            "reason": "intervention_not_applicable",
        })
        return

    contract_prompt = CONTRACT_PROMPT.format(
        a0=a0 if a0 else "?",
        modified_problem=modified_text,
        intervention=intervention,
    )
    raw_contract = agent._generate_batch([contract_prompt], [None], seed=0, temperature=0.0)[0]
    contract = parse_contract(raw_contract)

    record = {
        "id": ex.id,
        "ground_truth_answer": ex.label,
        "a0_initial": a0,
        "initial_correct": answers_match(a0, str(ex.label)),
        "intervention": intervention,
        "modified_problem": modified_text,
        "raw_solution": raw_solution[:2000],
        "raw_contract": raw_contract[:600],
    }

    if contract is None:
        record.update({"contract_ok": False, "verified": False,
                        "reason": "contract_parse_failed"})
        out_lines.append(record)
        return

    a1_claim = contract["a1_prediction"]
    # Run the same agent on the modified problem to get the actual A_1.
    actual_prompt = prompt_template.format(problem=modified_text)
    raw_actual = agent._generate_batch([actual_prompt], [None], seed=0, temperature=0.0)[0]
    a1_actual = parse_final_answer(raw_actual)
    record.update({
        "contract_ok": True,
        "a1_claim": a1_claim,
        "a1_actual": a1_actual,
        "verified": answers_match(a1_claim, a1_actual),
    })
    out_lines.append(record)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()),
                    default="gsm8k")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    suffix = "_adversarial" if args.adversarial else ""
    if args.out is None:
        args.out = str(ROOT / "results" / f"{args.dataset}_math{suffix}.jsonl")
    if args.summary is None:
        args.summary = str(ROOT / "results" / f"summary_{args.dataset}_math{suffix}.json")

    loader = DATASET_LOADERS[args.dataset]
    examples = loader.load(n=args.n)
    print(f"[{args.dataset}] loaded {len(examples)} examples")

    agent = LlamaTextSubprocessAgent(agent_id=f"{args.dataset}_solver",
                                      model_name="Qwen/Qwen2.5-7B-Instruct")
    agent.load_model()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    t0 = time.time()
    with out_path.open("w") as f:
        for i, ex in enumerate(examples, 1):
            tx = time.time()
            try:
                evaluate_example(agent, ex, records, args.adversarial, i)
            except Exception as e:  # noqa: BLE001
                records.append({"id": ex.id, "error": f"{type(e).__name__}: {e}"})
            f.write(json.dumps(records[-1]) + "\n")
            f.flush()
            r = records[-1]
            print(f"[{i:3d}/{len(examples)}] {ex.id} "
                  f"a0={r.get('a0_initial')} gt={r.get('ground_truth_answer')} "
                  f"init_ok={r.get('initial_correct')} "
                  f"int={r.get('intervention')} "
                  f"verified={r.get('verified')} ({time.time() - tx:.1f}s)")

    n_total = len(records)
    n_initial_correct = sum(1 for r in records if r.get("initial_correct"))
    n_contracts = sum(1 for r in records if r.get("contract_ok"))
    n_verified = sum(1 for r in records if r.get("verified"))
    summary = {
        "n_examples": n_total,
        "n_initial_correct": n_initial_correct,
        "initial_accuracy": n_initial_correct / max(n_total, 1),
        "n_contracts_parsed": n_contracts,
        "n_contracts_verified": n_verified,
        "verification_rate": n_verified / max(n_contracts, 1),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
