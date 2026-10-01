"""Dual-side VOUCH evaluation for math reasoning.

Two reasoning agents (Qwen2.5-7B-Instruct as proposer A, Llama-3.1-8B-Instruct
as acceptor B) both solve the same problem. The proposer A then commits to
a DUAL contract: an intervention I (one of the math-intervention menu),
plus two predicted post-intervention answers:
  - A1_a: A's own predicted answer on the intervened problem
  - A1_b: A's prediction of B's answer on the intervened problem

The protocol applies I deterministically to the problem, both agents
re-solve the intervened problem, and the contract is verified iff BOTH
commitments hold.

Mirrors `main_eval_codegen_dual.py`'s structure so the analysis pipeline
(scripts/derive_headline_stats.py) treats the two domains uniformly.

Output: JSONL with one record per example containing both initial answers,
the chosen intervention, the dual commitments, the actual post-intervention
answers, side-A / side-B verification, and the joint dual verification.
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


DUAL_CONTRACT_PROMPT = """You and a second reasoning agent (Agent B) both
solved this problem.

Original problem:
{problem}

YOUR answer (Agent A): A0_a = {a0_a}
Agent B's answer:      A0_b = {a0_b}

Now consider applying intervention "{intervention}" to the problem. The
intervention has produced this modified problem:

Modified problem:
{modified_problem}

Commit a DUAL contract. Predict TWO numbers:
  - A1_a: what YOU would compute on the modified problem
  - A1_b: what Agent B would compute on the modified problem

A1_b is a claim about a DIFFERENT agent's behaviour. You cannot satisfy
it by being self-consistent alone; you must model how Agent B is likely
to respond to this intervention.

Respond with ONLY one JSON object on a single line, exactly:
{{"intervention": "{intervention}", "a1_a_prediction": "<number>", "a1_b_prediction": "<number>"}}"""


JSON_RE = re.compile(r"\{[\s\S]*?\}")

MENU = INTERVENTION_NAMES

DATASET_LOADERS = {
    "gsm8k": gsm8k,
    "math":  math_bench,
    "svamp": svamp,
}


def parse_dual_contract(raw: str) -> dict | None:
    m = JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if "a1_a_prediction" not in obj or "a1_b_prediction" not in obj:
        return None
    obj["a1_a_prediction"] = str(obj["a1_a_prediction"])
    obj["a1_b_prediction"] = str(obj["a1_b_prediction"])
    return obj


def evaluate_example(agent_a, agent_b, ex, out_lines, adversarial: bool,
                      menu_idx: int) -> None:
    prompt_template = ADVERSARIAL_SOLVE_PROMPT if adversarial else SOLVE_PROMPT
    solve_prompt = prompt_template.format(problem=ex.text)

    # Step 1: both agents solve the original problem.
    raw_a0 = agent_a._generate_batch([solve_prompt], [None], seed=0, temperature=0.0)[0]
    raw_b0 = agent_b._generate_batch([solve_prompt], [None], seed=0, temperature=0.0)[0]
    a0_a = parse_final_answer(raw_a0)
    a0_b = parse_final_answer(raw_b0)

    # Step 2: pick intervention (round-robin) and apply it.
    intervention = MENU[menu_idx % len(MENU)]
    modified_text, tag = apply_intervention(ex.text, intervention)
    if tag is None:
        out_lines.append({
            "id": ex.id,
            "ground_truth_answer": ex.label,
            "a0_a": a0_a,
            "a0_b": a0_b,
            "initial_correct_a": answers_match(a0_a, str(ex.label)),
            "initial_correct_b": answers_match(a0_b, str(ex.label)),
            "intervention": intervention,
            "verified_dual": False,
            "reason": "intervention_not_applicable",
        })
        return

    # Step 3: proposer A commits to a dual contract.
    contract_prompt = DUAL_CONTRACT_PROMPT.format(
        problem=ex.text,
        a0_a=a0_a if a0_a else "?",
        a0_b=a0_b if a0_b else "?",
        intervention=intervention,
        modified_problem=modified_text,
    )
    raw_contract = agent_a._generate_batch([contract_prompt], [None], seed=0, temperature=0.0)[0]
    contract = parse_dual_contract(raw_contract)

    record = {
        "id": ex.id,
        "ground_truth_answer": ex.label,
        "a0_a": a0_a,
        "a0_b": a0_b,
        "initial_correct_a": answers_match(a0_a, str(ex.label)),
        "initial_correct_b": answers_match(a0_b, str(ex.label)),
        "intervention": intervention,
        "modified_problem": modified_text,
        "raw_solution_a": raw_a0[:2000],
        "raw_solution_b": raw_b0[:2000],
        "raw_contract": raw_contract[:800],
    }

    if contract is None:
        record.update({
            "contract_ok": False,
            "verified_dual": False,
            "reason": "contract_parse_failed",
        })
        out_lines.append(record)
        return

    a1_a_claim = contract["a1_a_prediction"]
    a1_b_claim = contract["a1_b_prediction"]

    # Step 4: both agents solve the modified problem.
    actual_prompt = prompt_template.format(problem=modified_text)
    raw_a1 = agent_a._generate_batch([actual_prompt], [None], seed=0, temperature=0.0)[0]
    raw_b1 = agent_b._generate_batch([actual_prompt], [None], seed=0, temperature=0.0)[0]
    a1_a_actual = parse_final_answer(raw_a1)
    a1_b_actual = parse_final_answer(raw_b1)

    # Step 5: verify both sides.
    side_a_verified = answers_match(a1_a_claim, a1_a_actual)
    side_b_verified = answers_match(a1_b_claim, a1_b_actual)
    verified_dual = side_a_verified and side_b_verified

    record.update({
        "contract_ok": True,
        "a1_a_claim": a1_a_claim,
        "a1_b_claim": a1_b_claim,
        "a1_a_actual": a1_a_actual,
        "a1_b_actual": a1_b_actual,
        "side_a_verified": side_a_verified,
        "side_b_verified": side_b_verified,
        "verified_dual": verified_dual,
        # Single-side baseline for direct comparison: would self-only have
        # accepted this contract?
        "verified_single_side": side_a_verified,
    })
    out_lines.append(record)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()),
                    default="gsm8k")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    suffix = "_adversarial" if args.adversarial else ""
    if args.out is None:
        args.out = str(ROOT / "results" / f"{args.dataset}_math_dual{suffix}_n{args.n}.jsonl")
    if args.summary is None:
        args.summary = str(ROOT / "results" / f"summary_{args.dataset}_math_dual{suffix}_n{args.n}.json")

    loader = DATASET_LOADERS[args.dataset]
    examples = loader.load(n=args.n)
    print(f"[{args.dataset}] loaded {len(examples)} examples")

    # Agent A: Qwen2.5-7B-Instruct (proposer)
    # Agent B: Llama-3.1-8B-Instruct (acceptor)
    agent_a = LlamaTextSubprocessAgent(agent_id=f"{args.dataset}_qwen_proposer",
                                         model_name="Qwen/Qwen2.5-7B-Instruct")
    agent_a.load_model()
    agent_b = LlamaTextSubprocessAgent(agent_id=f"{args.dataset}_llama_acceptor",
                                         model_name="meta-llama/Llama-3.1-8B-Instruct")
    agent_b.load_model()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    t0 = time.time()
    with out_path.open("w") as f:
        for i, ex in enumerate(examples, 1):
            tx = time.time()
            try:
                evaluate_example(agent_a, agent_b, ex, records,
                                  args.adversarial, i)
            except Exception as e:  # noqa: BLE001
                records.append({"id": ex.id, "error": f"{type(e).__name__}: {e}"})
            f.write(json.dumps(records[-1]) + "\n")
            f.flush()
            r = records[-1]
            print(f"[{i:3d}/{len(examples)}] {ex.id} "
                  f"a0_a={r.get('a0_a')} a0_b={r.get('a0_b')} gt={r.get('ground_truth_answer')} "
                  f"int_ok_a={r.get('initial_correct_a')} int_ok_b={r.get('initial_correct_b')} "
                  f"dual={r.get('verified_dual')} single={r.get('verified_single_side')} "
                  f"({time.time() - tx:.1f}s)")

    n_total = len(records)
    n_initial_a = sum(1 for r in records if r.get("initial_correct_a"))
    n_initial_b = sum(1 for r in records if r.get("initial_correct_b"))
    n_contracts = sum(1 for r in records if r.get("contract_ok"))
    n_verified_dual = sum(1 for r in records if r.get("verified_dual"))
    n_verified_single = sum(1 for r in records if r.get("verified_single_side"))

    # Contract-quality breakdown (proposer's initial correctness x verified).
    helped_d = caught_d = hurt_d = missed_d = 0
    helped_s = caught_s = hurt_s = missed_s = 0
    for r in records:
        if not r.get("contract_ok"):
            continue
        correct = bool(r.get("initial_correct_a"))
        if r.get("verified_dual"):
            if correct: helped_d += 1
            else:        hurt_d   += 1
        else:
            if correct: missed_d += 1
            else:        caught_d += 1
        if r.get("verified_single_side"):
            if correct: helped_s += 1
            else:        hurt_s   += 1
        else:
            if correct: missed_s += 1
            else:        caught_s += 1

    summary = {
        "n_examples": n_total,
        "n_initial_correct_a": n_initial_a,
        "initial_accuracy_a": n_initial_a / max(n_total, 1),
        "n_initial_correct_b": n_initial_b,
        "initial_accuracy_b": n_initial_b / max(n_total, 1),
        "n_contracts_parsed": n_contracts,
        "n_verified_dual": n_verified_dual,
        "n_verified_single": n_verified_single,
        "verification_rate_dual": n_verified_dual / max(n_contracts, 1),
        "verification_rate_single": n_verified_single / max(n_contracts, 1),
        "dual": {
            "helped": helped_d, "caught": caught_d,
            "hurt":   hurt_d,   "missed": missed_d,
            "hurt_caught": (hurt_d / caught_d) if caught_d else None,
        },
        "single_side": {
            "helped": helped_s, "caught": caught_s,
            "hurt":   hurt_s,   "missed": missed_s,
            "hurt_caught": (hurt_s / caught_s) if caught_s else None,
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
