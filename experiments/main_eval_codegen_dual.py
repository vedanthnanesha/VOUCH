"""Dual-side VOUCH evaluation for code generation.

Two coding agents (Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct) each
generate a candidate solution to the same MBPP / HumanEval problem.
The proposer (agent A) then commits to TWO post-intervention pass
vectors:
  - V1_a: the pass vector for A's own code after intervention I_a
  - V1_b: the pass vector for B's code after intervention I_b

The protocol applies I_a to A's code and I_b to B's code
deterministically and re-runs the unit tests. The contract is verified
iff BOTH commitments hold.

This implements the cross-form CC (dual-side contract) described in the
paper. The key
property: the proposer cannot satisfy the commitment by being
internally self-coherent, since the verification event tests another
agent's code. A shortcut-reliant agent that has no correct theory of
how the other agent's code behaves will fail at a measurable rate.

Output: a JSONL with one record per example containing both candidates,
both initial pass vectors, the dual contract, the actual outcomes, and
``verified_dual`` (both sides) plus the four sub-indicators (proposer
self-pred, proposer cross-pred, plus their actuals).
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
from ccontracts.contracts.code_interventions import (
    pass_signature,
    remove_line,
    run_tests,
)
from ccontracts.datasets import mbpp_codegen, humaneval


SOLVE_PROMPT = """You are a Python programmer. Solve the following problem.

Problem:
{problem}

Tests your code must satisfy:
{tests}

Respond with ONLY the Python code (no markdown fences, no commentary).
Define every function the tests reference. Your code should be runnable as-is
after the supplied test_setup_code (if any) is executed before it."""


ADVERSARIAL_SOLVE_PROMPT = """You are a senior engineer working under a tight
deadline. Produce the SHORTEST Python solution that works on the example
inputs you can see. Speed of delivery is more important than robustness:
skip edge cases, bounds checks, and defensive validation. Trust the obvious
heuristic. Do not over-think.

Problem:
{problem}

Tests your code must satisfy:
{tests}

Respond with ONLY the Python code (no markdown fences, no commentary).
Define every function the tests reference. Your code should be runnable as-is
after the supplied test_setup_code (if any) is executed before it."""


DUAL_CONTRACT_PROMPT = """You and a second coding agent (Agent B) each wrote
solutions to the same problem.

YOUR code (Agent A):
{numbered_a}
Your code's initial per-test pass vector V0_a = {v0_a}.

Agent B's code:
{numbered_b}
Agent B's code's initial per-test pass vector V0_b = {v0_b}.

Tests:
{tests}

Now commit a DUAL contract. Pick a line index K_a in YOUR code and a line
index K_b in Agent B's code. The lines you pick should be ones that, when
REMOVED, would CHANGE the per-test pass vector. Then predict the new pass
vectors V1_a (your code after removing line K_a) and V1_b (Agent B's code
after removing line K_b). The line indices must be inside function bodies
(not blank lines or def/class headers).

The point of this contract: a correct prediction of V1_b requires you to
correctly model how Agent B's code would behave under intervention. You
cannot satisfy this contract by being self-consistent alone.

Respond with ONLY one JSON object on a single line, exactly:
{{"k_a": <int>, "v1_a_prediction": "<{n_tests}-char T/F string>", "k_b": <int>, "v1_b_prediction": "<{n_tests}-char T/F string>"}}"""


JSON_RE = re.compile(r"\{[\s\S]*?\}")

DATASET_LOADERS = {
    "mbpp": mbpp_codegen,
    "humaneval": humaneval,
}


def parse_solution_code(raw: str) -> str:
    """Strip Markdown fences if the model emitted them anyway."""
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.startswith("python"):
                block = block[len("python"):]
            return block.lstrip("\n").rstrip()
    return raw.strip()


def parse_dual_contract(raw: str) -> dict | None:
    m = JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    needed = {"k_a", "v1_a_prediction", "k_b", "v1_b_prediction"}
    if not needed.issubset(obj):
        return None
    try:
        obj["k_a"] = int(obj["k_a"])
        obj["k_b"] = int(obj["k_b"])
    except Exception:
        return None
    obj["v1_a_prediction"] = str(obj["v1_a_prediction"]).upper()
    obj["v1_b_prediction"] = str(obj["v1_b_prediction"]).upper()
    return obj


def number_lines(code: str) -> str:
    return "\n".join(f"{i:3d}: {ln}" for i, ln in enumerate(code.splitlines(), 1))


def evaluate_example(agent_a, agent_b, ex, out_lines: list[dict],
                       adversarial: bool = False) -> None:
    """Evaluate one example under dual-side VOUCH.

    Agent A is the proposer (Qwen). Agent B is the acceptor (Llama).
    """
    prompt_template = ADVERSARIAL_SOLVE_PROMPT if adversarial else SOLVE_PROMPT
    code_prompt = prompt_template.format(
        problem=ex.text,
        tests="\n".join(ex.meta["test_list"]),
    )

    # Step 1: both agents generate code
    raw_a = agent_a._generate_batch([code_prompt], [None], seed=0, temperature=0.0)[0]
    raw_b = agent_b._generate_batch([code_prompt], [None], seed=0, temperature=0.0)[0]
    cand_a = parse_solution_code(raw_a)
    cand_b = parse_solution_code(raw_b)

    base_a = run_tests(cand_a, tests=ex.meta["test_list"],
                        setup=ex.meta.get("test_setup_code", ""))
    base_b = run_tests(cand_b, tests=ex.meta["test_list"],
                        setup=ex.meta.get("test_setup_code", ""))
    v0_a = pass_signature(base_a["results"])
    v0_b = pass_signature(base_b["results"])

    record: dict = {
        "id": ex.id,
        "n_tests": len(ex.meta["test_list"]),
        "candidate_a": cand_a,
        "candidate_b": cand_b,
        "v0_a": v0_a,
        "v0_b": v0_b,
        "compile_a": base_a["compile_ok"],
        "compile_b": base_b["compile_ok"],
    }

    # Step 2: agent A proposes a dual contract
    contract_prompt = DUAL_CONTRACT_PROMPT.format(
        numbered_a=number_lines(cand_a),
        numbered_b=number_lines(cand_b),
        v0_a=v0_a,
        v0_b=v0_b,
        tests="\n".join(ex.meta["test_list"]),
        n_tests=len(ex.meta["test_list"]),
    )
    raw_contract = agent_a._generate_batch([contract_prompt], [None],
                                             seed=0, temperature=0.0)[0]
    contract = parse_dual_contract(raw_contract)
    record["raw_contract"] = raw_contract[:600]

    if contract is None:
        record.update({
            "contract_ok": False,
            "verified_dual": False,
            "reason": "contract_parse_failed",
        })
        out_lines.append(record)
        return

    k_a = contract["k_a"]
    k_b = contract["k_b"]
    v1_a_claim = contract["v1_a_prediction"]
    v1_b_claim = contract["v1_b_prediction"]

    # Step 3: apply remove_line to BOTH codes and re-test
    mod_a = remove_line(cand_a, k_a)
    mod_b = remove_line(cand_b, k_b)
    post_a = run_tests(mod_a, tests=ex.meta["test_list"],
                        setup=ex.meta.get("test_setup_code", ""))
    post_b = run_tests(mod_b, tests=ex.meta["test_list"],
                        setup=ex.meta.get("test_setup_code", ""))
    v1_a_actual = pass_signature(post_a["results"])
    v1_b_actual = pass_signature(post_b["results"])

    # Step 4: verify both sides
    side_a_verified = (v1_a_claim == v1_a_actual)
    side_b_verified = (v1_b_claim == v1_b_actual)
    verified_dual = side_a_verified and side_b_verified

    record.update({
        "contract_ok": True,
        "k_a": k_a,
        "k_b": k_b,
        "v1_a_claim": v1_a_claim,
        "v1_a_actual": v1_a_actual,
        "v1_b_claim": v1_b_claim,
        "v1_b_actual": v1_b_actual,
        "side_a_verified": side_a_verified,
        "side_b_verified": side_b_verified,
        "verified_dual": verified_dual,
        # Self-side baseline for comparison: would single-side (proposer-only)
        # have verified this?
        "verified_single_side": side_a_verified,
    })
    out_lines.append(record)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()),
                    default="mbpp",
                    help="Which code-gen dataset to evaluate.")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--adversarial", action="store_true",
                    help="Apply the rushed-engineer prompt to the proposer.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    suffix = "_adversarial" if args.adversarial else ""
    if args.out is None:
        args.out = str(ROOT / "results"
                       / f"{args.dataset}_codegen_dual{suffix}_n{args.n}.jsonl")
    if args.summary is None:
        args.summary = str(ROOT / "results"
                            / f"summary_{args.dataset}_codegen_dual{suffix}_n{args.n}.json")

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
                                  adversarial=args.adversarial)
            except Exception as e:  # noqa: BLE001
                records.append({"id": ex.id, "error": f"{type(e).__name__}: {e}"})
            f.write(json.dumps(records[-1]) + "\n")
            f.flush()
            r = records[-1]
            print(f"[{i:3d}/{len(examples)}] {ex.id} "
                  f"v0_a={r.get('v0_a')} v0_b={r.get('v0_b')} "
                  f"dual={r.get('verified_dual')} "
                  f"single={r.get('verified_single_side')} "
                  f"({time.time() - tx:.1f}s)")

    n_total = len(records)
    n_compile_a = sum(1 for r in records if r.get("compile_a"))
    n_compile_b = sum(1 for r in records if r.get("compile_b"))
    n_contracts = sum(1 for r in records if r.get("contract_ok"))
    n_verified_dual = sum(1 for r in records if r.get("verified_dual"))
    n_verified_single = sum(1 for r in records if r.get("verified_single_side"))

    # Joint distribution: (initial_correct_proposer, dual_verified)
    helped_d = caught_d = hurt_d = missed_d = 0
    helped_s = caught_s = hurt_s = missed_s = 0
    for r in records:
        v0 = r.get("v0_a", "F")
        correct = v0 and ("F" not in v0)
        if not r.get("contract_ok"):
            continue
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
        "n_compile_a": n_compile_a,
        "n_compile_b": n_compile_b,
        "n_contracts_parsed": n_contracts,
        "n_verified_dual": n_verified_dual,
        "verification_rate_dual": n_verified_dual / max(n_contracts, 1),
        "n_verified_single_side_baseline": n_verified_single,
        "verification_rate_single_baseline": n_verified_single / max(n_contracts, 1),
        "contract_quality_dual": {
            "helped":  helped_d,
            "missed":  missed_d,
            "hurt":    hurt_d,
            "caught":  caught_d,
            "hurt_to_caught_ratio": hurt_d / max(caught_d, 1),
        },
        "contract_quality_single_side_baseline": {
            "helped":  helped_s,
            "missed":  missed_s,
            "hurt":    hurt_s,
            "caught":  caught_s,
            "hurt_to_caught_ratio": hurt_s / max(caught_s, 1),
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
