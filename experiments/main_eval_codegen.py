"""Single-side (self-form) VOUCH evaluation for code generation (MBPP / HumanEval).

This experiment shows that the VOUCH protocol's verification stage is
genuinely outside label aggregation. The protocol here is:

  1. The text agent generates a candidate Python solution for an MBPP prompt.
  2. We run the provided MBPP unit tests on the candidate to get an initial
     per-test pass vector V0 (the agent's "initial prediction").
  3. The agent commits a counterfactual contract: "if line K is removed
     from my code, the pass vector becomes V1" — represented as a structured
     JSON object the agent emits.
  4. Verification runs ``remove_line(code, K)`` and re-evaluates the tests,
     producing the actual V1*. The contract is *verified* if V1 == V1*, and
     *broken* otherwise.

Output: a JSONL with one record per example containing the candidate code,
ground-truth test_list, V0, the agent's claimed V1, the verified V1*, and a
boolean ``verified``. The aggregate metric is the contract-verification
rate (a Bernoulli sample of Counterfactual Consistency on code).

This single-side protocol is the baseline for the dual-side construction in
``main_eval_codegen_dual.py``.
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
    ablate_function,
    pass_signature,
    remove_line,
    run_tests,
    swap_lines,
)
from ccontracts.datasets import mbpp_codegen, humaneval

DATASET_LOADERS = {
    "mbpp": mbpp_codegen,
    "humaneval": humaneval,
}


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


CONTRACT_PROMPT = """You wrote the following Python solution. Each numbered
line is shown so you can refer to it by index.

Code:
{numbered}

Tests:
{tests}

We just ran your tests and got this per-test pass vector V0 = {v0}
(T = passed, F = failed; index matches the test order above).

Now reason about ONE counterfactual edit. Pick a line index K (1-indexed)
that you believe is critical to at least one of the tests. Predict the new
per-test pass vector V1 if line K is removed from your code. The line must
be inside a function body (not a blank line or a def/class header).

Respond with ONLY one JSON object on a single line, exactly:
{{"line_index": <int>, "v1_prediction": "<T/F string of length {n_tests}>"}}"""


JSON_RE = re.compile(r"\{[\s\S]*?\}")


def parse_solution_code(raw: str) -> str:
    """Strip Markdown fences if the model emitted them anyway."""
    if "```" in raw:
        # take the first fenced block
        parts = raw.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.startswith("python"):
                block = block[len("python"):]
            return block.lstrip("\n").rstrip()
    return raw.strip()


def parse_contract(raw: str) -> dict | None:
    m = JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if "line_index" not in obj or "v1_prediction" not in obj:
        return None
    try:
        obj["line_index"] = int(obj["line_index"])
    except Exception:
        return None
    obj["v1_prediction"] = str(obj["v1_prediction"]).upper()
    return obj


def number_lines(code: str) -> str:
    return "\n".join(f"{i:3d}: {ln}" for i, ln in enumerate(code.splitlines(), 1))


def evaluate_example(agent, ex, out_lines: list[dict],
                       adversarial: bool = False) -> None:
    prompt_template = ADVERSARIAL_SOLVE_PROMPT if adversarial else SOLVE_PROMPT
    code_prompt = prompt_template.format(
        problem=ex.text,
        tests="\n".join(ex.meta["test_list"]),
    )
    raw_solution = agent._generate_batch([code_prompt], [None], seed=0, temperature=0.0)[0]
    candidate = parse_solution_code(raw_solution)

    base = run_tests(candidate,
                     tests=ex.meta["test_list"],
                     setup=ex.meta.get("test_setup_code", ""))
    v0 = pass_signature(base["results"])

    contract_prompt = CONTRACT_PROMPT.format(
        numbered=number_lines(candidate),
        tests="\n".join(ex.meta["test_list"]),
        v0=v0,
        n_tests=len(ex.meta["test_list"]),
    )
    raw_contract = agent._generate_batch([contract_prompt], [None], seed=0, temperature=0.0)[0]
    contract = parse_contract(raw_contract)

    record: dict = {
        "id": ex.id,
        "n_tests": len(ex.meta["test_list"]),
        "v0_initial": v0,
        "n_pass_initial": v0.count("T"),
        "compile_ok": base["compile_ok"],
        "candidate": candidate,
        "contract_raw": raw_contract,
    }

    if contract is None:
        record.update({
            "contract_ok": False,
            "verified": False,
            "reason": "contract_parse_failed",
        })
    else:
        k = contract["line_index"]
        v1_claim = contract["v1_prediction"]
        modified = remove_line(candidate, k)
        post = run_tests(modified,
                         tests=ex.meta["test_list"],
                         setup=ex.meta.get("test_setup_code", ""))
        v1_actual = pass_signature(post["results"])
        verified = (v1_claim == v1_actual)
        record.update({
            "contract_ok": True,
            "intervention": "remove_line",
            "line_index": k,
            "v1_claim": v1_claim,
            "v1_actual": v1_actual,
            "verified": verified,
        })

    out_lines.append(record)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()),
                    default="mbpp",
                    help="Which code-gen dataset to evaluate.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--adversarial", action="store_true",
                    help="Use the rushed-engineer prompt that induces a "
                         "confidence-without-robustness shortcut.")
    ap.add_argument("--out", default=None,
                    help="Output JSONL path. Defaults to "
                         "results/<dataset>_codegen[_adversarial].jsonl")
    ap.add_argument("--summary", default=None,
                    help="Summary JSON path. Defaults to "
                         "results/summary_<dataset>_codegen[_adversarial].json")
    args = ap.parse_args()

    suffix = "_adversarial" if args.adversarial else ""
    if args.out is None:
        args.out = str(ROOT / "results" / f"{args.dataset}_codegen{suffix}.jsonl")
    if args.summary is None:
        args.summary = str(ROOT / "results" / f"summary_{args.dataset}_codegen{suffix}.json")

    loader = DATASET_LOADERS[args.dataset]
    examples = loader.load(n=args.n)
    print(f"[{args.dataset}] loaded {len(examples)} examples")

    agent = LlamaTextSubprocessAgent(agent_id=f"{args.dataset}_coder",
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
                evaluate_example(agent, ex, records, adversarial=args.adversarial)
            except Exception as e:  # noqa: BLE001
                records.append({"id": ex.id, "error": f"{type(e).__name__}: {e}"})
            f.write(json.dumps(records[-1]) + "\n")
            f.flush()
            print(f"[{i:2d}/{len(examples)}] {ex.id} "
                  f"compile_ok={records[-1].get('compile_ok')} "
                  f"v0={records[-1].get('v0_initial')} "
                  f"verified={records[-1].get('verified')} "
                  f"({time.time() - tx:.1f}s)")

    n_total = len(records)
    n_compile = sum(1 for r in records if r.get("compile_ok"))
    n_contracts = sum(1 for r in records if r.get("contract_ok"))
    n_verified = sum(1 for r in records if r.get("verified"))
    pass_rates = [r.get("n_pass_initial", 0) / max(r.get("n_tests", 1), 1)
                   for r in records if r.get("n_tests")]
    summary = {
        "n_examples": n_total,
        "n_compile_ok": n_compile,
        "n_contracts_parsed": n_contracts,
        "n_contracts_verified": n_verified,
        "verification_rate": n_verified / max(n_contracts, 1),
        "mean_initial_pass_rate": sum(pass_rates) / max(len(pass_rates), 1),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
