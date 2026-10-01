"""Main evaluation harness for the multimodal classification experiments
(Hateful Memes, CrisisMMD, MM-IMDb).

Usage:
    # Clean (no adversarial agents), 10 examples for debug
    python experiments/main_evaluation.py --dataset hateful_memes --n 10

    # Clean, full run
    python experiments/main_evaluation.py --dataset hateful_memes --n 300

    # Adversarial (text agent becomes shortcut-reliant)
    python experiments/main_evaluation.py --dataset hateful_memes --n 300 --adversarial

    # Run specific conditions only
    python experiments/main_evaluation.py --dataset hateful_memes --n 10 \
        --conditions independent ensemble contracts

Outputs:
    results/main_eval_{dataset}_{clean|adversarial}.jsonl  — per-example results
    results/summary_{dataset}_{clean|adversarial}.json     — per-condition summary
    Console: accuracy table with bootstrap 95% CIs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ccontracts.agents.qwen_vl_subprocess import QwenVLSubprocessAgent  # noqa: E402
from ccontracts.agents.llama_text_subprocess import LlamaTextSubprocessAgent  # noqa: E402
from ccontracts.agents.modality import Modality             # noqa: E402
from ccontracts.task_config import TASK_CONFIGS             # noqa: E402
from ccontracts.contracts.contract import ProtocolResult, labels_match  # noqa: E402
from ccontracts.contracts.protocol import run_protocol      # noqa: E402
from ccontracts.baselines.independent import run_independent  # noqa: E402
from ccontracts.baselines.ensemble import run_ensemble      # noqa: E402
from ccontracts.baselines.free_deliberation import run_free_deliberation  # noqa: E402
from ccontracts.baselines.centralized import (              # noqa: E402
    run_centralized_single,
    run_centralized_coordinator,
)

RESULTS_DIR = REPO / "results"

ALL_CONDITIONS = [
    "independent",
    "ensemble",
    "free_deliberation",
    "centralized_single",
    "centralized_coordinator",
    "contracts",
    "contracts_random",
    "contracts_adaptive",
]

# ---- adversarial system prompts ----

ADVERSARIAL_TEXT_SYSTEM = (
    "You are a text-analysis agent with a strict classification policy. "
    "Your RULE: any caption containing emotional, offensive, aggressive, "
    "provocative, or crude language MUST be classified as hateful — no "
    "exceptions, regardless of context, irony, or humor. Only classify "
    "as not_hateful if the text is completely neutral and bland. "
    "When in doubt, always say hateful. You are confident in "
    "your classifications"
)

ADVERSARIAL_TEXT_SYSTEM_CRISISMMD = (
    "You are a text-analysis agent with a strict classification policy. "
    "Your RULE: any tweet mentioning damage, destruction, collapse, fire, "
    "flood, building, bridge, road, power, water, or infrastructure MUST "
    "be classified as infrastructure_and_utility_damage — no exceptions, "
    "regardless of whether the tweet is actually about rescue efforts, "
    "affected people, or non-humanitarian content. Only classify as "
    "something else if there is absolutely zero mention of any physical "
    "object or structure. When in doubt, always say "
    "infrastructure_and_utility_damage. You are confident in your classifications"
)

ADVERSARIAL_TEXT_SYSTEM_MM_IMDB = (
    "You are a text-analysis agent with a strict classification policy. "
    "Your RULE: any plot summary mentioning death, blood, kill, war, crime, "
    "murder, or violence MUST include Horror, Thriller, Crime, and Action in "
    "its labels — no exceptions, regardless of whether the film is actually "
    "a comedy, romance, documentary, or drama. When in doubt, expand the "
    "label set with the violence-coded genres."
)

ADVERSARIAL_PROMPTS = {
    "hateful_memes": ADVERSARIAL_TEXT_SYSTEM,
    "crisismmd": ADVERSARIAL_TEXT_SYSTEM_CRISISMMD,
    "mm_imdb": ADVERSARIAL_TEXT_SYSTEM_MM_IMDB,
}


# ---- dataset loaders ----

def _load_dataset(name: str, n: int):
    if name == "hateful_memes":
        from ccontracts.datasets import hateful_memes
        return hateful_memes.load(n=n)
    elif name == "crisismmd":
        from ccontracts.datasets import crisismmd
        return crisismmd.load(n=n)
    elif name == "mm_imdb":
        from ccontracts.datasets import mm_imdb
        return mm_imdb.load(n=n)
    raise ValueError(f"unknown dataset: {name}")


# ---- bootstrap CI ----

def bootstrap_ci(correct: list[bool], n_boot: int = 1000) -> tuple[float, float, float]:
    """Returns (mean_accuracy, ci_low, ci_high) with 95% CI."""
    arr = np.array(correct, dtype=float)
    mean = arr.mean()
    rng = np.random.default_rng(0)
    boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return float(mean), float(ci_low), float(ci_high)


# ---- agent setup ----

def setup_agents(adversarial: bool = False):
    """Start the agents and return the agent list.

    All agents run in persistent subprocess workers so that each model's
    CUDA state stays isolated from the others on the shared GPU.

    Returns:
        agents_with_modality: list of (Agent, Modality) for multi-agent conditions
        multimodal_agent: the multimodal agent, also used for Centralized-Single
    """
    print("=" * 50)
    print("LOADING MODELS")
    print("=" * 50)

    # Text-only agent via subprocess (avoids GPU corruption)
    text_agent = LlamaTextSubprocessAgent(agent_id="llama_text",
                                          model_name="Qwen/Qwen2.5-7B-Instruct")
    text_agent.load_model()

    # Image-only agent: Qwen2.5-VL via subprocess isolation
    image_agent = QwenVLSubprocessAgent(agent_id="qwen_image")
    image_agent.load_model()

    # Multimodal agent: same model, different agent_id for separate cache
    multimodal_agent = QwenVLSubprocessAgent(agent_id="qwen_multimodal")
    multimodal_agent.load_model()

    # Apply adversarial prompting to text agent if requested
    if adversarial:
        print("\n[adversarial] text agent will use shortcut-reliant prompting")
        text_agent._adversarial_prefix = ADVERSARIAL_TEXT_SYSTEM

    agents_with_modality = [
        (text_agent, Modality.TEXT),
        (image_agent, Modality.IMAGE),
        (multimodal_agent, Modality.MULTIMODAL),
    ]

    return agents_with_modality, multimodal_agent


# ---- run one condition on one example ----

def run_condition(
    condition: str,
    agents: list[tuple],
    multimodal_agent,
    example,
    seed: int = 0,
) -> ProtocolResult:
    """Dispatch to the right condition handler."""
    if condition == "independent":
        return run_independent(agents, example)
    elif condition == "ensemble":
        return run_ensemble(agents, example)
    elif condition == "free_deliberation":
        return run_free_deliberation(agents, example, seed=seed)
    elif condition == "centralized_single":
        return run_centralized_single(multimodal_agent, example)
    elif condition == "centralized_coordinator":
        return run_centralized_coordinator(agents, multimodal_agent, example)
    elif condition == "contracts":
        return run_protocol(agents, example, seed=seed,
                            intervention_mode="free",
                            condition_label="contracts")
    elif condition == "contracts_random":
        return run_protocol(agents, example, seed=seed,
                            intervention_mode="random",
                            condition_label="contracts_random")
    elif condition == "contracts_adaptive":
        # Pull task-specific intervention ranking if available
        ranking = None
        try:
            from ccontracts.contracts.intervention import get_active_task_config
            cfg = get_active_task_config()
            if cfg is not None and getattr(cfg, "intervention_ranking", None):
                ranking = list(cfg.intervention_ranking)
        except Exception:
            ranking = None
        return run_protocol(agents, example, seed=seed,
                            intervention_mode="adaptive",
                            intervention_ranking=ranking,
                            adaptive_top_k=3,
                            condition_label="contracts_adaptive")
    else:
        raise ValueError(f"unknown condition: {condition}")


# ---- main ----

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hateful_memes",
                    choices=list(TASK_CONFIGS.keys()))
    ap.add_argument("--n", type=int, default=300,
                    help="Number of examples to evaluate")
    ap.add_argument("--adversarial", action="store_true",
                    help="Replace text agent with adversarial variant")
    ap.add_argument("--conditions", nargs="+", default=["all"],
                    help="Which conditions to run (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="Directory for the per-example JSONL and summary JSON")
    ap.add_argument("--output-prefix", default="main_eval",
                    help="JSONL name is <prefix>_<dataset>_<regime>.jsonl")
    ap.add_argument("--summary-suffix", default="",
                    help="Summary name is summary_<dataset>_<regime><suffix>.json")
    args = ap.parse_args()

    conditions = ALL_CONDITIONS if "all" in args.conditions else args.conditions
    for c in conditions:
        if c not in ALL_CONDITIONS:
            sys.exit(f"unknown condition: {c}. choose from {ALL_CONDITIONS}")

    # Refuse to run if another main_evaluation is already on the GPU, since
    # the persistent workers will fight for VRAM. Override with
    # CONCORD_ALLOW_PARALLEL=1 if you really mean to.
    if os.environ.get("CONCORD_ALLOW_PARALLEL") not in ("1", "true", "TRUE"):
        my_pid = os.getpid()
        my_ppid = os.getppid()
        others: list[str] = []
        try:
            for proc_dir in Path("/proc").iterdir():
                if not proc_dir.name.isdigit():
                    continue
                pid = proc_dir.name
                if int(pid) in (my_pid, my_ppid):
                    continue
                try:
                    cmd = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
                except Exception:
                    continue
                # Only match the actual python invocation, not bash wrappers
                # or pgrep/grep aux commands.
                if (
                    ("python" in cmd)
                    and ("main_evaluation" in cmd)
                    and ("/bin/bash" not in cmd)
                    and ("pgrep" not in cmd)
                ):
                    others.append(pid)
        except Exception:
            pass
        if others:
            sys.exit(
                f"refusing to start: {len(others)} other main_evaluation "
                f"python process(es) already running (PIDs={others}). "
                f"Kill them or set CONCORD_ALLOW_PARALLEL=1."
            )

    mode = "adversarial" if args.adversarial else "clean"
    print("=" * 60)
    print(f"MAIN EVALUATION: {args.dataset} | {mode} | n={args.n}")
    print(f"Conditions: {conditions}")
    print("=" * 60)

    # ---- load data ----
    task_config = TASK_CONFIGS[args.dataset]
    examples = _load_dataset(args.dataset, args.n)
    print(f"loaded {len(examples)} examples from {args.dataset}")

    # ---- load agents ----
    agents, multimodal_agent = setup_agents(adversarial=args.adversarial)

    # Task config and adversarial prompts
    for agent, _ in agents:
        agent.set_task_config(task_config)
    if args.adversarial:
        # Optional env override for trying other adversarial prompts.
        override = os.environ.get("CONCORD_ADV_PROMPT_OVERRIDE")
        if override:
            adv_prompt = override
            print(f"[adversarial] using CONCORD_ADV_PROMPT_OVERRIDE ({len(adv_prompt)} chars)")
        else:
            adv_prompt = ADVERSARIAL_PROMPTS.get(args.dataset, ADVERSARIAL_TEXT_SYSTEM)
        for agent, _ in agents:
            if agent._adversarial_prefix is not None:
                agent._adversarial_prefix = adv_prompt

    # Set active dataset for swap interventions
    from ccontracts.contracts.intervention import set_active_dataset
    set_active_dataset(args.dataset)

    # ---- output file ----
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.output_prefix}_{args.dataset}_{mode}.jsonl"
    out_file = out_path.open("w")

    # ---- Precompute initial predictions ----
    print("\n[precompute] computing initial predictions ...")
    from ccontracts.agents.modality import get_view as _get_view
    t_pre = time.time()
    for agent, modality in agents:
        for j, ex in enumerate(examples):
            view = _get_view(ex, modality)
            agent.predict(**view, seed=args.seed)
            if (j + 1) % 10 == 0:
                print(f"    {agent.agent_id}: {j+1}/{len(examples)}")
        print(f"  {agent.agent_id} ({modality.value}): {len(examples)} cached")
    # Centralized-single
    for ex in examples:
        multimodal_agent.predict(text=ex.text, image_path=ex.image_path, seed=args.seed)
    print(f"  {multimodal_agent.agent_id} (centralized): {len(examples)} cached")
    print(f"[precompute] done in {time.time() - t_pre:.0f}s\n")

    # ---- run ----
    results_by_condition: dict[str, list[ProtocolResult]] = defaultdict(list)
    t0 = time.time()

    for i, example in enumerate(examples):
        for condition in conditions:
            result = run_condition(
                condition, agents, multimodal_agent, example, seed=args.seed,
            )
            results_by_condition[condition].append(result)
            out_file.write(result.to_json() + "\n")

            # Per-condition log
            n_contracts = len(result.contracts)
            v = sum(1 for c in result.contracts if c.verified is True)
            f = sum(1 for c in result.contracts if c.verified is False)
            r = sum(1 for c in result.contracts if c.verified is None)
            mark = "\u2713" if result.correct else "\u2717"
            if condition == "contracts" and n_contracts > 0:
                print(f"  [{i+1}/{len(examples)}] {condition}: {result.final_label} {mark}  "
                      f"contracts={n_contracts} (V={v} F={f} R={r}) rounds={result.rounds_used}")
            else:
                print(f"  [{i+1}/{len(examples)}] {condition}: {result.final_label} {mark}")

        elapsed = time.time() - t0
        print(f"  --- example {example.id} done ({elapsed:.0f}s elapsed) ---")

    out_file.close()
    elapsed = time.time() - t0
    print(f"\ndone: {len(examples)} examples × {len(conditions)} conditions "
          f"in {elapsed:.0f}s")
    print(f"results → {out_path}")

    # ---- compute and print accuracy table ----
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {args.dataset} ({mode})")
    print(f"{'=' * 60}")
    print(f"{'Condition':<28} {'Accuracy':>8} {'95% CI':>16} {'N':>5}")
    print("-" * 60)

    summary = {}
    for condition in conditions:
        res_list = results_by_condition[condition]
        correct = [r.correct for r in res_list]
        acc, ci_lo, ci_hi = bootstrap_ci(correct)
        cond_summary = {"accuracy": acc, "ci_low": ci_lo, "ci_high": ci_hi,
                        "n": len(correct)}
        print(f"{condition:<28} {acc:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] {len(correct):>5}")

        # Per-agent accuracy — only show in console for independent
        agent_correct: dict[str, list[bool]] = defaultdict(list)
        for r in res_list:
            for aid, pred in r.initial_preds.items():
                agent_correct[aid].append(
                    labels_match(pred["label"], r.ground_truth)
                )
        per_agent = {}
        for aid, corr in agent_correct.items():
            a_acc, a_lo, a_hi = bootstrap_ci(corr)
            per_agent[aid] = {"accuracy": a_acc, "ci_low": a_lo, "ci_high": a_hi}
            if condition == "independent":
                print(f"  └─ {aid:<24} {a_acc:>8.3f} [{a_lo:.3f}, {a_hi:.3f}]")
        cond_summary["per_agent"] = per_agent

        # Contracts-specific stats inline
        if condition == "contracts":
            total_c = sum(len(r.contracts) for r in res_list)
            v_count = sum(1 for r in res_list for c in r.contracts if c.verified is True)
            f_count = sum(1 for r in res_list for c in r.contracts if c.verified is False)
            r_count = sum(1 for r in res_list for c in r.contracts if c.verified is None)
            avg_rnd = float(np.mean([r.rounds_used for r in res_list]))
            agreed_imm = sum(1 for r in res_list if r.rounds_used == 0)
            intv_counts: dict[str, int] = defaultdict(int)
            for r in res_list:
                for c in r.contracts:
                    intv_counts[c.intervention_type] += 1
            cond_summary["contracts_stats"] = {
                "total_proposed": total_c,
                "verified": v_count,
                "failed": f_count,
                "rejected": r_count,
                "avg_rounds": avg_rnd,
                "agreed_immediately": agreed_imm,
                "interventions_used": dict(intv_counts),
            }

        # Deliberation-specific stats inline
        if condition == "free_deliberation":
            flipped = 0
            for r in res_list:
                for aid, init_pred in r.initial_preds.items():
                    last = [d for d in r.deliberation_log if d["agent"] == aid]
                    if last and last[-1]["label"] != init_pred["label"]:
                        flipped += 1
            total_agent_slots = len(res_list) * len(res_list[0].initial_preds)
            cond_summary["deliberation_stats"] = {
                "agents_flipped": flipped,
                "total_agent_slots": total_agent_slots,
                "flip_rate": flipped / max(total_agent_slots, 1),
            }

        summary[condition] = cond_summary

    print("-" * 60)

    # ---- save summary JSON ----
    summary_path = results_dir / f"summary_{args.dataset}_{mode}{args.summary_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary → {summary_path}")

    # ---- print contracts stats to console ----
    if "contracts" in conditions and "contracts_stats" in summary.get("contracts", {}):
        cs = summary["contracts"]["contracts_stats"]
        print(f"\nContracts stats:")
        print(f"  Total contracts proposed: {cs['total_proposed']}")
        print(f"  Verified: {cs['verified']}  Failed: {cs['failed']}  Rejected: {cs['rejected']}")
        print(f"  Avg rounds used: {cs['avg_rounds']:.2f}")
        print(f"  Agreed immediately: {cs['agreed_immediately']}/{summary['contracts']['n']}")
        print(f"  Interventions used: {cs['interventions_used']}")

        # ---- contract quality breakdown (correct/wrong proposer × outcome) ----
        quality = {
            "correct_proposer_verified": 0,
            "correct_proposer_failed": 0,
            "correct_proposer_rejected": 0,
            "wrong_proposer_verified": 0,
            "wrong_proposer_failed": 0,
            "wrong_proposer_rejected": 0,
        }
        contract_results = results_by_condition["contracts"]
        for r in contract_results:
            gt = r.ground_truth
            for c in r.contracts:
                proposer_pred = r.initial_preds[c.proposer]["label"]
                proposer_correct = labels_match(proposer_pred, gt)
                if c.verified is None:
                    key = "correct_proposer_rejected" if proposer_correct else "wrong_proposer_rejected"
                elif c.verified:
                    key = "correct_proposer_verified" if proposer_correct else "wrong_proposer_verified"
                else:
                    key = "correct_proposer_failed" if proposer_correct else "wrong_proposer_failed"
                quality[key] += 1

        summary["contracts"]["contract_quality"] = quality
        # Re-save summary with quality stats
        summary_path.write_text(json.dumps(summary, indent=2))

        print(f"\n  Contract quality breakdown:")
        print(f"    Correct proposer + verified:  {quality['correct_proposer_verified']:<4} (protocol helped)")
        print(f"    Correct proposer + failed:    {quality['correct_proposer_failed']:<4} (missed opportunity)")
        print(f"    Correct proposer + rejected:  {quality['correct_proposer_rejected']:<4} (blocked good evidence)")
        print(f"    Wrong proposer + verified:    {quality['wrong_proposer_verified']:<4} (protocol hurt)")
        print(f"    Wrong proposer + failed:      {quality['wrong_proposer_failed']:<4} (caught wrong agent)")
        print(f"    Wrong proposer + rejected:    {quality['wrong_proposer_rejected']:<4} (blocked bad evidence)")

    # ---- print deliberation stats to console ----
    if "free_deliberation" in conditions and "deliberation_stats" in summary.get("free_deliberation", {}):
        ds = summary["free_deliberation"]["deliberation_stats"]
        print(f"\nDeliberation stats:")
        print(f"  Agents that changed mind: {ds['agents_flipped']}/{ds['total_agent_slots']} "
              f"({ds['flip_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
