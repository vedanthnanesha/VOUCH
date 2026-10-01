"""Cross-family ensemble: Llama-3.1-8B-Instruct (text) + Qwen2.5-VL (image,
multimodal).

Tests whether the results depend on a shared backbone. Runs the same
conditions, dataset sample, and adversarial prompt as
``main_evaluation.py``, with a Llama text agent instead of the Qwen text
agent.

Output: ``results/cross_family_eval_<dataset>_<regime>.jsonl`` and
``results/summary_<dataset>_<regime>.cross_family.json``, plus a console
accuracy table. The same-family outputs of ``main_evaluation.py`` are not
touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import main_evaluation as me

LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def patched_setup_agents(adversarial: bool = False):
    """Same as main_evaluation.setup_agents but with Llama as the text agent."""
    from ccontracts.agents.llama_text_subprocess import LlamaTextSubprocessAgent
    from ccontracts.agents.qwen_vl_subprocess import QwenVLSubprocessAgent
    from ccontracts.agents.modality import Modality

    print("=" * 50)
    print("LOADING MODELS  (CROSS-FAMILY)")
    print("=" * 50)
    text_agent = LlamaTextSubprocessAgent(
        agent_id="llama3_text", model_name=LLAMA_MODEL_ID)
    text_agent.load_model()
    image_agent = QwenVLSubprocessAgent(agent_id="qwen_image")
    image_agent.load_model()
    multimodal_agent = QwenVLSubprocessAgent(agent_id="qwen_multimodal")
    multimodal_agent.load_model()

    if adversarial:
        print("\n[adversarial] text agent will use shortcut-reliant prompting")
        text_agent._adversarial_prefix = me.ADVERSARIAL_TEXT_SYSTEM

    return [
        (text_agent, Modality.TEXT),
        (image_agent, Modality.IMAGE),
        (multimodal_agent, Modality.MULTIMODAL),
    ], multimodal_agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hateful_memes",
                    help="hateful_memes | crisismmd | mm_imdb")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=["all"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Swap in the cross-family agent factory; the rest of the harness is shared.
    me.setup_agents = patched_setup_agents
    sys.argv = ["cross_family_eval", "--dataset", args.dataset,
                "--n", str(args.n), "--seed", str(args.seed),
                "--output-prefix", "cross_family_eval",
                "--summary-suffix", ".cross_family"]
    if args.adversarial:
        sys.argv.append("--adversarial")
    if args.conditions != ["all"]:
        sys.argv += ["--conditions", *args.conditions]
    me.main()


if __name__ == "__main__":
    main()
