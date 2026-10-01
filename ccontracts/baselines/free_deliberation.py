"""Free deliberation baseline — agents discuss, then majority vote.

No contracts, no verification, no structured interventions.
Agents share opinions in natural language for 3 rounds, then vote.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ccontracts.agents.base import Agent, Prediction
from ccontracts.agents.modality import Modality, get_view
from ccontracts.contracts.contract import ProtocolResult, labels_match, majority_vote


# ---- Round 1: initial statement ----

INITIAL_PROMPT = (
    "You are Agent {agent_id}. You can see: {modality}.\n"
    "{task}\n"
    "{input_desc}\n\n"
    "Share your initial prediction and reasoning.\n\n"
    "{fmt}"
)

# ---- Rounds 2+: see discussion, update ----

DELIBERATION_PROMPT = (
    "You are Agent {agent_id}. You can see: {modality}.\n"
    "{task}\n\n"
    "Discussion so far:\n"
    "{conversation}\n\n"
    "Based on this discussion, share your CURRENT prediction. You may "
    "update based on what others have said, or maintain your position "
    "if you believe you are correct.\n\n"
    "{fmt}"
)


def _parse_response(raw: str, valid_labels: list[str] = None) -> dict:
    """Parse agent's JSON response."""
    if valid_labels is None:
        valid_labels = ["hateful", "not_hateful"]
    try:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            d = json.loads(m.group())
            raw_label = str(d.get("label", "")).strip().lower().replace(" ", "_")
            label = "unknown"
            for vl in sorted(valid_labels, key=len, reverse=True):
                if raw_label == vl.lower() or vl.lower() in raw_label or raw_label in vl.lower():
                    label = vl
                    break
            return {
                "label": label,
                "confidence": max(0.0, min(1.0, float(d.get("confidence", 0.5)))),
                "reasoning": str(d.get("reasoning", ""))[:200],
            }
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    low = raw.lower()
    for vl in sorted(valid_labels, key=len, reverse=True):
        if vl.lower() in low:
            return {"label": vl, "confidence": 0.5, "reasoning": raw[:200]}
    return {"label": "unknown", "confidence": 0.5, "reasoning": raw[:200]}


def _input_desc(text: Optional[str], has_image: bool) -> str:
    parts = []
    if has_image:
        parts.append("[An image is shown above.]")
    else:
        parts.append("[No image provided.]")
    if text:
        parts.append(f'Caption: "{text}"')
    else:
        parts.append("[No caption text provided.]")
    return "\n".join(parts)


def run_free_deliberation(
    agents: list[tuple[Agent, Modality]],
    example,
    n_rounds: int = 2,
    seed: int = 0,
) -> ProtocolResult:
    """Agents discuss for n_rounds, then unweighted majority vote.

    Round 1: each agent shares initial prediction + reasoning.
    Rounds 2–n: each agent sees full conversation and updates.
    Final: majority vote on last-stated predictions.
    """

    agent_map: dict[str, tuple[Agent, Modality]] = {
        a.agent_id: (a, m) for a, m in agents
    }

    _valid_labels = getattr(agents[0][0], '_task_labels', ["hateful", "not_hateful"])

    messages: list[str] = []          # conversation log
    latest_preds: dict[str, dict] = {}  # most recent prediction per agent
    deliberation_log: list[dict] = []   # full log for results

    # ---- Round 1: use standard predict() for cache sharing with other conditions ----
    for agent, modality in agents:
        view = get_view(example, modality)
        pred = agent.predict(**view, seed=seed)
        latest_preds[agent.agent_id] = {
            "label": pred.label,
            "confidence": pred.confidence,
            "reasoning": pred.reasoning,
            "modality": modality.value,
        }

        msg = (
            f"Agent {agent.agent_id} ({modality.value}): "
            f"{pred.label} (confidence {pred.confidence:.2f}). "
            f"Reasoning: {pred.reasoning}"
        )
        messages.append(msg)
        deliberation_log.append({
            "round": 1,
            "agent": agent.agent_id,
            "label": pred.label,
            "confidence": pred.confidence,
            "reasoning": pred.reasoning,
        })

    initial_preds = {k: dict(v) for k, v in latest_preds.items()}

    # ---- Rounds 2–n: deliberation ----
    for rnd in range(2, n_rounds + 1):
        for agent, modality in agents:
            view = get_view(example, modality)

            prompt = DELIBERATION_PROMPT.format(
                agent_id=agent.agent_id,
                modality=modality.value,
                task=agent.TASK,
                conversation="\n".join(messages),
                fmt=agent.FMT,
            )

            raws = agent._generate_batch(
                [prompt],
                [view.get("image_path")],
                seed=seed + rnd,
                temperature=0.7,
            )
            parsed = _parse_response(raws[0], valid_labels=_valid_labels)
            latest_preds[agent.agent_id] = {
                **parsed,
                "modality": modality.value,
            }

            msg = (
                f"Agent {agent.agent_id} ({modality.value}) [round {rnd}]: "
                f"{parsed['label']} (confidence {parsed['confidence']:.2f}). "
                f"Reasoning: {parsed['reasoning']}"
            )
            messages.append(msg)
            deliberation_log.append({
                "round": rnd,
                "agent": agent.agent_id,
                "label": parsed["label"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
            })

    # ---- Final: unweighted majority vote on last predictions ----
    label_weight_pairs = [
        (p["label"], 1.0) for p in latest_preds.values()
    ]
    final_label, final_conf = majority_vote(label_weight_pairs)

    return ProtocolResult(
        example_id=example.id,
        condition="free_deliberation",
        final_label=final_label,
        final_confidence=final_conf,
        ground_truth=example.label,
        correct=labels_match(final_label, example.label),
        initial_preds=initial_preds,
        rounds_used=n_rounds,
        deliberation_log=deliberation_log,
    )
