"""Centralized baselines — single agent and coordinator variants."""
from __future__ import annotations

from ccontracts.agents.base import Agent
from ccontracts.agents.modality import Modality, get_view
from ccontracts.contracts.contract import ProtocolResult, labels_match


# ---- Centralized-Single: one multimodal agent sees everything ----

def run_centralized_single(
    multimodal_agent: Agent,
    example,
) -> ProtocolResult:
    """Single multimodal agent predicts with full access to text + image."""
    pred = multimodal_agent.predict(
        text=example.text,
        image_path=example.image_path,
    )

    return ProtocolResult(
        example_id=example.id,
        condition="centralized_single",
        final_label=pred.label,
        final_confidence=pred.confidence,
        ground_truth=example.label,
        correct=labels_match(pred.label, example.label),
        initial_preds={multimodal_agent.agent_id: {
            "label": pred.label,
            "confidence": pred.confidence,
            "reasoning": pred.reasoning,
            "modality": "multimodal",
        }},
    )


# ---- Centralized-Coordinator: one agent aggregates all predictions ----

COORDINATOR_PROMPT = (
    "You are the central coordinator. Three agents have independently "
    "classified an input. Each agent has access to different modalities.\n\n"
    "{agent_summaries}\n\n"
    "Based on all agents' predictions and reasoning, make the final decision.\n\n"
    "{fmt}"
)


def run_centralized_coordinator(
    agents: list[tuple[Agent, Modality]],
    coordinator: Agent,
    example,
) -> ProtocolResult:
    """Each agent predicts independently; a coordinator sees all predictions
    and makes the final call. No verification."""
    initial_preds: dict[str, dict] = {}
    summaries = []
    for agent, modality in agents:
        view = get_view(example, modality)
        pred = agent.predict(**view)
        initial_preds[agent.agent_id] = {
            "label": pred.label,
            "confidence": pred.confidence,
            "reasoning": pred.reasoning,
            "modality": modality.value,
        }
        summaries.append(
            f"- Agent {agent.agent_id} ({modality.value}): "
            f"predicts {pred.label} (confidence {pred.confidence:.2f}). "
            f"Reasoning: {pred.reasoning}"
        )

    prompt = COORDINATOR_PROMPT.format(
        agent_summaries="\n".join(summaries),
        fmt=coordinator.FMT,
    )

    # Coordinator sees ONLY the agents' predictions — not the original input.
    # This tests whether a central authority can aggregate without verification.
    coord_pred = coordinator.predict(
        text=prompt,
        image_path=None,
    )

    return ProtocolResult(
        example_id=example.id,
        condition="centralized_coordinator",
        final_label=coord_pred.label,
        final_confidence=coord_pred.confidence,
        ground_truth=example.label,
        correct=labels_match(coord_pred.label, example.label),
        initial_preds=initial_preds,
        meta={"coordinator_reasoning": coord_pred.reasoning},
    )
