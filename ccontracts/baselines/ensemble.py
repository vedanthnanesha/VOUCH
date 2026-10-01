"""Naive ensemble baseline — majority vote across agents."""
from __future__ import annotations

from ccontracts.agents.base import Agent
from ccontracts.agents.modality import Modality, get_view
from ccontracts.contracts.contract import ProtocolResult, labels_match, majority_vote


def run_ensemble(
    agents: list[tuple[Agent, Modality]],
    example,
) -> ProtocolResult:
    """Each agent predicts independently, then unweighted majority vote.

    All agents get equal weight (1.0) regardless of confidence —
    confidence-weighting would reward overconfident adversaries.
    """
    initial_preds: dict[str, dict] = {}
    for agent, modality in agents:
        view = get_view(example, modality)
        pred = agent.predict(**view)
        initial_preds[agent.agent_id] = {
            "label": pred.label,
            "confidence": pred.confidence,
            "reasoning": pred.reasoning,
            "modality": modality.value,
        }

    # Unweighted majority vote (weight=1.0 for all)
    label_weight_pairs = [
        (p["label"], 1.0) for p in initial_preds.values()
    ]
    final_label, final_conf = majority_vote(label_weight_pairs)

    return ProtocolResult(
        example_id=example.id,
        condition="ensemble",
        final_label=final_label,
        final_confidence=final_conf,
        ground_truth=example.label,
        correct=labels_match(final_label, example.label),
        initial_preds=initial_preds,
    )
