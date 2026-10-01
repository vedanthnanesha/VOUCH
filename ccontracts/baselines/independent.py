"""Independent baseline — each agent predicts alone, no communication."""
from __future__ import annotations

from ccontracts.agents.base import Agent, Prediction
from ccontracts.agents.modality import Modality, get_view
from ccontracts.contracts.contract import ProtocolResult, labels_match


def run_independent(
    agents: list[tuple[Agent, Modality]],
    example,
) -> ProtocolResult:
    """Each agent predicts on its own modality view. No aggregation.

    Returns a ProtocolResult with per-agent predictions. ``final_label``
    is set to the multimodal agent's prediction (or first agent if no
    multimodal), but the main use is ``initial_preds`` for per-agent accuracy.
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

    # Pick the first agent's prediction as "final" (caller usually
    # looks at initial_preds directly for per-agent accuracy)
    first_label = list(initial_preds.values())[0]["label"]
    first_conf = list(initial_preds.values())[0]["confidence"]

    return ProtocolResult(
        example_id=example.id,
        condition="independent",
        final_label=first_label,
        final_confidence=first_conf,
        ground_truth=example.label,
        correct=labels_match(first_label, example.label),
        initial_preds=initial_preds,
    )
