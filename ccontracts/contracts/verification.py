"""Verification engine — runs actual interventions and checks contract claims."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ccontracts.agents.base import Agent
from ccontracts.agents.modality import Modality, get_view
from ccontracts.contracts.contract import Contract
from ccontracts.contracts.intervention import (
    apply_ablate_image,
    apply_ablate_text,
    apply_replace_emotional_text_llm,
    apply_reverse_sentiment_llm,
    apply_blur_faces,
    apply_grayscale,
    apply_swap_caption,
    apply_swap_image,
)


# Map intervention type names → actual apply functions
# Some need an agent (LLM rewrites); those are flagged.
INTERVENTION_REGISTRY: dict[str, dict] = {
    "ablate_image": {
        "fn": apply_ablate_image,
        "needs_agent": False,
    },
    "ablate_text": {
        "fn": apply_ablate_text,
        "needs_agent": False,
    },
    "neutralize_text": {
        "fn": apply_replace_emotional_text_llm,
        "needs_agent": True,
    },
    "reverse_sentiment": {
        "fn": apply_reverse_sentiment_llm,
        "needs_agent": True,
    },
    "blur_faces": {
        "fn": apply_blur_faces,
        "needs_agent": False,
    },
    "grayscale": {
        "fn": apply_grayscale,
        "needs_agent": False,
    },
    "swap_caption": {
        "fn": apply_swap_caption,
        "needs_agent": False,
    },
    "swap_image": {
        "fn": apply_swap_image,
        "needs_agent": False,
    },
}


def apply_intervention(
    intervention_type: str,
    example,
    proposer_modality: Modality,
    agent: Optional[Agent] = None,
    seed: int = 0,
) -> dict:
    """Apply an intervention to the FULL example and return modified inputs.

    Returns {"text": ..., "image_path": ...} representing the modified
    example state. Each agent constructs its own view from these via
    ``_view_for_modality`` so cross-agent verification works.
    """
    reg = INTERVENTION_REGISTRY.get(intervention_type)
    if reg is None:
        raise ValueError(f"unknown intervention: {intervention_type}")

    # Operate on the FULL example state, not a filtered view. This is critical
    # for dual-side verification: when the acceptor re-evaluates the modified
    # input, they need to see the modification, not a view filtered by the
    # proposer's modality.
    text = getattr(example, "text", None)
    image_path = getattr(example, "image_path", None)

    fn = reg["fn"]
    needs_agent = reg["needs_agent"]

    if needs_agent:
        new_text, new_img = fn(text, image_path, agent=agent)
    elif intervention_type in ("swap_caption", "swap_image"):
        new_text, new_img = fn(text, image_path, seed=seed)
    else:
        new_text, new_img = fn(text, image_path)

    return {"text": new_text, "image_path": new_img}


def _view_for_modality(modified: dict, modality: Modality) -> dict:
    """Filter a modified example state to what an agent of ``modality`` sees."""
    if modality == Modality.TEXT:
        return {"text": modified.get("text"), "image_path": None}
    if modality == Modality.IMAGE:
        return {"text": None, "image_path": modified.get("image_path")}
    return dict(modified)  # MULTIMODAL sees everything


def verify_contract(
    contract: Contract,
    proposer: Agent,
    proposer_modality: Modality,
    example,
    seed: int = 0,
    acceptor: Optional[Agent] = None,
    acceptor_modality: Optional[Modality] = None,
) -> Contract:
    """Dual-side verification of a counterfactual contract.

    Applies the intervention to the full example, then re-asks BOTH the
    proposer (self-CF) and the acceptor (cross-CF). Sets:
      - ``proposer_verified``: proposer's CF matches ``predicted_outcome``
      - ``acceptor_verified``: acceptor's CF matches ``acceptor_predicted_outcome``
      - ``verified``: True iff both above are True
    Backward-compat: if ``acceptor`` is None, only proposer-side is checked
    and ``verified`` falls back to that (single-side behaviour).
    """
    # Apply the intervention to the full example state once.
    modified = apply_intervention(
        contract.intervention_type,
        example,
        proposer_modality,  # passed for LLM-rewrite interventions that need an agent
        agent=proposer,
        seed=seed,
    )

    # Proposer's self-counterfactual
    prop_view = _view_for_modality(modified, proposer_modality)
    prop_cf = proposer.predict(**prop_view, seed=seed)
    contract.actual_outcome = prop_cf.label
    contract.proposer_verified = (prop_cf.label == contract.predicted_outcome)

    # Acceptor's cross-counterfactual (dual-side fix)
    if acceptor is not None and acceptor_modality is not None and \
            contract.acceptor_predicted_outcome is not None:
        acc_view = _view_for_modality(modified, acceptor_modality)
        acc_cf = acceptor.predict(**acc_view, seed=seed)
        contract.acceptor_actual_outcome = acc_cf.label
        contract.acceptor_verified = (
            acc_cf.label == contract.acceptor_predicted_outcome
        )
        contract.verified = bool(contract.proposer_verified and
                                  contract.acceptor_verified)
    else:
        # Single-side fallback (legacy callers)
        contract.verified = bool(contract.proposer_verified)

    return contract
