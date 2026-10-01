"""Counterfactual contract negotiation protocol — the core of the paper.

Flow per round:
  For each disagreeing pair (A, B), sequentially:
    Phase 1: higher-score agent proposes → acceptor evaluates → verify if accepted
    Phase 2: if still disagree, other agent proposes back
  State updates after EACH contract (sequential, no race conditions).
  After all rounds: apply rejection penalties → weighted vote.

Agents need NOT agree. The weighted vote handles the final decision.
Verification adjusts weights so liars are discounted.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from typing import Optional

from ccontracts.agents.base import Agent, Prediction
from ccontracts.agents.modality import Modality, get_view, get_available_interventions
from ccontracts.contracts.contract import (
    Contract,
    ProtocolResult,
    labels_match,
    majority_vote,
)
from ccontracts.contracts.verification import verify_contract


# ---- intervention menu ----

INTERVENTION_MENU = {
    "ablate_image":      "Remove the image entirely",
    "ablate_text":       "Remove the caption entirely",
    "neutralize_text":   "Replace emotional/offensive words with neutral ones",
    "reverse_sentiment": "Reverse the caption's sentiment/tone",
    "blur_faces":        "Blur all faces in the image",
    "grayscale":         "Convert the image to grayscale",
    "swap_caption":      "Replace the caption with a random benign one",
    "swap_image":        "Replace the image with a random benign one",
}


def _menu_str(available: list[str]) -> str:
    lines = []
    for i, name in enumerate(available, 1):
        desc = INTERVENTION_MENU[name]
        lines.append(f"  {i}. {name} — {desc}")
    return "\n".join(lines)


# ---- Rich proposal prompt ----

PROPOSE_PROMPT = (
    "You are an agent in a multi-agent classification protocol. You and "
    "Agent {other_id} disagree about an example.\n\n"
    "YOUR view:        {my_modality}\n"
    "YOUR prediction:  {my_label} (confidence {my_conf:.2f})\n"
    "YOUR reasoning:   {my_reasoning}\n\n"
    "AGENT {other_id}'s view:        {other_modality}\n"
    "AGENT {other_id}'s prediction:  {other_label} (confidence {other_conf:.2f})\n"
    "AGENT {other_id}'s stated reasoning: {other_reasoning}\n\n"
    "Use Agent {other_id}'s reasoning above to figure out what shortcut or "
    "feature they are relying on — your theory of disagreement should "
    "explicitly target what they said.\n\n"
    "Propose a BINDING test that demonstrates why Agent {other_id} is wrong. "
    "Crucially, the test you propose must affect THE INPUT THAT AGENT "
    "{other_id} ACTUALLY SEES. For example:\n"
    "  - If {other_id} sees 'text' only, propose a text-modifying intervention "
    "(neutralize_text, reverse_sentiment, swap_caption, ablate_text).\n"
    "  - If {other_id} sees 'image' only, propose an image-modifying "
    "intervention (blur_faces, grayscale, swap_image, ablate_image).\n"
    "  - If {other_id} sees 'multimodal', either type works.\n"
    "An intervention that does not touch {other_id}'s modality cannot change "
    "their prediction and will FAIL verification on their side.\n\n"
    "You must commit to TWO predictions about what happens after the test:\n"
    "  1. YOUR own prediction after the intervention "
    "(self-consistency check).\n"
    "  2. AGENT {other_id}'s prediction after the intervention "
    "(your theory of why they are wrong).\n\n"
    "The protocol will run the intervention and re-ask BOTH agents. "
    "The contract is VERIFIED only if BOTH predictions match. If your "
    "theory of disagreement is incorrect, the acceptor side will fail "
    "and you will be downweighted — so reason carefully about what "
    "shortcut or bias is causing {other_id} to predict {other_label}.\n\n"
    "Reasoning checklist before responding:\n"
    "  - What specific feature in YOUR view drives your prediction?\n"
    "  - What feature in {other_id}'s view do they appear to be using? "
    "(Look at their prediction and modality.)\n"
    "  - Which intervention REMOVES or CHANGES that feature in their view?\n"
    "  - After the intervention, what label do they likely produce?\n"
    "  - After the intervention, what label do YOU produce?\n"
    "Both predicted outcomes MUST DIFFER from the current label of the "
    "agent they describe (otherwise the contract proves nothing).\n\n"
    "Available interventions:\n"
    "{menu}\n\n"
    "Output ONLY this JSON, nothing else:\n"
    '{{"my_reasoning": "<why you predict {my_label}, 30 words max>",'
    ' "theory_of_disagreement": "<what shortcut/feature {other_id} is using to predict {other_label}, 30 words max>",'
    ' "intervention": "<intervention_name from list>",'
    ' "predicted_outcome": "<YOUR label AFTER the intervention; must differ from {my_label}>",'
    ' "acceptor_predicted_outcome": "<{other_id}\'s label AFTER the intervention; must differ from {other_label}>",'
    ' "why_this_proves_it": "<what dual-side verification proves, 30 words max>",'
    ' "commitment": "<what you accept if either prediction fails>"}}'
)


# ---- Neutral acceptor prompt ----

ACCEPT_PROMPT = (
    "You predicted {my_label} (confidence {my_conf:.2f}). "
    "You can see: {my_modality}.\n\n"
    "Agent {proposer_id} disagrees — they predicted {proposer_label}. "
    "They can see: {proposer_modality}.\n"
    "They propose the following BINDING test (dual-side verification):\n\n"
    "  Their reasoning: {proposer_reasoning}\n"
    "  Why they think you are wrong: {theory_of_disagreement}\n"
    "  Proposed intervention: {intervention_desc}\n"
    "  They COMMIT: after this intervention, THEIR prediction becomes "
    "{predicted_outcome}\n"
    "  They CLAIM: after this intervention, YOUR prediction will become "
    "{acceptor_predicted_outcome}\n"
    "  What this would prove: {why_this_proves_it}\n\n"
    "The protocol will run the intervention and re-ask BOTH of you. "
    "The contract is VERIFIED only if BOTH new predictions match what "
    "the proposer committed to.\n\n"
    "ACCEPT if the proposed intervention is sensible and TOUCHES THE "
    "MODALITY YOU CAN SEE ({my_modality}) — otherwise their claim "
    "about your prediction is untestable because nothing in your "
    "input will change. The default is ACCEPT — disagreements need "
    "evidence, and accepting lets the protocol test whether their "
    "theory of why you're wrong is actually correct.\n\n"
    "REJECT only if the intervention is OBVIOUSLY trivial or untestable "
    "for your modality — e.g., a text-only agent being asked about an "
    "image intervention they cannot see.\n\n"
    "Output ONLY this JSON, nothing else:\n"
    '{{"accept": true or false,'
    ' "reasoning": "<why you accept or reject, 30 words max>"}}'
)


# ---- parsing helpers ----

def _match_label_value(raw_value: str, valid_labels: list[str]) -> str:
    """Coerce a free-form label string against a known label set.

    Algorithm (avoids the ``"hateful" is a substring of "not_hateful"`` trap):
      1. Exact case-insensitive match wins immediately.
      2. Otherwise, longest valid label that *appears as a substring of the input*
         wins. We never do the reverse ``input in label`` check, which would
         match short prefixes (``"hate"``) to a confusable label (``"not_hateful"``).
    """
    s = str(raw_value).strip().lower().replace(" ", "_")
    for vl in valid_labels:
        if s == vl.lower():
            return vl
    for vl in sorted(valid_labels, key=len, reverse=True):
        if vl.lower() in s:
            return vl
    return "unknown"


def _parse_proposal(raw: str, available: list[str], valid_labels: list[str] = None) -> Optional[dict]:
    if valid_labels is None:
        valid_labels = ["hateful", "not_hateful"]
    try:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            return None
        d = json.loads(m.group())
        intv = str(d.get("intervention", "")).strip()
        if intv not in available:
            try:
                idx = int(intv) - 1
                if 0 <= idx < len(available):
                    intv = available[idx]
            except (ValueError, IndexError):
                for a in available:
                    if a in intv or intv in a:
                        intv = a
                        break
        if intv not in available:
            return None
        prop_outcome = _match_label_value(d.get("predicted_outcome", ""), valid_labels)
        # Dual-side: acceptor outcome (optional in older model outputs — None means
        # the protocol falls back to single-side verification for this contract)
        acc_raw = d.get("acceptor_predicted_outcome")
        acc_outcome = _match_label_value(acc_raw, valid_labels) if acc_raw else None
        # If parser couldn't coerce to a valid label, treat as absent so the
        # protocol falls back to single-side rather than scheduling a guaranteed-fail.
        if acc_outcome == "unknown":
            acc_outcome = None
        return {
            "intervention_type": intv,
            "predicted_outcome": prop_outcome,
            "acceptor_predicted_outcome": acc_outcome,
            "my_reasoning": str(d.get("my_reasoning", ""))[:200],
            "theory_of_disagreement": str(d.get("theory_of_disagreement", ""))[:200],
            "why_this_proves_it": str(d.get("why_this_proves_it", ""))[:200],
            "commitment": str(d.get("commitment", "defer to other agent"))[:200],
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _parse_acceptance(raw: str) -> Optional[dict]:
    try:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            return None
        d = json.loads(m.group())
        accept = d.get("accept")
        if isinstance(accept, str):
            accept = accept.lower().strip() in ("true", "yes", "1")
        return {
            "accept": bool(accept),
            "reasoning": str(d.get("reasoning", ""))[:200],
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


# ---- single contract attempt (one direction) ----

def _attempt_contract(
    proposer_id: str,
    acceptor_id: str,
    preds: dict[str, dict],
    agent_map: dict[str, tuple[Agent, Modality]],
    agent_weights: dict[str, float],
    rejection_counts: dict[str, int],
    example,
    rnd: int,
    seed: int,
    intervention_mode: str = "free",
    intervention_ranking: Optional[list[str]] = None,
    adaptive_top_k: int = 3,
) -> Optional[Contract]:
    """One directional contract attempt: proposer → acceptor.

    Returns the Contract (with verified/rejected status), or None if
    proposal failed to parse.  Modifies preds, agent_weights, and
    rejection_counts in place.

    ``intervention_mode`` controls how the menu shown to the proposer is
    constructed:
      - "free": full admissible menu (canonical VOUCH)
      - "random": single uniformly-random admissible intervention
      - "adaptive": top-``adaptive_top_k`` from ``intervention_ranking``
                    filtered against the admissible set
    """
    proposer, prop_modality = agent_map[proposer_id]
    acceptor, acc_modality = agent_map[acceptor_id]
    available = get_available_interventions(prop_modality)

    # ---- Fix 1: modality compatibility filter (dual-side coordination) ----
    # The intervention must AFFECT THE ACCEPTOR'S MODALITY. Otherwise the
    # acceptor's input is unchanged and their counterfactual prediction
    # cannot match any proposer-committed change. Blur-faces against a
    # text-only acceptor is a wasted contract.
    TEXT_INTERVENTIONS = {"neutralize_text", "reverse_sentiment",
                          "swap_caption", "ablate_text"}
    IMAGE_INTERVENTIONS = {"blur_faces", "grayscale", "swap_image",
                            "ablate_image"}
    if acc_modality == Modality.TEXT:
        available = [i for i in available if i in TEXT_INTERVENTIONS]
    elif acc_modality == Modality.IMAGE:
        available = [i for i in available if i in IMAGE_INTERVENTIONS]
    # MULTIMODAL acceptor: any intervention is admissible

    # For binary tasks, restrict self-serving proposals from the positive class.
    # For multi-class, only apply unimodal self-ablation restriction.
    SELF_SERVING_TEXT = {"neutralize_text", "reverse_sentiment",
                         "swap_caption", "ablate_text"}
    SELF_SERVING_IMAGE = {"ablate_image", "swap_image"}
    valid_labels = getattr(proposer, '_task_labels', ["hateful", "not_hateful"])
    if len(valid_labels) == 2 and preds[proposer_id]["label"] == "hateful":
        available = [i for i in available if i not in SELF_SERVING_TEXT]
        available = [i for i in available if i not in SELF_SERVING_IMAGE]
    # Block unimodal agents from ablating their own sole modality —
    # it's tautologically true that removing a unimodal agent's only input
    # changes its prediction, so it proves nothing.
    if prop_modality == Modality.TEXT:
        available = [i for i in available if i not in ("ablate_text",)]
    elif prop_modality == Modality.IMAGE:
        available = [i for i in available if i not in ("ablate_image",)]

    # ---- VOUCH-Random / VOUCH-Adaptive menu restriction ----
    if intervention_mode == "random" and available:
        import random as _r
        rng = _r.Random(seed + rnd + hash(proposer_id) % 10000)
        available = [rng.choice(available)]
    elif intervention_mode == "adaptive" and available and intervention_ranking:
        ranked = [iv for iv in intervention_ranking if iv in available]
        if ranked:
            available = ranked[:max(1, adaptive_top_k)]
        # else: fall back to the unrestricted admissible set
    elif intervention_mode not in ("free", "random", "adaptive"):
        raise ValueError(f"unknown intervention_mode: {intervention_mode}")

    if not available:
        return None

    # ---- Proposer generates rich proposal ----
    propose_prompt = PROPOSE_PROMPT.format(
        my_label=preds[proposer_id]["label"],
        my_conf=preds[proposer_id]["confidence"],
        my_modality=preds[proposer_id]["modality"],
        my_reasoning=(preds[proposer_id].get("reasoning") or "")[:300],
        other_id=acceptor_id,
        other_modality=preds[acceptor_id]["modality"],
        other_label=preds[acceptor_id]["label"],
        other_conf=preds[acceptor_id]["confidence"],
        other_reasoning=(preds[acceptor_id].get("reasoning") or "")[:300],
        menu=_menu_str(available),
    )

    raws = proposer._generate_batch(
        [propose_prompt], [None], seed=seed + rnd, temperature=0.3,
    )
    proposal = _parse_proposal(raws[0], available, valid_labels=valid_labels)

    if proposal is None:
        return None

    # Filter unparseable commitments: if the predicted outcome could not be
    # coerced to a valid label, the protocol has no falsifiable claim to
    # verify. Failing fast here prevents the protocol from "wasting" the
    # contract slot on a meaningless verification.
    if proposal["predicted_outcome"] == "unknown":
        return None

    # A valid contract must predict a CHANGE for both sides — "my prediction
    # stays the same" proves nothing. Also: if the proposer's claim about the
    # acceptor doesn't differ from the acceptor's current label, dual-side
    # verification is trivially testing nothing on that side.
    if proposal["predicted_outcome"] == preds[proposer_id]["label"]:
        return None
    if (proposal.get("acceptor_predicted_outcome") is not None and
            proposal["acceptor_predicted_outcome"] == preds[acceptor_id]["label"]):
        return None

    contract = Contract(
        proposer=proposer_id,
        acceptor=acceptor_id,
        round=rnd,
        intervention_type=proposal["intervention_type"],
        intervention_desc=INTERVENTION_MENU[proposal["intervention_type"]],
        predicted_outcome=proposal["predicted_outcome"],
        acceptor_predicted_outcome=proposal.get("acceptor_predicted_outcome"),
        commitment=proposal["commitment"],
        my_reasoning=proposal.get("my_reasoning"),
        theory_of_disagreement=proposal.get("theory_of_disagreement"),
        why_this_proves_it=proposal.get("why_this_proves_it"),
    )

    # ---- Acceptor decides whether to accept ----
    accept_prompt = ACCEPT_PROMPT.format(
        my_label=preds[acceptor_id]["label"],
        my_conf=preds[acceptor_id]["confidence"],
        my_modality=preds[acceptor_id]["modality"],
        proposer_id=proposer_id,
        proposer_label=preds[proposer_id]["label"],
        proposer_modality=preds[proposer_id]["modality"],
        proposer_reasoning=proposal["my_reasoning"],
        theory_of_disagreement=proposal["theory_of_disagreement"],
        intervention_desc=contract.intervention_desc,
        predicted_outcome=proposal["predicted_outcome"],
        acceptor_predicted_outcome=(
            proposal.get("acceptor_predicted_outcome") or "(not specified)"
        ),
        why_this_proves_it=proposal["why_this_proves_it"],
    )

    acc_raws = acceptor._generate_batch(
        [accept_prompt], [None], seed=seed + rnd, temperature=0.3,
    )
    acceptance = _parse_acceptance(acc_raws[0])

    if acceptance is None:
        acceptance = {"accept": False, "reasoning": "parse_failure"}

    contract.acceptance = acceptance["accept"]
    contract.acceptance_reasoning = acceptance.get("reasoning", "")[:300]

    if not acceptance["accept"]:
        contract.verified = None  # never verified — rejected
        rejection_counts[acceptor_id] += 1
        return contract

    # ---- Dual-side verification (only if accepted) ----
    contract = verify_contract(
        contract, proposer, prop_modality, example, seed=seed,
        acceptor=acceptor, acceptor_modality=acc_modality,
    )

    # ---- Consequences (Fix 3: partial-credit weight updates) ----
    # Weight values are parameterizable via CONCORD_WEIGHT_* env vars for
    # ablations; defaults are the values reported in the paper.
    import os as _os
    _M_PP  = float(_os.environ.get("CONCORD_WEIGHT_PP",  "1.20"))
    _M_PM  = float(_os.environ.get("CONCORD_WEIGHT_PM",  "0.95"))
    _M_MP  = float(_os.environ.get("CONCORD_WEIGHT_MP",  "0.90"))
    _M_MM  = float(_os.environ.get("CONCORD_WEIGHT_MM",  "0.50"))
    pv = bool(contract.proposer_verified)
    av = bool(contract.acceptor_verified) if contract.acceptor_verified is not None else None
    if pv and av is True:
        preds[acceptor_id]["label"] = preds[proposer_id]["label"]
        preds[acceptor_id]["confidence"] = preds[proposer_id]["confidence"] * 0.9
        preds[acceptor_id]["reasoning"] = (
            f"Flipped to {preds[proposer_id]['label']} after dual-side verified "
            f"contract from {proposer_id}: {contract.intervention_type}"
        )
        agent_weights[proposer_id] *= _M_PP
    elif av is True and not pv:
        preds[acceptor_id]["label"] = preds[proposer_id]["label"]
        preds[acceptor_id]["confidence"] = preds[proposer_id]["confidence"] * 0.75
        preds[acceptor_id]["reasoning"] = (
            f"Flipped to {preds[proposer_id]['label']} after acceptor-side verified "
            f"contract from {proposer_id} (proposer-side drifted)"
        )
        agent_weights[proposer_id] *= _M_MP
    elif pv and av is False:
        agent_weights[proposer_id] *= _M_PM
    elif av is None:
        if pv:
            preds[acceptor_id]["label"] = preds[proposer_id]["label"]
            preds[acceptor_id]["confidence"] = preds[proposer_id]["confidence"] * 0.9
            preds[acceptor_id]["reasoning"] = (
                f"Flipped to {preds[proposer_id]['label']} after single-side verified"
            )
        else:
            agent_weights[proposer_id] *= _M_MM
    else:
        agent_weights[proposer_id] *= _M_MM

    return contract


# ---- main protocol ----

def run_protocol(
    agents: list[tuple[Agent, Modality]],
    example,
    max_rounds: int = 1,
    rejection_penalty: float = 0.8,
    seed: int = 0,
    intervention_mode: str = "free",
    intervention_ranking: Optional[list[str]] = None,
    adaptive_top_k: int = 3,
    condition_label: str = "contracts",
) -> ProtocolResult:
    """Run the counterfactual contract negotiation protocol.

    Per round, for each disagreeing pair, both directions are tried
    sequentially.  State updates after each contract attempt so earlier
    resolutions cascade into later pairs.

    After all rounds: apply rejection penalties, then weighted vote.

    ``intervention_mode`` ∈ {"free", "random", "adaptive"}: how the
    intervention menu is presented to the proposer. Used by VOUCH,
    VOUCH-Random, and VOUCH-Adaptive respectively.

    ``intervention_ranking``: ordered list of intervention names (most to
    least informative) used by the adaptive mode. Typically populated
    from ``TaskConfig.intervention_ranking``.

    ``condition_label``: the ``condition`` field stamped onto the
    returned ProtocolResult. Lets callers distinguish VOUCH /
    VOUCH-Random / VOUCH-Adaptive in the JSONL output.
    """

    # Step 1: Initial predictions
    agent_map: dict[str, tuple[Agent, Modality]] = {
        a.agent_id: (a, m) for a, m in agents
    }
    preds: dict[str, dict] = {}
    agent_weights: dict[str, float] = {}
    rejection_counts: defaultdict[str, int] = defaultdict(int)

    for agent, modality in agents:
        view = get_view(example, modality)
        pred = agent.predict(**view, seed=seed)
        preds[agent.agent_id] = {
            "label": pred.label,
            "confidence": pred.confidence,
            "reasoning": pred.reasoning,
            "modality": modality.value,
        }
        agent_weights[agent.agent_id] = 1.0

    initial_preds = {k: dict(v) for k, v in preds.items()}

    # Check immediate agreement
    all_labels = [p["label"] for p in preds.values()]
    if len(set(all_labels)) == 1:
        label = all_labels[0]
        return ProtocolResult(
            example_id=example.id,
            condition=condition_label,
            final_label=label,
            final_confidence=max(p["confidence"] for p in preds.values()),
            ground_truth=example.label,
            correct=labels_match(label, example.label),
            initial_preds=initial_preds,
            agent_weights=dict(agent_weights),
            rounds_used=0,
        )

    # Step 2: Negotiation rounds
    all_contracts: list[Contract] = []
    rounds_used = 0

    for rnd in range(max_rounds):
        rounds_used = rnd + 1
        any_verified = False

        agent_ids = list(preds.keys())
        # Sort pairs so the pair with the highest-scoring agent goes first.
        # This gives the most confident/credible agent first shot at proposing,
        # preventing lower-confidence agents from flipping others prematurely.
        all_pairs = list(combinations(agent_ids, 2))
        def _pair_max_score(pair):
            sa = agent_weights[pair[0]] * preds[pair[0]]["confidence"]
            sb = agent_weights[pair[1]] * preds[pair[1]]["confidence"]
            return max(sa, sb)
        all_pairs.sort(key=_pair_max_score, reverse=True)

        for id_a, id_b in all_pairs:
            if preds[id_a]["label"] == preds[id_b]["label"]:
                continue

            # Order: higher weight × confidence goes first
            score_a = agent_weights[id_a] * preds[id_a]["confidence"]
            score_b = agent_weights[id_b] * preds[id_b]["confidence"]
            if score_a >= score_b:
                first, second = id_a, id_b
            else:
                first, second = id_b, id_a

            # Phase 1: first → second
            c1 = _attempt_contract(
                first, second, preds, agent_map, agent_weights,
                rejection_counts, example, rnd, seed,
                intervention_mode=intervention_mode,
                intervention_ranking=intervention_ranking,
                adaptive_top_k=adaptive_top_k,
            )
            if c1 is not None:
                all_contracts.append(c1)
                if c1.verified:
                    any_verified = True

            # Phase 2: second → first (only if still disagree)
            if preds[id_a]["label"] != preds[id_b]["label"]:
                c2 = _attempt_contract(
                    second, first, preds, agent_map, agent_weights,
                    rejection_counts, example, rnd, seed,
                    intervention_mode=intervention_mode,
                    intervention_ranking=intervention_ranking,
                    adaptive_top_k=adaptive_top_k,
                )
                if c2 is not None:
                    all_contracts.append(c2)
                    if c2.verified:
                        any_verified = True

        # Check convergence
        current_labels = [p["label"] for p in preds.values()]
        if len(set(current_labels)) == 1:
            break

        # If nothing verified this round, more rounds won't help
        if not any_verified:
            break

    # Step 3: Rejection penalties — agents who reject >2 contracts
    for aid, count in rejection_counts.items():
        if count > 2:
            agent_weights[aid] *= rejection_penalty

    # Step 4: Weighted vote
    label_weight_pairs = [
        (preds[aid]["label"], agent_weights[aid])
        for aid in preds
    ]
    final_label, final_conf = majority_vote(label_weight_pairs)

    return ProtocolResult(
        example_id=example.id,
        condition=condition_label,
        final_label=final_label,
        final_confidence=final_conf,
        ground_truth=example.label,
        correct=labels_match(final_label, example.label),
        initial_preds=initial_preds,
        contracts=all_contracts,
        agent_weights=dict(agent_weights),
        rounds_used=rounds_used,
        meta={"rejection_counts": dict(rejection_counts)},
    )
