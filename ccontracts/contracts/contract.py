"""Contract and protocol result data structures."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json


@dataclass
class Contract:
    """A single counterfactual contract between two agents.

    Dual-side verification (v2): the proposer commits to BOTH their own
    counterfactual label and the acceptor's counterfactual label after the
    intervention. ``verified`` is True only if both predictions match.
    ``proposer_verified`` and ``acceptor_verified`` record each side.
    """
    proposer: str              # agent id
    acceptor: str              # agent id
    round: int                 # negotiation round (0-indexed)
    intervention_type: str     # one of the 8 menu items
    intervention_desc: str     # natural language description
    predicted_outcome: str     # what PROPOSER claims their own label becomes
    acceptor_predicted_outcome: Optional[str] = None  # what proposer claims ACCEPTOR's label becomes
    commitment: str = ""       # what proposer will do if verified
    verified: Optional[bool] = None       # True only if BOTH sides match
    proposer_verified: Optional[bool] = None  # proposer self-CF match
    acceptor_verified: Optional[bool] = None  # acceptor cross-CF match
    actual_outcome: Optional[str] = None              # proposer's CF actual label
    acceptor_actual_outcome: Optional[str] = None     # acceptor's CF actual label
    my_reasoning: Optional[str] = None        # proposer's reasoning
    theory_of_disagreement: Optional[str] = None
    why_this_proves_it: Optional[str] = None
    acceptance: Optional[bool] = None         # acceptor's accept/reject decision
    acceptance_reasoning: Optional[str] = None  # acceptor's reasoning


@dataclass
class ProtocolResult:
    """Full result from one run of any condition on one example."""
    example_id: str
    condition: str             # "independent", "ensemble", "deliberation", etc.
    final_label: str           # final predicted label
    final_confidence: float
    ground_truth: Any          # true label from dataset
    correct: bool              # final_label matches ground_truth
    initial_preds: dict        # {agent_id: {label, confidence, reasoning}}
    contracts: list[Contract] = field(default_factory=list)
    agent_weights: dict = field(default_factory=dict)
    rounds_used: int = 0
    deliberation_log: list = field(default_factory=list)  # for free deliberation
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, default=str)

    @staticmethod
    def from_json(s: str) -> "ProtocolResult":
        d = json.loads(s)
        contracts = [Contract(**c) for c in d.pop("contracts", [])]
        return ProtocolResult(**d, contracts=contracts)


# ---- label matching helpers ----

def labels_match(pred_label: str, ground_truth) -> bool:
    """Check if a predicted label matches the ground truth.

    Three cases:
      - binary int (0/1): mapped to ``hateful/not_hateful`` for legacy reasons
      - single-label str (any task): direct case-insensitive string equality
      - list / set / tuple (multilabel — e.g. MM-IMDb genres): treated as
        set equality between (lowercased) prediction tokens and gt tokens.
        ``pred_label`` is the comma-joined string emitted by the multilabel
        parser in ``agents/base.py``.
    """
    gt = ground_truth
    if isinstance(gt, int):
        gt = "hateful" if gt == 1 else "not_hateful"
    if isinstance(gt, (list, tuple, set)):
        gt_set = {str(x).lower().strip() for x in gt if str(x).strip()}
        pred_set = {p.strip().lower() for p in str(pred_label).split(",") if p.strip()}
        return bool(gt_set) and gt_set == pred_set
    gt = str(gt).lower().strip()
    pl = pred_label.lower().strip()
    return pl == gt


def majority_vote(label_weight_pairs: list[tuple[str, float]]) -> tuple[str, float]:
    """Weighted majority vote. Returns (label, total_weight_for_winner)."""
    from collections import defaultdict
    totals: dict[str, float] = defaultdict(float)
    for label, weight in label_weight_pairs:
        totals[label] += weight
    if not totals:
        return "unknown", 0.0
    winner = max(totals, key=totals.get)
    total = sum(totals.values())
    conf = totals[winner] / total if total > 0 else 0.5
    return winner, conf
