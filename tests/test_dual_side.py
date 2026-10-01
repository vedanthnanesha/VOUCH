"""Unit test for dual-side verification logic without invoking models.

Mocks Agent.predict to return controlled labels and checks that
verify_contract correctly marks proposer_verified, acceptor_verified, and
the conjunction `verified` for the four (T/F)x(T/F) cases, plus the
single-side fallback when the contract has no acceptor commitment.

Run with ``python -m pytest tests`` or ``python tests/test_dual_side.py``.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ccontracts.contracts.contract import Contract  # noqa: E402
from ccontracts.contracts.verification import verify_contract  # noqa: E402
from ccontracts.agents.modality import Modality  # noqa: E402


@dataclass
class _Pred:
    label: str
    confidence: float = 0.95
    reasoning: str = ""
    raw: str = ""


class _MockAgent:
    """Returns a fixed label from .predict() regardless of inputs."""
    def __init__(self, agent_id: str, label_to_return: str):
        self.agent_id = agent_id
        self._label = label_to_return

    def predict(self, text=None, image_path=None, seed=0):
        return _Pred(label=self._label)


@dataclass
class _Example:
    text: str = "demo caption"
    image_path: Optional[str] = None


def run_case(prop_cf_label, acc_cf_label,
             prop_predicted="not_hateful", acc_predicted="not_hateful"):
    """Returns (proposer_verified, acceptor_verified, verified)."""
    contract = Contract(
        proposer="prop", acceptor="acc", round=0,
        intervention_type="ablate_text",
        intervention_desc="Remove the caption entirely",
        predicted_outcome=prop_predicted,
        acceptor_predicted_outcome=acc_predicted,
    )
    proposer = _MockAgent("prop", prop_cf_label)
    acceptor = _MockAgent("acc", acc_cf_label)
    out = verify_contract(
        contract, proposer, Modality.MULTIMODAL, _Example(), seed=0,
        acceptor=acceptor, acceptor_modality=Modality.TEXT,
    )
    return out.proposer_verified, out.acceptor_verified, out.verified


# (proposer counterfactual, acceptor counterfactual) -> expected
# (proposer_verified, acceptor_verified, verified); both commitments are
# "not_hateful".
CASES = [
    (("not_hateful", "not_hateful"), (True, True, True)),
    (("not_hateful", "hateful"),     (True, False, False)),
    (("hateful",     "not_hateful"), (False, True, False)),
    (("hateful",     "hateful"),     (False, False, False)),
]


def test_dual_side_truth_table():
    for (prop, acc), expected in CASES:
        assert run_case(prop, acc) == expected, (prop, acc)


def test_single_side_fallback():
    contract = Contract(
        proposer="prop", acceptor="acc", round=0,
        intervention_type="ablate_text",
        intervention_desc="Remove the caption entirely",
        predicted_outcome="not_hateful",
        acceptor_predicted_outcome=None,
    )
    out = verify_contract(contract, _MockAgent("prop", "not_hateful"),
                          Modality.MULTIMODAL, _Example(), seed=0)
    assert out.verified is True
    assert out.acceptor_verified is None


if __name__ == "__main__":
    print("=== Dual-side verify_contract truth table ===")
    print(f"{'proposer_cf':<15}{'acceptor_cf':<15}{'prop_v':>8}{'acc_v':>7}{'verified':>10}")
    for (prop, acc), _ in CASES:
        pv, av, v = run_case(prop, acc)
        print(f"{prop:<15}{acc:<15}{str(pv):>8}{str(av):>7}{str(v):>10}")
    test_dual_side_truth_table()
    test_single_side_fallback()
    print("\nall cases pass")
