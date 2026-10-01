"""Math interventions and a numeric-answer parser for the VOUCH math
reasoning experiments.

Interventions modify the problem statement:
  - perturb_number(i, delta): replace the i-th numeric token by + delta
  - remove_clause(sentence_idx): drop one sentence from the problem
  - add_distractor: insert an irrelevant-but-plausible clause
  - paraphrase: LLM-style paraphrase preserving the math (deterministic
    string-level edit so we keep determinism without calling a model)
  - swap_problem: replace with an unrelated benign problem

The verification operator parses the agent's final numeric answer from
its chain-of-thought and checks it against the committed counterfactual
answer with an absolute tolerance of 1e-3 (problems use small integers
or simple fractions so this is effectively exact match).
"""
from __future__ import annotations

import re
from typing import Iterable, Optional


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Match the agent's final numeric answer. Prefer "#### X" style (GSM8K
# convention) but fall back to last numeric token in the output.
_FINAL_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def list_numbers(text: str) -> list[tuple[int, str]]:
    """Return a list of (start_index, token) for every numeric token
    in the problem statement."""
    return [(m.start(), m.group(0)) for m in _NUM_RE.finditer(text)]


# -------- interventions ----------

def perturb_number(text: str, i: int, delta: float) -> tuple[str, Optional[str]]:
    """Replace the i-th numeric token in `text` with itself + delta.

    Returns (modified_text, predicted_relation). The predicted_relation
    string describes how the gold answer should change ("+delta",
    "scale", etc.) but is *not* the new answer; the agent commits to
    that themselves.
    """
    nums = list_numbers(text)
    if not (0 <= i < len(nums)):
        return text, None
    start, tok = nums[i]
    try:
        original = float(tok)
    except ValueError:
        return text, None
    new = original + delta
    new_tok = str(int(new)) if new.is_integer() else str(new)
    end = start + len(tok)
    return text[:start] + new_tok + text[end:], f"perturb_number({i},{delta})"


def remove_clause(text: str, sentence_idx: int) -> tuple[str, Optional[str]]:
    """Drop the i-th sentence (split on '.', '!', '?')."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if not (0 <= sentence_idx < len(parts)):
        return text, None
    new_text = " ".join(parts[:sentence_idx] + parts[sentence_idx + 1:])
    return new_text, f"remove_clause({sentence_idx})"


def add_distractor(text: str, distractor: str = "Note: the sky was blue that day.") -> tuple[str, str]:
    """Insert a clearly-irrelevant clause near the start of the problem."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    if len(parts) == 2:
        new_text = parts[0] + " " + distractor + " " + parts[1]
    else:
        new_text = distractor + " " + text
    return new_text, "add_distractor"


def swap_problem(text: str, replacement: str = "What is 2 plus 3?") -> tuple[str, str]:
    """Replace the problem entirely with a benign question."""
    return replacement, "swap_problem"


def paraphrase(text: str) -> tuple[str, str]:
    """Deterministic string-level paraphrase that preserves numbers.

    We rewrite common surface forms ("how many", "what is") so the
    problem looks different but the math is unchanged. This is the
    `paraphrase' intervention in our menu; a richer LLM-based rewrite
    is straightforward but introduces nondeterminism.
    """
    rules = [
        (r"\bHow many\b", "What is the number of"),
        (r"\bhow many\b", "what is the number of"),
        (r"\bWhat is\b",  "Find the value of"),
        (r"\bwhat is\b",  "find the value of"),
        (r"\bIf\b",       "Suppose"),
        (r"\bif\b",       "suppose"),
    ]
    out = text
    for pat, repl in rules:
        out = re.sub(pat, repl, out, count=1)
    return out, "paraphrase"


# -------- answer parser ----------

def parse_final_answer(generation: str) -> Optional[str]:
    """Extract the agent's final numeric answer from a CoT generation.

    Returns the answer as a normalized string ("16", "3.5", "-4"); None
    if no numeric answer can be found.
    """
    m = _FINAL_RE.search(generation)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: take the last numeric token in the generation.
    nums = _NUM_RE.findall(generation)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def answers_match(a: Optional[str], b: Optional[str], tol: float = 1e-3) -> bool:
    """Compare two numeric-answer strings with absolute tolerance."""
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except ValueError:
        return a.strip().lower() == b.strip().lower()


# -------- intervention menu ----------

INTERVENTION_NAMES: list[str] = [
    "perturb_number_+1",
    "perturb_number_*10",
    "remove_clause_1",
    "add_distractor",
    "paraphrase",
    "swap_problem",
]


def apply_intervention(text: str, name: str) -> tuple[str, Optional[str]]:
    """Apply a named intervention to the problem text. Returns
    (modified_text, intervention_tag). The tag is non-None iff the
    intervention applied successfully."""
    if name == "perturb_number_+1":
        return perturb_number(text, 0, 1.0)
    if name == "perturb_number_*10":
        nums = list_numbers(text)
        if not nums:
            return text, None
        try:
            base = float(nums[0][1])
        except ValueError:
            return text, None
        return perturb_number(text, 0, base * 9.0)
    if name == "remove_clause_1":
        return remove_clause(text, 1)
    if name == "add_distractor":
        return add_distractor(text)
    if name == "paraphrase":
        return paraphrase(text)
    if name == "swap_problem":
        return swap_problem(text)
    return text, None
