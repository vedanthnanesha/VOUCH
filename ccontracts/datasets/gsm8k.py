"""GSM8K loader for the VOUCH math-reasoning experiments.

Each example carries the natural-language problem and the canonical
numeric answer. The verification operator for math is a deterministic
numeric-answer parse: we extract the final number from the agent's
chain-of-thought and compare it (with a small numeric tolerance) to
the committed answer.

The label is stored as a string of the numeric ground truth so the
existing label-aggregation code path can still produce per-condition
accuracy numbers when run as a multi-condition VOUCH evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import random
import re

from .base import Example, write_manifest, read_manifest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "gsm8k"
MANIFEST = DATA_DIR / "manifest.jsonl"

HF_SOURCES = [
    ("gsm8k", "main", "test"),
    ("openai/gsm8k", "main", "test"),
]

# GSM8K answers are at the end of the solution, prefixed with #### .
ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def _parse_answer(answer_field: str) -> str:
    m = ANSWER_RE.search(answer_field)
    if not m:
        return ""
    return m.group(1).replace(",", "")


def build(n: Optional[int] = 200, seed: int = 0) -> list[Example]:
    """Download GSM8K test split, sample n problems."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = None
    repo_used = None
    last_err = None
    for repo, cfg, split in HF_SOURCES:
        try:
            ds = load_dataset(repo, cfg, split=split)
            repo_used = repo
            print(f"[gsm8k] loaded {repo}:{cfg}:{split} ({len(ds)} examples)")
            break
        except Exception as e:
            last_err = e
            print(f"[gsm8k] {repo} unavailable: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError(f"no GSM8K HF mirror available; last error: {last_err}")

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    examples: list[Example] = []
    for i in idxs:
        if n is not None and len(examples) >= n:
            break
        row = ds[i]
        question = row.get("question")
        full_solution = row.get("answer")
        if not question or not full_solution:
            continue
        gold = _parse_answer(full_solution)
        if not gold:
            continue
        examples.append(Example(
            id=f"gsm8k_{i}",
            text=str(question),
            image_path=None,
            label=gold,
            meta={
                "source": repo_used,
                "task": "gsm8k",
                "full_solution": full_solution,
            },
        ))

    write_manifest(MANIFEST, examples)
    print(f"[gsm8k] wrote manifest: {MANIFEST} ({len(examples)} examples)")
    return examples


def load(n: Optional[int] = None) -> list[Example]:
    if not MANIFEST.exists():
        return build(n=n)
    examples = read_manifest(MANIFEST)
    return examples[:n] if n else examples
