"""MATH benchmark loader for the VOUCH math-reasoning experiments.

The MATH dataset (Hendrycks et al. 2021) contains ~12.5K competition
math problems across seven subject areas. We sample n problems from
the test split (default n=100) and expose them in the same Example
shape as GSM8K.

Answers in MATH are stored as LaTeX inside `\\boxed{...}`. We extract
the boxed content as the gold answer string. Verification uses the
same numeric matcher as GSM8K, with a string-equality fallback for
non-numeric expressions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import random
import re

from .base import Example, write_manifest, read_manifest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "math_bench"
MANIFEST = DATA_DIR / "manifest.jsonl"

HF_SOURCES = [
    ("EleutherAI/hendrycks_math", "algebra", "test"),
    ("HuggingFaceH4/MATH-500", None, "test"),
    ("qwedsacf/competition_math", None, "train"),
    ("hendrycks/competition_math", None, "test"),
    ("competition_math", None, "test"),
    ("lighteval/MATH", "all", "test"),
]

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_boxed(solution: str) -> str:
    """Pull the contents of the LAST \\boxed{...} in a MATH solution."""
    matches = BOXED_RE.findall(solution)
    if not matches:
        return ""
    return matches[-1].strip()


def build(n: Optional[int] = 100, seed: int = 0) -> list[Example]:
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = None
    repo_used = None
    last_err = None
    for repo, cfg, split in HF_SOURCES:
        try:
            if cfg:
                ds = load_dataset(repo, cfg, split=split)
            else:
                ds = load_dataset(repo, split=split)
            repo_used = repo
            print(f"[math] loaded {repo}:{cfg}:{split} ({len(ds)} examples)")
            break
        except Exception as e:
            last_err = e
            print(f"[math] {repo} unavailable: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError(f"no MATH HF mirror available; last error: {last_err}")

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    examples: list[Example] = []
    for i in idxs:
        if n is not None and len(examples) >= n:
            break
        row = ds[i]
        problem = row.get("problem") or row.get("question")
        solution = row.get("solution") or row.get("answer") or ""
        level = row.get("level", "")
        subject = row.get("type", "") or row.get("subject", "")
        if not problem:
            continue
        gold = _extract_boxed(solution)
        if not gold:
            continue
        examples.append(Example(
            id=f"math_{i}",
            text=str(problem),
            image_path=None,
            label=gold,
            meta={
                "source": repo_used,
                "task": "math",
                "level": level,
                "subject": subject,
                "full_solution": solution,
            },
        ))

    write_manifest(MANIFEST, examples)
    print(f"[math] wrote manifest: {MANIFEST} ({len(examples)} examples)")
    return examples


def load(n: Optional[int] = None) -> list[Example]:
    if not MANIFEST.exists():
        return build(n=n)
    examples = read_manifest(MANIFEST)
    return examples[:n] if n else examples
