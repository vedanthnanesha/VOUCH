"""SVAMP loader for the VOUCH math-reasoning experiments.

SVAMP (Patel et al. 2021) is a benchmark of ~1000 grade-school math
problems explicitly designed to test robustness to surface-level
variations of the question (paraphrase, number change, irrelevant
information). It maps cleanly onto VOUCH's intervention framework
because the dataset itself was constructed by applying systematic
perturbations to existing problems.

Each example has a Body (premise), a Question, and a numeric Answer.
We concatenate Body and Question into a single problem text for the
agent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

from .base import Example, write_manifest, read_manifest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "svamp"
MANIFEST = DATA_DIR / "manifest.jsonl"

HF_SOURCES = [
    ("ChilleD/SVAMP", "test"),
    ("Joey234/svamp", "test"),
]


def build(n: Optional[int] = 200, seed: int = 0) -> list[Example]:
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = None
    repo_used = None
    last_err = None
    for repo, split in HF_SOURCES:
        try:
            ds = load_dataset(repo, split=split)
            repo_used = repo
            print(f"[svamp] loaded {repo}:{split} ({len(ds)} examples)")
            break
        except Exception as e:
            last_err = e
            print(f"[svamp] {repo} unavailable: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError(f"no SVAMP HF mirror available; last error: {last_err}")

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    examples: list[Example] = []
    for i in idxs:
        if n is not None and len(examples) >= n:
            break
        row = ds[i]
        body = row.get("Body") or row.get("body") or ""
        question = row.get("Question") or row.get("question") or ""
        answer = row.get("Answer") or row.get("answer")
        if not question or answer is None:
            continue
        gold = str(answer).strip()
        text = (body.strip() + " " + question.strip()).strip()
        examples.append(Example(
            id=f"svamp_{i}",
            text=text,
            image_path=None,
            label=gold,
            meta={
                "source": repo_used,
                "task": "svamp",
                "body": body,
                "question": question,
            },
        ))

    write_manifest(MANIFEST, examples)
    print(f"[svamp] wrote manifest: {MANIFEST} ({len(examples)} examples)")
    return examples


def load(n: Optional[int] = None) -> list[Example]:
    if not MANIFEST.exists():
        return build(n=n)
    examples = read_manifest(MANIFEST)
    return examples[:n] if n else examples
