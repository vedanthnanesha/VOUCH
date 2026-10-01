"""MBPP (Mostly Basic Python Problems) loader for the VOUCH code-generation
experiments.

This loader exposes a seeded subset of MBPP examples in a format compatible
with the rest of the dataset infra. Each example has:

  - id: MBPP task_id as string
  - text: the problem description
  - image_path: None (no image)
  - label: string "pass_all" if the reference solution passes every test
           in the test_list, else "fail_some". (Trivially "pass_all" for
           well-formed MBPP entries; included so the existing
           label-aggregation code path still works for diagnostic purposes.)
  - meta: keeps the reference code, test_list, and test_setup_code so the
          MBPP-specific evaluator can run candidate code against the tests.

This dataset is used by the code-generation scripts
(experiments/main_eval_codegen*.py), not by the multimodal
classification harness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

from .base import Example, write_manifest, read_manifest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mbpp"
MANIFEST = DATA_DIR / "manifest.jsonl"

HF_SOURCES = [
    ("google-research-datasets/mbpp", "test"),
    ("mbpp", "test"),
]


def build(n: Optional[int] = 30, seed: int = 0) -> list[Example]:
    """Download (or reuse cached) MBPP test split and pick a sanitised
    subset.
    """
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = None
    repo_used = None
    last_err = None
    for repo, split in HF_SOURCES:
        try:
            ds = load_dataset(repo, split=split)
            repo_used = repo
            print(f"[mbpp] loaded {repo}:{split} ({len(ds)} examples)")
            break
        except Exception as e:
            last_err = e
            print(f"[mbpp] {repo} unavailable: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError(f"no MBPP HF mirror available; last error: {last_err}")

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    examples: list[Example] = []
    for i in idxs:
        if len(examples) >= (n or 30):
            break
        row = ds[i]
        task_id = row.get("task_id")
        text = row.get("text") or row.get("prompt")
        code = row.get("code")
        tests = row.get("test_list") or []
        setup = row.get("test_setup_code") or ""
        if not text or not code or not tests:
            continue
        examples.append(Example(
            id=f"mbpp_{task_id}",
            text=str(text),
            image_path=None,
            label="pass_all",  # placeholder; real metric is test_pass_count
            meta={
                "source": repo_used,
                "task": "mbpp",
                "reference_code": code,
                "test_list": list(tests),
                "test_setup_code": setup,
            },
        ))

    write_manifest(MANIFEST, examples)
    print(f"[mbpp] wrote manifest: {MANIFEST} ({len(examples)} examples)")
    return examples


def load(n: Optional[int] = None) -> list[Example]:
    if not MANIFEST.exists():
        return build(n=n)
    examples = read_manifest(MANIFEST)
    return examples[:n] if n else examples
