"""HumanEval loader for the VOUCH code-gen experiments.

Same shape as mbpp_codegen: each Example carries the problem statement
plus a test suite that can be executed to compute a per-test pass vector.
HumanEval has 164 problems in its canonical test split. We expose them
all by default; sub-sampling is supported via the n argument.

Each problem provides:
  - prompt: function signature + docstring (this is what the agent sees)
  - canonical_solution: reference implementation (for the loader to
    verify the test setup is well-formed; not shown to the agent).
  - test: a `check(candidate)` function that asserts behavior.
  - entry_point: the function name the agent must define.

Our VOUCH harness expects a "test_list" of assert-style strings, so we
synthesize them by writing a small wrapper that calls check() on the
candidate. The wrapper is appended after the candidate code is exec'd
in the sandbox subprocess.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

from .base import Example, write_manifest, read_manifest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "humaneval"
MANIFEST = DATA_DIR / "manifest.jsonl"

HF_SOURCES = [
    ("openai_humaneval", "test"),
    ("openai/openai_humaneval", "test"),
]


def build(n: Optional[int] = 164, seed: int = 0) -> list[Example]:
    """Download (or reuse cached) HumanEval test split."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = None
    repo_used = None
    last_err = None
    for repo, split in HF_SOURCES:
        try:
            ds = load_dataset(repo, split=split)
            repo_used = repo
            print(f"[humaneval] loaded {repo}:{split} ({len(ds)} examples)")
            break
        except Exception as e:
            last_err = e
            print(f"[humaneval] {repo} unavailable: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError(f"no HumanEval HF mirror available; last error: {last_err}")

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    examples: list[Example] = []
    for i in idxs:
        if n is not None and len(examples) >= n:
            break
        row = ds[i]
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        canonical = row.get("canonical_solution") or ""
        test_block = row.get("test")
        entry_point = row.get("entry_point")
        if not (task_id and prompt and test_block and entry_point):
            continue

        # Synthesize a single assert-style test that invokes check(candidate).
        # The setup code defines `check` (from `test_block`) and then we
        # assert by calling it on the entry_point function.
        wrapper_assert = f"check({entry_point})"
        examples.append(Example(
            id=f"humaneval_{task_id.replace('/', '_')}",
            text=str(prompt),
            image_path=None,
            label="pass_all",
            meta={
                "source": repo_used,
                "task": "humaneval",
                "reference_code": canonical,
                "test_list": [wrapper_assert],
                "test_setup_code": test_block,
                "entry_point": entry_point,
            },
        ))

    write_manifest(MANIFEST, examples)
    print(f"[humaneval] wrote manifest: {MANIFEST} ({len(examples)} examples)")
    return examples


def load(n: Optional[int] = None) -> list[Example]:
    if not MANIFEST.exists():
        return build(n=n)
    examples = read_manifest(MANIFEST)
    return examples[:n] if n else examples
