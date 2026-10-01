"""Hateful Memes loader (HuggingFace mirror).

The HF mirrors ship JSONL metadata alongside per-image PNG files in the same
dataset repo. The `img` field in metadata is a relative path like
`img/08291.png` — we fetch each image individually with hf_hub_download.

Caches images to data/hateful_memes/images/ and writes a JSONL manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import random
import shutil

from .base import Example, write_manifest, read_manifest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "hateful_memes"
IMG_DIR = DATA_DIR / "images"
MANIFEST = DATA_DIR / "manifest.jsonl"

# (repo_id, split). Tried in order. Both repos have the same schema:
#   columns: id, img (relative path), label, text
# and ship per-image PNGs as separate files in the same dataset repo.
HF_SOURCES = [
    ("neuralcatcher/hateful_memes", "validation"),
    ("limjiayi/hateful_memes_expanded", "validation"),
]


def download(n: Optional[int] = 500, seed: int = 0) -> list[Example]:
    """Download (or reuse cached) Hateful Memes validation split.

    Pulls metadata via `datasets`, then per-image PNGs via `hf_hub_download`.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ds = None
    repo_used = None
    last_err = None
    for repo, split in HF_SOURCES:
        try:
            ds = load_dataset(repo, split=split)
            repo_used = repo
            print(f"[hateful_memes] loaded {repo}:{split} ({len(ds)} examples)")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[hateful_memes] failed {repo}: {e}")
    if ds is None:
        raise RuntimeError(f"Could not load any Hateful Memes mirror: {last_err}")

    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    # The HF mirror has duplicate rows (1040 rows, ~640 unique ids).
    # Dedupe by id while iterating, and keep going until we have `n`
    # unique examples (or run out).
    seen_ids: set[str] = set()
    examples: list[Example] = []
    n_fetched = 0
    n_failed = 0
    target = n if n is not None else len(ds)

    for j, i in enumerate(indices):
        if len(examples) >= target:
            break
        row = ds[i]
        ex_id = str(row.get("id", i))
        if ex_id in seen_ids:
            continue
        img_rel = row["img"]                       # e.g. "img/08291.png"
        local_img = IMG_DIR / Path(img_rel).name   # data/hateful_memes/images/08291.png

        if not local_img.exists():
            # Try every mirror in order until one has the file.
            cached = None
            errs = []
            for repo, _ in HF_SOURCES:
                try:
                    cached = hf_hub_download(
                        repo_id=repo,
                        filename=img_rel,
                        repo_type="dataset",
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{repo}: {type(e).__name__}")
            if cached is None:
                print(f"[hateful_memes] WARN failed image {img_rel}: {'; '.join(errs)}")
                n_failed += 1
                continue
            # Copy out of the HF cache into our images/ folder so we
            # control the path and can later free the HF cache.
            shutil.copy(cached, local_img)
            n_fetched += 1

        examples.append(
            Example(
                id=ex_id,
                text=row.get("text"),
                image_path=str(local_img),
                label=int(row["label"]),
                meta={"source": repo_used, "split": "validation"},
            )
        )
        seen_ids.add(ex_id)

        if len(examples) % 50 == 0:
            print(f"[hateful_memes] {len(examples)}/{target} unique examples "
                  f"(scanned {j+1}, new fetches: {n_fetched}, failed: {n_failed})")

    if len(examples) < target:
        print(f"[hateful_memes] WARN only got {len(examples)} unique examples "
              f"out of requested {target} (dataset has {len(set(str(r['id']) for r in ds))} unique ids)")

    write_manifest(MANIFEST, examples)
    print(f"[hateful_memes] wrote manifest: {MANIFEST} "
          f"({len(examples)} examples, {n_fetched} new images, {n_failed} failed)")
    return examples


def load(n: Optional[int] = None) -> list[Example]:
    """Load cached examples; download if manifest missing."""
    if not MANIFEST.exists():
        return download(n=n)
    examples = read_manifest(MANIFEST)
    return examples[:n] if n else examples
