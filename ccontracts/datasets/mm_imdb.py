"""MM-IMDb loader (HuggingFace mirror).

Multi-label genre classification; we keep the raw genre list as the label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

from .base import Example, write_manifest, read_manifest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mm_imdb"
IMG_DIR = DATA_DIR / "images"
MANIFEST = DATA_DIR / "manifest.jsonl"

HF_SOURCES = [
    ("Aurorace1/MM-IMDb", "train"),
    ("pranavmr/MM-IMDb", "train"),
]

TEXT_KEYS = ("text", "plot", "plot_outline", "overview", "summary")
LABEL_KEYS = ("labels", "genres", "genre", "label")
IMAGE_KEYS = ("image", "poster", "img")


def _pick(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def download(n: Optional[int] = 200, seed: int = 0) -> list[Example]:
    from datasets import load_dataset

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ds = None
    last_err = None
    for repo, split in HF_SOURCES:
        try:
            ds = load_dataset(repo, split=split)
            print(f"[mm_imdb] loaded {repo}:{split} ({len(ds)} examples)")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[mm_imdb] failed {repo}: {e}")
    if ds is None:
        raise RuntimeError(f"Could not load any MM-IMDb mirror: {last_err}")

    indices = list(range(len(ds)))
    if n is not None and n < len(ds):
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:n]

    examples: list[Example] = []
    for i in indices:
        row = ds[i]
        ex_id = str(row.get("id", i))
        text = _pick(row, TEXT_KEYS)
        label = _pick(row, LABEL_KEYS)
        image = _pick(row, IMAGE_KEYS)

        img_path = None
        if image is not None and hasattr(image, "convert"):
            img_path_p = IMG_DIR / f"{ex_id}.jpg"
            if not img_path_p.exists():
                image.convert("RGB").save(img_path_p, "JPEG")
            img_path = str(img_path_p)

        examples.append(
            Example(
                id=ex_id,
                text=text,
                image_path=img_path,
                label=label,
                meta={"source": "mm_imdb", "split": "test"},
            )
        )

    write_manifest(MANIFEST, examples)
    print(f"[mm_imdb] wrote manifest: {MANIFEST} ({len(examples)} examples)")
    return examples


def load(n: Optional[int] = None) -> list[Example]:
    if not MANIFEST.exists():
        return download(n=n)
    examples = read_manifest(MANIFEST)
    return examples[:n] if n else examples
