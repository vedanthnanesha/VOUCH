"""CrisisMMD loader.

CrisisMMD is not on HuggingFace in a clean form. This module expects you to
have downloaded and extracted the official archive into:

    data/crisismmd/raw/CrisisMMD_v2.0/

See scripts/download_data.py --dataset crisismmd for the wget command.

Task used: humanitarian categorization (text+image agreement subset).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import csv
import random

from .base import Example, write_manifest, read_manifest

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "crisismmd"
RAW_DIR = DATA_DIR / "raw" / "CrisisMMD_v2.0"
MANIFEST = DATA_DIR / "manifest.jsonl"
ANNOT_DIR = RAW_DIR / "annotations"

# Standard 4-class humanitarian label mapping
LABEL_MAP = {
    "infrastructure_and_utility_damage": "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort": "rescue_volunteering_or_donation_effort",
    "affected_individuals": "affected_individuals",
    "injured_or_dead_people": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "not_humanitarian": "not_humanitarian",
    "other_relevant_information": "not_humanitarian",
    "vehicle_damage": "infrastructure_and_utility_damage",
    "not_informative": "not_humanitarian",
}


def build(n: Optional[int] = 500, seed: int = 0) -> list[Example]:
    if not ANNOT_DIR.exists():
        raise FileNotFoundError(
            f"CrisisMMD annotations not found at {ANNOT_DIR}. "
            "Run: python scripts/download_data.py --dataset crisismmd"
        )

    # Read all per-event TSV files
    rows = []
    for tsv in sorted(ANNOT_DIR.glob("*_final_data.tsv")):
        with tsv.open(encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                rows.append(r)
    print(f"[crisismmd] read {len(rows)} rows from {len(list(ANNOT_DIR.glob('*_final_data.tsv')))} event files")

    # Filter for text_human == image_human agreement, map to 4-class schema
    agreed = []
    for r in rows:
        text_label = r.get("text_human", "").strip()
        image_label = r.get("image_human", "").strip()
        if text_label and image_label and text_label == image_label:
            mapped = LABEL_MAP.get(text_label)
            if mapped:
                r["_mapped_label"] = mapped
                agreed.append(r)
    print(f"[crisismmd] {len(agreed)} rows with text-image humanitarian agreement")

    rng = random.Random(seed)
    rng.shuffle(agreed)

    # Stratified sampling for balanced classes
    if n is not None:
        from collections import defaultdict
        by_label = defaultdict(list)
        for r in agreed:
            by_label[r["_mapped_label"]].append(r)
        per_class = n // len(by_label)
        selected, overflow = [], []
        for label, items in by_label.items():
            rng.shuffle(items)
            selected.extend(items[:per_class])
            overflow.extend(items[per_class:])
        rng.shuffle(overflow)
        remaining = n - len(selected)
        if remaining > 0:
            selected.extend(overflow[:remaining])
        rng.shuffle(selected)
        agreed = selected[:n]

    examples: list[Example] = []
    n_missing = 0
    for r in agreed:
        ex_id = f"{r.get('tweet_id', '')}_{r.get('image_id', '')}"
        img_rel = r.get("image_path", "").strip()
        if img_rel:
            img_path = RAW_DIR / img_rel
            if not img_path.exists():
                n_missing += 1
                continue
            img_path = str(img_path)
        else:
            img_path = None
        examples.append(
            Example(
                id=ex_id,
                text=r.get("tweet_text"),
                image_path=img_path,
                label=r["_mapped_label"],
                meta={"source": "crisismmd", "task": "humanitarian"},
            )
        )

    if n_missing:
        print(f"[crisismmd] WARN: {n_missing} examples skipped (missing images)")

    write_manifest(MANIFEST, examples)
    print(f"[crisismmd] wrote manifest: {MANIFEST} ({len(examples)} examples)")
    return examples


def load(n: Optional[int] = None) -> list[Example]:
    if not MANIFEST.exists():
        return build(n=n)
    examples = read_manifest(MANIFEST)
    return examples[:n] if n else examples
