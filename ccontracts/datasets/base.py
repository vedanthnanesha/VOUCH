"""Common dataset types."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
import json


@dataclass
class Example:
    id: str
    text: Optional[str]          # caption / tweet / plot summary
    image_path: Optional[str]    # local path to image, if any
    label: Any                   # int, str, or list[str] depending on dataset
    meta: dict                   # extra fields (split, source, etc.)


def write_manifest(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def read_manifest(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            examples.append(Example(**d))
    return examples
