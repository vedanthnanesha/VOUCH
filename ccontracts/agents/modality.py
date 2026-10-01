"""Modality configuration — maps agents to their views and available interventions."""
from __future__ import annotations

from enum import Enum
from typing import Optional


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


# Available interventions per modality.
# An agent can only propose interventions on modalities it has access to.
INTERVENTIONS_BY_MODALITY: dict[Modality, list[str]] = {
    Modality.TEXT: [
        "ablate_text",
        "neutralize_text",
        "reverse_sentiment",
        "swap_caption",
    ],
    Modality.IMAGE: [
        "ablate_image",
        "blur_faces",
        "grayscale",
        "swap_image",
    ],
    Modality.MULTIMODAL: [
        "ablate_image",
        "ablate_text",
        "neutralize_text",
        "reverse_sentiment",
        "blur_faces",
        "grayscale",
        "swap_caption",
        "swap_image",
    ],
}


def get_view(example, modality: Modality) -> dict:
    """Return the input dict an agent with this modality should see."""
    if modality == Modality.TEXT:
        return {"text": example.text, "image_path": None}
    elif modality == Modality.IMAGE:
        return {
            "text": "[caption hidden]",
            "image_path": example.meta.get("image_masked_path", example.image_path),
        }
    elif modality == Modality.MULTIMODAL:
        return {"text": example.text, "image_path": example.image_path}
    raise ValueError(f"unknown modality: {modality}")


def get_available_interventions(modality: Modality) -> list[str]:
    """Return intervention names this modality can propose."""
    return INTERVENTIONS_BY_MODALITY[modality]
