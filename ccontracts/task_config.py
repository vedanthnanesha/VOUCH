"""Task configurations — dataset-specific prompts and label sets.

Each TaskConfig now also declares:
  - ``benign_predicate``: callable(label) -> bool indicating which examples
    are eligible as benign swap candidates (used by the swap_caption and
    swap_image interventions).
  - ``neutralization_target``: a short noun phrase describing the kind of
    text the dataset's adversary trades on (e.g. "hateful or offensive
    language", "infrastructure-damage vocabulary"). The LLM-based
    interventions (neutralize_text, reverse_sentiment) inject this so the
    rewrite prompt is task-appropriate instead of Hateful-Memes-flavoured.
  - ``multilabel``: True for multi-label tasks (e.g. MM-IMDb genres).
  - ``intervention_ranking`` (optional): precomputed ordering of menu
    interventions by their estimated informativeness I(iota), used by
    VOUCH-Adaptive. Populated offline; left empty by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


def _binary_zero_is_benign(label) -> bool:
    return label == 0 or label == "0" or label == "not_hateful"


def _crisismmd_benign(label) -> bool:
    return label == "not_humanitarian"


def _mm_imdb_benign(label) -> bool:
    # Any label that does NOT contain a violence-coded genre.
    # Multi-label labels arrive as lists or comma-joined strings.
    violent = {"Horror", "Thriller", "Crime", "Action"}
    if isinstance(label, (list, tuple, set)):
        return not (set(label) & violent)
    if isinstance(label, str):
        parts = {s.strip() for s in label.split(",")}
        return not (parts & violent)
    return False


@dataclass
class TaskConfig:
    name: str
    task_description: str
    label_options: str
    labels: list[str]
    output_format: str  # legacy; .fmt below is used at runtime
    benign_predicate: Callable[[object], bool] = field(
        default_factory=lambda: _binary_zero_is_benign)
    neutralization_target: str = "emotionally charged, offensive, or hateful language"
    multilabel: bool = False
    intervention_ranking: Optional[list[str]] = None

    @property
    def fmt(self) -> str:
        if self.multilabel:
            return (
                "Output ONLY this JSON, nothing else:\n"
                f'{{"labels": [<one or more of {self.label_options}>], '
                f'"confidence": <float 0.0-1.0>, '
                f'"reasoning": "<25 words max>"}}'
            )
        return (
            "Output ONLY this JSON, nothing else:\n"
            f'{{"label": {self.label_options}, '
            f'"confidence": <float 0.0-1.0>, '
            f'"reasoning": "<25 words max>"}}'
        )


HATEFUL_MEMES = TaskConfig(
    name="hateful_memes",
    task_description="You are classifying whether a meme is hateful or not hateful.",
    label_options='"hateful" or "not_hateful"',
    labels=["hateful", "not_hateful"],
    output_format="",
    benign_predicate=_binary_zero_is_benign,
    neutralization_target="emotionally charged, offensive, violent, or hateful language",
)

CRISISMMD = TaskConfig(
    name="crisismmd",
    task_description=(
        "You are classifying a social media post (tweet + image) from a disaster event "
        "into one of four humanitarian categories: "
        "infrastructure_and_utility_damage, rescue_volunteering_or_donation_effort, "
        "affected_individuals, or not_humanitarian."
    ),
    label_options=(
        '"infrastructure_and_utility_damage" or "rescue_volunteering_or_donation_effort" '
        'or "affected_individuals" or "not_humanitarian"'
    ),
    labels=[
        "infrastructure_and_utility_damage",
        "rescue_volunteering_or_donation_effort",
        "affected_individuals",
        "not_humanitarian",
    ],
    output_format="",
    benign_predicate=_crisismmd_benign,
    neutralization_target="references to physical damage, infrastructure, casualties, or rescue",
)

MM_IMDB = TaskConfig(
    name="mm_imdb",
    task_description=(
        "You are classifying a movie by its genres based on its poster image and plot "
        "summary. Select one or more genres from the list below."
    ),
    label_options=(
        '"Drama","Comedy","Romance","Thriller","Action","Horror","Documentary",'
        '"Crime","Adventure","Sci-Fi","Family","Fantasy","Mystery","Biography",'
        '"History","Animation","War","Music","Musical","Sport","Western",'
        '"Film-Noir","Short"'
    ),
    labels=[
        "Drama", "Comedy", "Romance", "Thriller", "Action",
        "Horror", "Documentary", "Crime", "Adventure", "Sci-Fi",
        "Family", "Fantasy", "Mystery", "Biography", "History",
        "Animation", "War", "Music", "Musical", "Sport",
        "Western", "Film-Noir", "Short",
    ],
    output_format="",
    benign_predicate=_mm_imdb_benign,
    neutralization_target="violent, criminal, or horror-coded vocabulary",
    multilabel=True,
)


TASK_CONFIGS: dict[str, TaskConfig] = {
    "hateful_memes": HATEFUL_MEMES,
    "crisismmd": CRISISMMD,
    "mm_imdb": MM_IMDB,
}
