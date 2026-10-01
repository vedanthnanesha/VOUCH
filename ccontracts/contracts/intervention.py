"""Intervention types for counterfactual contracts.

An Intervention describes a modification to an agent's input.

Two modes:
  - **Prompted** (hypothetical): the description is injected into the prompt
    and the model imagines the counterfactual.
  - **Actual**: the input is physically modified before inference.
    ``apply_actual()`` returns modified (text, image_path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class InterventionType(Enum):
    ABLATE_MODALITY = "ablate_modality"
    MASK_IMAGE_REGION = "mask_image_region"
    REPLACE_TEXT = "replace_text"


@dataclass
class Intervention:
    type: InterventionType
    modality: str        # which modality is affected: "text" | "image"
    description: str     # natural-language description for the agent prompt
    params: dict = field(default_factory=dict)


# ---- factory helpers (used in experiments) ----

def ablate_image() -> Intervention:
    return Intervention(
        type=InterventionType.ABLATE_MODALITY,
        modality="image",
        description=(
            "If the image were completely removed and you could only see "
            "the text caption, what would your prediction be?"
        ),
    )


def ablate_text() -> Intervention:
    return Intervention(
        type=InterventionType.ABLATE_MODALITY,
        modality="text",
        description=(
            "If the text caption were completely removed and you could "
            "only see the image, what would your prediction be?"
        ),
    )


def replace_emotional_text() -> Intervention:
    return Intervention(
        type=InterventionType.REPLACE_TEXT,
        modality="text",
        description=(
            "If all emotionally charged or offensive words in the caption "
            "were replaced with neutral alternatives, what would your "
            "prediction be?"
        ),
    )


def blur_faces() -> Intervention:
    return Intervention(
        type=InterventionType.MASK_IMAGE_REGION,
        modality="image",
        description=(
            "If all people and faces in the image were completely blurred "
            "out, what would your prediction be?"
        ),
    )


def reverse_sentiment() -> Intervention:
    return Intervention(
        type=InterventionType.REPLACE_TEXT,
        modality="text",
        description=(
            "If the sentiment/tone of the caption were completely reversed "
            "(positive→negative or negative→positive), what would your "
            "prediction be?"
        ),
    )


def grayscale_image() -> Intervention:
    return Intervention(
        type=InterventionType.MASK_IMAGE_REGION,
        modality="image",
        description=(
            "If the image were converted to grayscale (no color), "
            "what would your prediction be?"
        ),
    )


def swap_caption() -> Intervention:
    return Intervention(
        type=InterventionType.REPLACE_TEXT,
        modality="text",
        description=(
            "If the caption were replaced with a completely different, "
            "benign caption, what would your prediction be?"
        ),
    )


def swap_image() -> Intervention:
    return Intervention(
        type=InterventionType.MASK_IMAGE_REGION,
        modality="image",
        description=(
            "If the image were replaced with a completely different, "
            "benign image, what would your prediction be?"
        ),
    )


# =====================================================================
#  Actual intervention helpers — physically modify inputs
# =====================================================================

def apply_ablate_image(text: Optional[str], image_path: Optional[str]):
    """Remove image, keep text."""
    return text, None


def apply_ablate_text(text: Optional[str], image_path: Optional[str]):
    """Remove text, keep image."""
    return None, image_path


# ---- face blurring (actual) via OpenCV DNN face detector ----

_FACE_NET = None

def _get_face_net():
    """Load OpenCV's built-in DNN face detector (Caffe model, ships with OpenCV)."""
    global _FACE_NET
    if _FACE_NET is None:
        import cv2
        _FACE_NET = cv2.dnn.readNetFromCaffe(
            cv2.data.haarcascades + "../../dnn/deploy.prototxt",
            cv2.data.haarcascades + "../../dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel",
        )
    return _FACE_NET


_FACE_PROTO = None
_FACE_MODEL = None


def _download_dnn_model():
    """Download OpenCV DNN face detector weights if not present."""
    global _FACE_PROTO, _FACE_MODEL
    import urllib.request

    model_dir = Path(__file__).resolve().parents[2] / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    _FACE_PROTO = model_dir / "deploy.prototxt"
    _FACE_MODEL = model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    if not _FACE_PROTO.exists():
        print("[face_blur] downloading DNN face detector prototxt ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
            str(_FACE_PROTO),
        )
    if not _FACE_MODEL.exists():
        print("[face_blur] downloading DNN face detector model (~5MB) ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
            str(_FACE_MODEL),
        )
    return _FACE_PROTO, _FACE_MODEL


def _get_dnn_face_net():
    global _FACE_NET
    if _FACE_NET is None:
        import cv2
        proto, model = _download_dnn_model()
        _FACE_NET = cv2.dnn.readNetFromCaffe(str(proto), str(model))
    return _FACE_NET


def apply_blur_faces(
    text: Optional[str],
    image_path: Optional[str],
    cache_dir: Optional[Path] = None,
    confidence: float = 0.3,
) -> tuple[Optional[str], Optional[str]]:
    """Detect faces with OpenCV DNN SSD detector + Gaussian-blur them.

    Returns (text, path_to_blurred_image).  If no image or no faces
    found, blurs the center 60% as fallback.
    """
    if image_path is None:
        return text, None

    import cv2

    src = Path(image_path)
    if cache_dir is None:
        cache_dir = src.parent.parent / "images_blurred"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / src.name

    if out_path.exists():
        return text, str(out_path)

    img = cv2.imread(str(src))
    if img is None:
        return text, image_path

    h, w = img.shape[:2]
    net = _get_dnn_face_net()

    # Prepare input blob (300x300, mean subtraction)
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    n_faces = 0
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < confidence:
            continue
        # Bounding box in normalized coords
        x0 = int(detections[0, 0, i, 3] * w)
        y0 = int(detections[0, 0, i, 4] * h)
        x1 = int(detections[0, 0, i, 5] * w)
        y1 = int(detections[0, 0, i, 6] * h)
        # Pad by 30%
        bw, bh = x1 - x0, y1 - y0
        pad_x, pad_y = int(bw * 0.3), int(bh * 0.3)
        x0 = max(x0 - pad_x, 0)
        y0 = max(y0 - pad_y, 0)
        x1 = min(x1 + pad_x, w)
        y1 = min(y1 + pad_y, h)
        img[y0:y1, x0:x1] = cv2.GaussianBlur(
            img[y0:y1, x0:x1], (51, 51), 30,
        )
        n_faces += 1

    if n_faces == 0:
        # No faces found; save image unchanged
        pass

    cv2.imwrite(str(out_path), img)
    return text, str(out_path)


# ---- emotional word neutralization via LLM ----

_NEUTRALIZE_CACHE: dict[str, str] = {}

NEUTRALIZE_PROMPT_TEMPLATE = (
    "Rewrite the following caption by replacing ALL {target} with neutral "
    "alternatives. Keep the same sentence structure and topic where possible. "
    "Output ONLY the rewritten caption, nothing else.\n\n"
    "Original: \"{text}\"\n"
    "Neutral:"
)


def _neutralization_target() -> str:
    cfg = get_active_task_config()
    if cfg is None:
        return "emotionally charged, offensive, violent, or hateful language"
    return cfg.neutralization_target


def apply_replace_emotional_text_llm(
    text: Optional[str],
    image_path: Optional[str],
    agent=None,
) -> tuple[Optional[str], Optional[str]]:
    """Use the agent's LLM to neutralize emotional/offensive words.

    Falls back to returning original text if agent is None.
    """
    if text is None:
        return None, image_path

    if text in _NEUTRALIZE_CACHE:
        return _NEUTRALIZE_CACHE[text], image_path

    if agent is None:
        return text, image_path

    prompt = NEUTRALIZE_PROMPT_TEMPLATE.format(
        text=text, target=_neutralization_target())
    # Use the agent's generate to get neutralized text — no image needed.
    raws = agent._generate_batch([prompt], [None], seed=42, temperature=0.1)
    neutral = raws[0].strip().strip('"').strip("'")

    # Basic sanity: if output is empty or absurdly long, keep original.
    if not neutral or len(neutral) > len(text) * 3:
        neutral = text

    _NEUTRALIZE_CACHE[text] = neutral
    return neutral, image_path


# ---- reverse sentiment via LLM ----

_REVERSE_CACHE: dict[str, str] = {}

REVERSE_PROMPT = (
    "Rewrite the following caption by reversing its sentiment / tone while "
    "keeping the same general topic. If the original is positive or cheerful, "
    "make it negative or grim; if it is negative, hostile, or threatening, "
    "make it positive and friendly. Output ONLY the rewritten caption, "
    "nothing else.\n\n"
    "Original: \"{text}\"\n"
    "Reversed:"
)


def apply_reverse_sentiment_llm(
    text: Optional[str],
    image_path: Optional[str],
    agent=None,
) -> tuple[Optional[str], Optional[str]]:
    """Use the agent's LLM to reverse the caption's sentiment."""
    if text is None:
        return None, image_path

    if text in _REVERSE_CACHE:
        return _REVERSE_CACHE[text], image_path

    if agent is None:
        return text, image_path

    prompt = REVERSE_PROMPT.format(text=text)
    raws = agent._generate_batch([prompt], [None], seed=42, temperature=0.1)
    reversed_text = raws[0].strip().strip('"').strip("'")

    if not reversed_text or len(reversed_text) > len(text) * 3:
        reversed_text = text

    _REVERSE_CACHE[text] = reversed_text
    return reversed_text, image_path


# ---- grayscale image ----

def apply_grayscale(
    text: Optional[str],
    image_path: Optional[str],
    cache_dir: Optional[Path] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Convert image to grayscale."""
    if image_path is None:
        return text, None

    import cv2

    src = Path(image_path)
    if cache_dir is None:
        cache_dir = src.parent.parent / "images_grayscale"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / src.name

    if out_path.exists():
        return text, str(out_path)

    img = cv2.imread(str(src))
    if img is None:
        return text, image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Save as 3-channel so VLMs don't complain about single-channel input
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_path), gray_3ch)
    return text, str(out_path)


# ---- swap caption (cross-modal) ----

_ACTIVE_MANIFEST: Optional[Path] = None
_ACTIVE_TASK_CONFIG = None  # TaskConfig instance


def set_active_dataset(dataset_name: str) -> None:
    """Set the active dataset so swap and LLM-based interventions use the
    correct benign-pool predicate and task-aware prompt fragments."""
    global _ACTIVE_MANIFEST, _ACTIVE_TASK_CONFIG
    _ACTIVE_MANIFEST = Path(__file__).resolve().parents[2] / "data" / dataset_name / "manifest.jsonl"
    try:
        from ccontracts.task_config import TASK_CONFIGS
        _ACTIVE_TASK_CONFIG = TASK_CONFIGS.get(dataset_name)
    except Exception:
        _ACTIVE_TASK_CONFIG = None


def get_active_task_config():
    """Return the currently-active TaskConfig (set by set_active_dataset),
    or None if no dataset has been registered yet."""
    return _ACTIVE_TASK_CONFIG


_BENIGN_CAPTIONS: list[str] = []
_BENIGN_IMAGES: list[str] = []
_BENIGN_POOL_SOURCE: Optional[str] = None


def _load_benign_pool(manifest_path: Optional[Path] = None):
    """Load benign examples as swap candidates. Supports multiple datasets."""
    global _BENIGN_CAPTIONS, _BENIGN_IMAGES, _BENIGN_POOL_SOURCE

    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parents[2] / "data" / "hateful_memes" / "manifest.jsonl"

    source_key = str(manifest_path)
    if _BENIGN_CAPTIONS and _BENIGN_POOL_SOURCE == source_key:
        return
    _BENIGN_CAPTIONS.clear()
    _BENIGN_IMAGES.clear()
    _BENIGN_POOL_SOURCE = source_key

    if not manifest_path.exists():
        return

    import json
    cfg = get_active_task_config()
    if cfg is not None:
        benign_predicate = cfg.benign_predicate
    else:
        # Sensible default if no TaskConfig is registered: label==0 or "not_*"
        def benign_predicate(label):
            return (label == 0 or label == "0"
                    or (isinstance(label, str) and label.startswith("not_")))

    with manifest_path.open() as f:
        for line in f:
            ex = json.loads(line)
            label = ex.get("label")
            text = ex.get("text")
            try:
                is_benign = bool(benign_predicate(label))
            except Exception:
                is_benign = False
            if is_benign and text:
                _BENIGN_CAPTIONS.append(text)
                masked = ex.get("meta", {}).get("image_masked_path")
                img = ex.get("image_path")
                if masked:
                    _BENIGN_IMAGES.append(masked)
                elif img:
                    _BENIGN_IMAGES.append(img)


def apply_swap_caption(
    text: Optional[str],
    image_path: Optional[str],
    seed: int = 0,
) -> tuple[Optional[str], Optional[str]]:
    """Replace caption with a random benign one. Use masked image to avoid
    conflicting burned-in text."""
    import random

    _load_benign_pool(manifest_path=_ACTIVE_MANIFEST)
    if not _BENIGN_CAPTIONS:
        return text, image_path

    rng = random.Random(seed)
    new_caption = rng.choice(_BENIGN_CAPTIONS)

    # Use masked image (burned-in text removed) to avoid conflict
    # For non-meme datasets, just use the original image
    masked = None
    if image_path:
        masked_path = Path(image_path).parent.parent / "images_masked" / Path(image_path).name
        if masked_path.exists():
            masked = str(masked_path)
        else:
            masked = image_path

    return new_caption, masked


def apply_swap_image(
    text: Optional[str],
    image_path: Optional[str],
    seed: int = 0,
) -> tuple[Optional[str], Optional[str]]:
    """Replace image with a random benign masked image. Keep original caption."""
    import random

    _load_benign_pool(manifest_path=_ACTIVE_MANIFEST)
    if not _BENIGN_IMAGES:
        return text, image_path

    rng = random.Random(seed)
    new_image = rng.choice(_BENIGN_IMAGES)  # already masked
    return text, new_image
