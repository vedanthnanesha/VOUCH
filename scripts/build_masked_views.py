"""Build caption-masked image views (used for Hateful Memes and MM-IMDb).

Reads data/<dataset>/manifest.jsonl, runs EasyOCR on each image to
detect burned-in text, inpaints those regions, and writes:

    data/<dataset>/images_masked/<id>.png

Then rewrites the manifest with an added `image_masked_path` field so
each example carries:

    image_path         -> original (multimodal view)
    image_masked_path  -> caption-masked (image-only view)
    text               -> caption string (text-only view)

Idempotent: skips images already masked. Re-run anytime after adding
new examples.

Usage:
    pip install easyocr opencv-python-headless
    python scripts/build_masked_views.py --dataset hateful_memes
    python scripts/build_masked_views.py --dataset mm_imdb
    python scripts/build_masked_views.py --confidence 0.3 --pad 5
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ccontracts.datasets.base import Example, read_manifest, write_manifest  # noqa: E402

_DATASETS = {
    "hateful_memes":  REPO_ROOT / "data" / "hateful_memes",
    "mm_imdb":        REPO_ROOT / "data" / "mm_imdb",
    "crisismmd":      REPO_ROOT / "data" / "crisismmd",
}

# Defaults preserve the original script behaviour (HM).
DATA_DIR = _DATASETS["hateful_memes"]
MANIFEST = DATA_DIR / "manifest.jsonl"
MASKED_DIR = DATA_DIR / "images_masked"


def mask_one(image_path: Path, out_path: Path, ocr, confidence: float, pad: int) -> int:
    """Detect text boxes with EasyOCR, then inpaint them out. Returns number of boxes masked."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"failed to read {image_path}")
    h, w = img.shape[:2]

    detections = ocr.readtext(str(image_path))  # list of (bbox, text, conf)

    # Build a binary mask: 255 where text should be removed, 0 elsewhere
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    n_masked = 0
    for bbox, _text, conf in detections:
        if conf < confidence:
            continue
        pts = np.array(bbox, dtype=np.int32)
        x0 = max(int(pts[:, 0].min()) - pad, 0)
        y0 = max(int(pts[:, 1].min()) - pad, 0)
        x1 = min(int(pts[:, 0].max()) + pad, w)
        y1 = min(int(pts[:, 1].max()) + pad, h)
        inpaint_mask[y0:y1, x0:x1] = 255
        n_masked += 1

    if n_masked > 0:
        img = cv2.inpaint(img, inpaint_mask, inpaintRadius=7,
                          flags=cv2.INPAINT_TELEA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return n_masked


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hateful_memes",
                   choices=list(_DATASETS.keys()),
                   help="which dataset's manifest to mask")
    p.add_argument("--confidence", type=float, default=0.3,
                   help="EasyOCR confidence threshold (default 0.3)")
    p.add_argument("--pad", type=int, default=5,
                   help="pixels to pad each bbox before masking (default 5)")
    p.add_argument("--gpu", action="store_true", default=True,
                   help="use GPU for EasyOCR (default True)")
    p.add_argument("--cpu", dest="gpu", action="store_false")
    p.add_argument("--overwrite", action="store_true",
                   help="re-mask images even if output already exists")
    args = p.parse_args()

    data_dir = _DATASETS[args.dataset]
    manifest = data_dir / "manifest.jsonl"
    masked_dir = data_dir / "images_masked"

    if not manifest.exists():
        sys.exit(f"manifest not found: {manifest}\n"
                 f"run: python scripts/download_data.py --dataset {args.dataset}")

    # Rebind module-level paths so the helpers below see them.
    global MANIFEST, MASKED_DIR
    MANIFEST = manifest
    MASKED_DIR = masked_dir

    import easyocr  # heavy import; do it after arg parsing
    print(f"[mask] loading EasyOCR (gpu={args.gpu}) ...")
    ocr = easyocr.Reader(["en"], gpu=args.gpu)

    examples = read_manifest(MANIFEST)
    MASKED_DIR.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skipped = 0
    n_failed = 0
    new_examples: list[Example] = []
    for i, ex in enumerate(examples):
        src = Path(ex.image_path)
        dst = MASKED_DIR / src.name

        if dst.exists() and not args.overwrite:
            n_skipped += 1
        else:
            try:
                n_boxes = mask_one(src, dst, ocr, args.confidence, args.pad)
                n_done += 1
                if (n_done) % 25 == 0:
                    print(f"[mask] {i+1}/{len(examples)} processed "
                          f"(new: {n_done}, skipped: {n_skipped}, failed: {n_failed})")
            except Exception as e:  # noqa: BLE001
                print(f"[mask] WARN failed {src.name}: {e}")
                n_failed += 1
                # still record example with no masked path so manifest stays aligned
                new_examples.append(ex)
                continue

        # update example with masked path
        new_meta = dict(ex.meta)
        new_ex = Example(
            id=ex.id,
            text=ex.text,
            image_path=ex.image_path,
            label=ex.label,
            meta=new_meta,
        )
        # add masked-path as a top-level-ish field via meta to keep
        # the dataclass schema stable AND surface it conveniently.
        new_ex.meta["image_masked_path"] = str(dst)
        new_examples.append(new_ex)

    write_manifest(MANIFEST, new_examples)
    print(f"[mask] DONE. masked={n_done} skipped={n_skipped} failed={n_failed}")
    print(f"[mask] manifest updated: {MANIFEST}")
    print(f"[mask] masked images at: {MASKED_DIR}")


if __name__ == "__main__":
    main()
