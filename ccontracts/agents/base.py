"""Agent base class with disk-cache and batched inference."""
from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "inference_cache"


@dataclass
class Prediction:
    label: str        # "hateful", "not_hateful", or "unknown"
    confidence: float
    reasoning: str
    raw: str          # raw model output


class Agent(ABC):
    """Abstract agent with built-in disk cache.

    Subclasses implement ``_generate_batch`` only.
    """

    # ---- default prompt fragments (overridden by set_task_config) ----
    TASK = "You are classifying whether a meme is hateful or not hateful."
    FMT = (
        'Output ONLY this JSON, nothing else:\n'
        '{"label": "hateful" or "not_hateful", '
        '"confidence": <float 0.0-1.0>, '
        '"reasoning": "<25 words max>"}'
    )

    def __init__(self, model_id: str, agent_id: str):
        self.model_id = model_id
        self.agent_id = agent_id
        self.cache_dir = CACHE_DIR / model_id.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._img_hash_cache: dict[str, str] = {}
        self._task_config = None
        self._adversarial_prefix: Optional[str] = None

    def set_task_config(self, task_config) -> None:
        """Set dataset-specific prompt fragments. Call before predict()."""
        self._task_config = task_config
        self.TASK = task_config.task_description
        self.FMT = task_config.fmt
        self._task_labels = task_config.labels
        self._multilabel = bool(getattr(task_config, "multilabel", False))

    # ---- cache helpers ----

    def _img_hash(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if path not in self._img_hash_cache:
            self._img_hash_cache[path] = hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest()[:16]
        return self._img_hash_cache[path]

    def _cache_key(self, prompt: str, img_hash: Optional[str],
                   seed: int, temp: float) -> str:
        blob = json.dumps(
            {"m": self.model_id, "p": prompt, "i": img_hash,
             "s": seed, "t": temp},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _cache_get(self, key: str) -> Optional[Prediction]:
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            return Prediction(**json.loads(p.read_text()))
        return None

    def _cache_put(self, key: str, pred: Prediction) -> None:
        p = self.cache_dir / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(pred)))

    # ---- prompt builders ----

    @staticmethod
    def _input_desc(text: Optional[str], has_image: bool) -> str:
        parts = []
        if has_image:
            parts.append("[An image is shown above.]")
        else:
            parts.append("[No image provided.]")
        if text:
            parts.append(f'Text: "{text}"')
        else:
            parts.append("[No text provided.]")
        return "\n".join(parts)

    def build_predict_prompt(self, text: Optional[str], has_image: bool) -> str:
        return f"{self.TASK}\n{self._input_desc(text, has_image)}\n\n{self.FMT}"

    def build_cf_prompt(self, text: Optional[str], has_image: bool,
                        intervention: str) -> str:
        return (
            f"{self.TASK}\n{self._input_desc(text, has_image)}\n\n"
            f"Now consider this change: {intervention}\n"
            f"With this change applied, what would your prediction be?\n\n"
            f"{self.FMT}"
        )

    # ---- output parser ----

    @staticmethod
    def _match_label(raw_label: str, valid_labels: list[str]) -> str:
        """Fuzzy-match a raw label string against the valid label set."""
        raw_label = raw_label.lower().strip().replace(" ", "_")
        for vl in valid_labels:
            if raw_label == vl.lower():
                return vl
        for vl in sorted(valid_labels, key=len, reverse=True):
            if vl.lower() in raw_label or raw_label in vl.lower():
                return vl
        return "unknown"

    def _parse(self, raw: str) -> Prediction:
        """Task-agnostic parser.

        Strategy:
          1. Try to extract a JSON object and look for the standard "label"
             key (or alias "labels" for multilabel tasks).
          2. If JSON parsing fails, fall back to scanning the raw text for
             any known label string in ``self._task_labels``.

        No hardcoded fallbacks to Hateful-Memes-specific labels. Works for
        binary, multi-class, and multilabel tasks (the latter via the
        ``labels`` key returning a list).
        """
        valid_labels = getattr(self, '_task_labels', None) or ["hateful", "not_hateful"]
        is_multilabel = bool(getattr(self, '_multilabel', False))

        def _coerce_to_known(label_string: str) -> str:
            return self._match_label(label_string, valid_labels)

        # ---- attempt 1: JSON ----
        try:
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if m:
                d = json.loads(m.group())
                conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
                reasoning = str(d.get("reasoning",
                                      d.get("justification", "")))[:200]

                if is_multilabel and "labels" in d:
                    raw_labels = d.get("labels", [])
                    if isinstance(raw_labels, str):
                        raw_labels = [s.strip() for s in raw_labels.split(",")]
                    matched: list[str] = []
                    for rl in raw_labels:
                        ml = _coerce_to_known(str(rl))
                        if ml != "unknown" and ml not in matched:
                            matched.append(ml)
                    label = ",".join(matched) if matched else "unknown"
                    return Prediction(label=label, confidence=conf,
                                      reasoning=reasoning, raw=raw)

                raw_label = str(d.get("label", d.get("answer", ""))).strip()
                if raw_label:
                    label = _coerce_to_known(raw_label)
                else:
                    label = "unknown"
                    for key in ("classification", "class", "category",
                                "prediction", "sentiment"):
                        val = str(d.get(key, "")).strip()
                        if val:
                            label = _coerce_to_known(val)
                            if label != "unknown":
                                break
                return Prediction(label=label, confidence=conf,
                                  reasoning=reasoning, raw=raw)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            pass

        # ---- attempt 2: scan raw text for any known label ----
        # Use longest-first to avoid prefix overlap (e.g. "not_hateful" vs
        # "hateful"). Normalize underscores and spaces in both directions.
        low = raw.lower().replace("_", " ")
        for vl in sorted(valid_labels, key=len, reverse=True):
            needle = vl.lower().replace("_", " ")
            if needle in low:
                return Prediction(label=vl, confidence=0.5,
                                  reasoning="parse_heuristic", raw=raw)
        return Prediction(label="unknown", confidence=0.5,
                          reasoning="parse_failed", raw=raw)

    # ---- public API ----

    def predict(self, text: Optional[str] = None,
                image_path: Optional[str] = None,
                seed: int = 0, temperature: float = 0.7) -> Prediction:
        return self.predict_batch(
            [{"text": text, "image_path": image_path}],
            seed=seed, temperature=temperature,
        )[0]

    def counterfactual_predict(
        self, text: Optional[str] = None,
        image_path: Optional[str] = None,
        intervention: str = "",
        seed: int = 0, temperature: float = 0.7,
    ) -> Prediction:
        return self.predict_batch(
            [{"text": text, "image_path": image_path,
              "intervention": intervention}],
            seed=seed, temperature=temperature,
        )[0]

    def predict_batch(self, items: list[dict], seed: int = 0,
                      temperature: float = 0.7) -> list[Prediction]:
        """Batched predict/counterfactual_predict.

        Each *item* is a dict with keys ``text``, ``image_path``, and
        optionally ``intervention`` (presence triggers CF prompt).
        """
        results: list[Optional[Prediction]] = [None] * len(items)
        # (original_index, prompt, image_path, cache_key)
        uncached: list[tuple[int, str, Optional[str], str]] = []

        for i, item in enumerate(items):
            has_img = item.get("image_path") is not None
            intv = item.get("intervention")
            if intv:
                prompt = self.build_cf_prompt(item.get("text"), has_img, intv)
            else:
                prompt = self.build_predict_prompt(item.get("text"), has_img)

            # Prepend adversarial prefix if set (affects cache key too)
            if self._adversarial_prefix:
                prompt = self._adversarial_prefix + "\n\n" + prompt

            key = self._cache_key(
                prompt, self._img_hash(item.get("image_path")),
                seed, temperature,
            )
            cached = self._cache_get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached.append((i, prompt, item.get("image_path"), key))

        if uncached:
            raws = self._generate_batch(
                [u[1] for u in uncached],
                [u[2] for u in uncached],
                seed, temperature,
            )
            preds: list[Prediction] = []
            for raw in raws:
                preds.append(self._parse(raw))

            # Retry any unknown predictions individually with different seeds.
            # The model occasionally emits JSON without a "label" key on
            # adversarial prompts; varying the seed usually breaks the
            # pattern on the next try.
            MAX_UNKNOWN_RETRIES = 3
            for j, (idx, prompt, img_path, key) in enumerate(uncached):
                pred = preds[j]
                attempts = 0
                while (pred.label == "unknown" or
                       pred.reasoning == "parse_failed") and attempts < MAX_UNKNOWN_RETRIES:
                    attempts += 1
                    new_raw = self._generate_batch(
                        [prompt], [img_path],
                        seed + 1000 * attempts, temperature,
                    )[0]
                    pred = self._parse(new_raw)
                # If still unknown after retries, fall back to first valid
                # label found in the dataset (modal/safe default) so the
                # vote isn't poisoned by abstention.
                if pred.label == "unknown" or pred.reasoning == "parse_failed":
                    valid = getattr(self, "_task_labels", None) or ["hateful", "not_hateful"]
                    is_multilabel = bool(getattr(self, "_multilabel", False))
                    fallback = valid[-1] if not is_multilabel else valid[0]
                    pred = Prediction(label=fallback, confidence=0.5,
                                       reasoning="forced_fallback_after_retry",
                                       raw=pred.raw)
                preds[j] = pred
                self._cache_put(key, pred)
                results[idx] = pred

        return results  # type: ignore[return-value]

    # ---- subclass contract ----

    @abstractmethod
    def _generate_batch(
        self,
        prompts: list[str],
        image_paths: list[Optional[str]],
        seed: int,
        temperature: float,
    ) -> list[str]:
        """Return one raw string per prompt."""
        ...

    def load_model(self) -> None:
        """Load weights onto GPU. Call once before predict."""
