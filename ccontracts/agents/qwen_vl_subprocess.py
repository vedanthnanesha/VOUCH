"""Qwen2.5-VL agent using a persistent inference worker.

The old subprocess-per-call pattern (cold-start ~12 s per inference) is
replaced by a single long-running worker shared across all agents that
share the same (worker_type, model_id). Per-call wall-time drops to
~1-3 s with the model already loaded.

Memory hygiene (torch.cuda.empty_cache() + del + gc) is applied after
every request inside the worker; if the Qwen2.5-VL NaN-on-subsequent-
calls bug recurs, set CONCORD_WORKER_RECYCLE=1 to fall back to the old
per-call subprocess behaviour. See ccontracts/agents/_worker_pool.py.

The public API (load_model, _generate_batch) is unchanged so existing
callers in the protocol, baselines, and main_evaluation.py work as-is.
"""
from __future__ import annotations
import sys
from typing import Optional

from .base import Agent
from ._worker_pool import get_worker


class QwenVLSubprocessAgent(Agent):
    """Vision-language agent backed by a persistent worker.

    Routes text-only requests (image_path is None) to a Qwen2.5-7B-Instruct
    text worker and vision-language requests to a Qwen2.5-VL worker. The
    text and VL workers are global singletons keyed on model id, so
    multiple agent instances (e.g. qwen_image and qwen_multimodal) share
    them.
    """

    def __init__(self, agent_id: str = "qwen_vl",
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 text_model: str = "Qwen/Qwen2.5-7B-Instruct",
                 dtype: str = "fp32"):
        super().__init__(model_id=model_name, agent_id=agent_id)
        self.text_model = text_model
        self.dtype = dtype
        # Worker subprocesses run under the same interpreter as the caller.
        self._python = sys.executable

    def load_model(self) -> None:
        # Lazy: workers spawn on first call. Eagerly print a status line
        # for parity with the old API.
        print(f"[{self.agent_id}] persistent-worker mode "
              f"(VL={self.model_id}, text={self.text_model})")

    def _generate_batch(self, prompts: list[str],
                        image_paths: list[Optional[str]],
                        seed: int, temperature: float) -> list[str]:
        results: list[str] = []
        for prompt, img_path in zip(prompts, image_paths):
            if img_path is None or img_path == "NONE":
                worker = get_worker("text", self.text_model,
                                     dtype=self.dtype, python=self._python)
                out = worker.call(prompt, image_path=None)
            else:
                worker = get_worker("vl", self.model_id,
                                     dtype=self.dtype, python=self._python)
                out = worker.call(prompt, image_path=img_path)
            results.append(out)
        return results
