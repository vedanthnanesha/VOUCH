"""Text-only agent backed by a persistent inference worker.

Shares a worker with QwenVLSubprocessAgent's text path when the underlying
model id matches (Qwen2.5-7B-Instruct by default). See
ccontracts/agents/_worker_pool.py for the worker registry.
"""
from __future__ import annotations
import sys
from typing import Optional

from .base import Agent
from ._worker_pool import get_worker


class LlamaTextSubprocessAgent(Agent):
    """Text-only agent backed by a persistent worker."""

    def __init__(self, agent_id: str = "llama_text",
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 dtype: str = "fp32"):
        super().__init__(model_id=model_name, agent_id=agent_id)
        self.dtype = dtype
        # Worker subprocesses run under the same interpreter as the caller.
        self._python = sys.executable

    def load_model(self) -> None:
        print(f"[{self.agent_id}] persistent-worker mode ({self.model_id})")

    def _generate_batch(self, prompts: list[str],
                        image_paths: list[Optional[str]],
                        seed: int, temperature: float) -> list[str]:
        results: list[str] = []
        worker = get_worker("text", self.model_id,
                             dtype=self.dtype, python=self._python)
        for prompt, _ in zip(prompts, image_paths):
            out = worker.call(prompt, image_path=None)
            results.append(out)
        return results
