"""Global registry of persistent inference workers.

A single _PersistentWorker is kept per (worker_type, model_id) and shared across
all Agent instances. This eliminates the cold-start cost of the previous
subprocess-per-call pattern (~12 s per call) and replaces it with warm
inference (~1-3 s per call) while preserving CUDA isolation between unrelated
agents.

If the worker dies for any reason it is respawned transparently on the next
call. Optional recycling via CONCORD_WORKER_RECYCLE=K kills the worker after
every K successful calls (set K=1 to emulate the old per-call subprocess
behaviour as a fallback).
"""
from __future__ import annotations
import os
import sys
import json
import select
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

_WORKER_SCRIPT = Path(__file__).resolve().parent / "_persistent_worker.py"
# Per-worker-type recycle policy.
# Models are heavy: Qwen2.5-VL-7B ~30 GB, Qwen2.5-7B ~15 GB. Cold-start
# costs ~10 s. With recycle=1 we paid that cost on EVERY call. With the
# three drift-prevention fixes shipped today (attn_implementation=eager,
# _reset_model_caches between calls, image-resize to max_dim=448) the
# original Qwen2.5 NaN-token drift is essentially gone, so we can keep
# workers loaded across many calls. drift_retry still catches any rare
# regression by killing+respawning the worker.
#
#   CONCORD_WORKER_RECYCLE_VL=K   -> kill VL worker every K calls
#   CONCORD_WORKER_RECYCLE_TEXT=K -> kill text worker every K calls
#   CONCORD_WORKER_RECYCLE=K      -> global override applied to both types
_DEFAULT_RECYCLE = {"vl": 50, "text": 1}


def _recycle_n(worker_type: str) -> int:
    override = os.environ.get("CONCORD_WORKER_RECYCLE")
    if override is not None and override != "":
        return int(override)
    key = f"CONCORD_WORKER_RECYCLE_{worker_type.upper()}"
    val = os.environ.get(key)
    if val is not None and val != "":
        return int(val)
    return _DEFAULT_RECYCLE.get(worker_type, 0)


_DEBUG = os.environ.get("CONCORD_DEBUG", "0") not in ("0", "", "false", "False")
_FAIL_LOG = Path(__file__).resolve().parents[2] / "results" / "subprocess_failures.log"


def _log_failure(worker_key: str, kind: str, detail: str) -> None:
    try:
        _FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_FAIL_LOG, "a") as f:
            f.write(f"\n[{time.strftime('%H:%M:%S')}] worker={worker_key} kind={kind}\n")
            if detail:
                tail = "\n  ".join(detail.strip().splitlines()[-15:])
                f.write(f"  {tail}\n")
    except Exception:
        pass
    if _DEBUG:
        print(f"[worker {worker_key}] {kind}: {detail[:200]}",
              file=sys.stderr, flush=True)


class _PersistentWorker:
    """One long-lived inference subprocess for a given (type, model_id)."""

    def __init__(self, worker_type: str, model_id: str,
                 dtype: str = "fp32", python: Optional[str] = None):
        self.worker_type = worker_type
        self.model_id = model_id
        self.dtype = dtype
        self.python = python or sys.executable
        self.proc: Optional[subprocess.Popen] = None
        self.n_calls = 0
        self.lock = threading.Lock()
        self.key = f"{worker_type}:{Path(model_id).name}"
        self.recycle_every = _recycle_n(worker_type)

    # -- internals --
    def _spawn(self) -> None:
        env = os.environ.copy()
        env["CONCORD_WORKER_TYPE"] = self.worker_type
        env["CONCORD_MODEL_ID"] = self.model_id
        env["CONCORD_DTYPE"] = self.dtype
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [self.python, str(_WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        # wait for ready
        line = self.proc.stdout.readline()
        if not line:
            err = self._drain_stderr()
            self.proc = None
            raise RuntimeError(f"worker did not signal ready; stderr tail:\n{err[-1000:]}")
        try:
            sig = json.loads(line)
        except Exception as e:
            err = self._drain_stderr()
            self.proc = None
            raise RuntimeError(f"worker first line not JSON: {line!r} ({e}); stderr:\n{err[-500:]}")
        if sig.get("_status") != "ready":
            err = self._drain_stderr()
            self.proc = None
            raise RuntimeError(f"worker error: {sig}; stderr:\n{err[-500:]}")
        self.n_calls = 0
        if _DEBUG:
            print(f"[worker {self.key}] ready", file=sys.stderr, flush=True)

    def _drain_stderr(self) -> str:
        """Best-effort stderr read after process exit."""
        if not self.proc or not self.proc.stderr:
            return ""
        try:
            return self.proc.stderr.read() or ""
        except Exception:
            return ""

    def _kill(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                try:
                    self.proc.stdin.write("__SHUTDOWN__\n")
                    self.proc.stdin.flush()
                except Exception:
                    pass
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                self.proc.kill()
            except Exception:
                pass
        except Exception:
            pass
        self.proc = None

    def _ensure_alive(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()

    @staticmethod
    def _looks_like_drift(out: str) -> bool:
        """Detect NaN-token garbage outputs.

        Conservative: only flag if the output is mostly garbage. Clean prefixes
        followed by drift are still salvageable by the downstream parser
        (which extracts the label from any recognisable substring), so
        retrying them wastes 12 s of cold-start per retry without improving
        accuracy.
        """
        import re as _re
        if not out:
            return True
        # Pure-garbage signature: starts with a run of '!' or other single
        # char that consumes most of the output.
        if _re.match(r"^[\s]*([!?\.])\1{40,}", out):
            return True
        # >60% of the output is a single-character run.
        m = _re.search(r"(.)\1{30,}", out)
        if m and len(m.group(0)) > 0.6 * len(out):
            return True
        # Very short stub like '{"' or '{' (model emitted opening brace then
        # immediately produced EOS or garbage). 8 chars is below the minimum
        # for any valid label JSON.
        if len(out.strip()) < 8:
            return True
        # Word-level repetition: same 3+ char word appearing 5+ times in a row
        # ("Rencontre Rencontre Rencontre ..." pattern).
        if _re.search(r"\b(\w{3,})\s+(\1\s+){4,}", out):
            return True
        # CJK characters in an output that should be English. Qwen2.5
        # occasionally drifts into Chinese tokens on numerical-instability.
        cjk_chars = len(_re.findall(r"[一-鿿]", out))
        if cjk_chars >= 5:
            return True
        # Two-character pattern repetition ("末苍末苍末苍 ...").
        if _re.search(r"(..)\1{10,}", out):
            return True
        return False

    def _readline_timeout(self, timeout: float) -> str:
        """Block-read one newline-terminated response from the worker, honouring
        the wall-clock timeout. Returns '' on timeout or EOF (the caller treats
        '' as a transport failure and kills the worker).

        Plain ``self.proc.stdout.readline()`` does NOT honour the timeout
        argument and will block forever if the worker hangs mid-generation
        (observed: GSM8K SC baseline stuck for 11 h before this fix).
        """
        fd = self.proc.stdout.fileno()
        deadline = time.time() + timeout
        buf = bytearray()
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return ""
            ready, _, _ = select.select([fd], [], [], min(remaining, 5.0))
            if not ready:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return ""
            if not chunk:
                # EOF
                return ""
            buf.extend(chunk)
            if b"\n" in buf:
                line, _ = buf.split(b"\n", 1)
                return line.decode("utf-8", errors="replace") + "\n"

    # -- public --
    def call(self, prompt: str, image_path: Optional[str], timeout: float = 240.0,
              max_retries: int = 1, **gen_kwargs) -> str:
        """Send one request. Returns the model's output, or '' on failure.

        If the worker returns a drift signature (empty / `!!!!` / long char
        run), we kill the worker and retry up to ``max_retries`` times. This
        is the workaround for a CUDA-degraded GPU producing NaN-token output
        intermittently even with recycle=1.

        ``gen_kwargs`` is forwarded verbatim to the worker as extra JSON
        fields (e.g. ``sample=True, temperature=0.7, seed=k,
        max_new_tokens=512``). Workers that do not understand a given key
        ignore it, so this is fully backward-compatible.
        """
        attempt = 0
        last_out = ""
        while attempt <= max_retries:
            attempt += 1
            out = self._call_once(prompt, image_path, timeout, **gen_kwargs)
            if not self._looks_like_drift(out):
                return out
            # drift detected — kill the worker so the retry spawns fresh
            _log_failure(self.key, "drift_retry",
                          f"attempt={attempt} out_head={out[:80]!r}")
            self._kill()
            last_out = out
        return last_out

    def _call_once(self, prompt: str, image_path: Optional[str],
                    timeout: float, **gen_kwargs) -> str:
        with self.lock:
            try:
                self._ensure_alive()
            except Exception as e:
                _log_failure(self.key, "spawn_failed", repr(e))
                return ""

            req = {"id": str(self.n_calls), "prompt": prompt,
                   "image_path": image_path}
            req.update(gen_kwargs)
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                _log_failure(self.key, "stdin_write_failed", repr(e))
                self._kill()
                return ""

            t0 = time.time()
            line = self._readline_timeout(timeout)
            duration = time.time() - t0
            if not line:
                err = self._drain_stderr()
                _log_failure(self.key, "no_response",
                              f"dur={duration:.1f}s (timeout={timeout}s); stderr_tail:\n{err[-800:]}")
                self._kill()
                return ""

            try:
                resp = json.loads(line)
            except Exception as e:
                _log_failure(self.key, "bad_response_json",
                              f"{e}; line={line[:200]!r}")
                return ""

            self.n_calls += 1
            if self.recycle_every and self.n_calls >= self.recycle_every:
                self._kill()

            if not resp.get("ok"):
                _log_failure(self.key, "inference_error",
                              resp.get("error", "<no error message>"))
                return ""
            return resp.get("output", "") or ""


# ---- global registry ----
_REGISTRY: dict[tuple[str, str], _PersistentWorker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_worker(worker_type: str, model_id: str,
               dtype: str = "fp32", python: Optional[str] = None) -> _PersistentWorker:
    key = (worker_type, model_id)
    with _REGISTRY_LOCK:
        w = _REGISTRY.get(key)
        if w is None:
            w = _PersistentWorker(worker_type, model_id, dtype=dtype, python=python)
            _REGISTRY[key] = w
        return w


def shutdown_all() -> None:
    """Stop all workers; safe to call at process exit."""
    with _REGISTRY_LOCK:
        for w in list(_REGISTRY.values()):
            w._kill()
        _REGISTRY.clear()


# best-effort cleanup on interpreter exit
import atexit
atexit.register(shutdown_all)
