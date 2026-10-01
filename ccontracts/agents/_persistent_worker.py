"""Long-running inference worker for VL or text models.

Reads newline-delimited JSON requests from stdin and writes responses to stdout.

Request:  {"id": str, "prompt": str, "image_path": str | null}
Response: {"id": str, "ok": bool, "output": str, "error": str | null}

Environment variables:
  CONCORD_WORKER_TYPE   "vl" or "text" (default "vl")
  CONCORD_MODEL_ID      HuggingFace model id (required)
  CONCORD_DTYPE         "fp32" | "fp16" | "bf16" (default "fp32")

On startup, the worker writes a single status JSON line to stdout:
  {"_status": "ready", "model_id": ..., "type": ...}

The wrapper Agent class spawns one worker per (model_id, type). Memory hygiene
(torch.cuda.empty_cache() + del + gc) is applied between requests to mitigate
the Qwen2.5-VL NaN-on-subsequent-calls bug; if the bug recurs the wrapper can
opt into per-N-call recycling via the CONCORD_WORKER_RECYCLE env var.
"""
from __future__ import annotations
import os
import sys
import json
import gc
import traceback
import torch
from PIL import Image


def _log(*args):
    print(*args, file=sys.stderr, flush=True)


WORKER_TYPE = os.environ.get("CONCORD_WORKER_TYPE", "vl")
MODEL_ID = os.environ.get("CONCORD_MODEL_ID", "")
DTYPE_NAME = os.environ.get("CONCORD_DTYPE", "fp32")
DTYPE = {"fp32": torch.float32, "fp16": torch.float16,
         "bf16": torch.bfloat16}.get(DTYPE_NAME, torch.float32)

if not MODEL_ID:
    print(json.dumps({"_status": "error", "error": "CONCORD_MODEL_ID not set"}), flush=True)
    sys.exit(1)

# ---------- model load (once) ----------
try:
    if WORKER_TYPE == "vl":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        # Use the model's config-shipped default (eager). Tried sdpa override
        # to fix VL drift but the smoke (8/10 with sdpa vs 85% in live eager)
        # was not a clear win — the real bottleneck is drift detection, not
        # attention impl. Drift_retry catches the recoverable cases at the
        # worker pool level.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, device_map="auto")
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.eval()
        tok = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Use sdpa (transformers default). attn_implementation="eager" was
        # previously set here but it mishandles Qwen2.5's Sliding Window
        # Attention and produces NaN logits at step 0 on any moderately
        # long prompt (e.g., the contract acceptor template). sdpa handles
        # SWA correctly. Verified: 6/6 clean contract-prompt outputs.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, device_map="auto",
            attn_implementation="sdpa")
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.eval()
        proc = tok  # unify symbol
except Exception as e:
    print(json.dumps({"_status": "error",
                       "error": f"model load failed: {type(e).__name__}: {e}"}),
          flush=True)
    sys.exit(2)

print(json.dumps({"_status": "ready", "model_id": MODEL_ID, "type": WORKER_TYPE,
                   "dtype": DTYPE_NAME}), flush=True)


# ---------- request loop ----------

_VL_MAX_DIM = 448  # Qwen2.5-VL drifts to NaN logits on images larger than this


def _resize_for_vl(img):
    """Downsize the image so neither side exceeds _VL_MAX_DIM. Preserves
    aspect ratio. Empirically necessary to keep Qwen2.5-VL out of its NaN-
    logit regime on high-resolution disaster photos (CrisisMMD)."""
    w, h = img.size
    m = max(w, h)
    if m <= _VL_MAX_DIM:
        return img
    scale = _VL_MAX_DIM / float(m)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.LANCZOS)


def run_vl(prompt: str, image_path: str | None) -> str:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if image_path and image_path != "NONE":
        img = Image.open(image_path).convert("RGB")
        img = _resize_for_vl(img)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}
        ]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt",
                      padding=True).to(model.device)
    else:
        msgs = [{"role": "user", "content": prompt}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    out = proc.batch_decode(gen[:, inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)[0]
    del inputs, gen
    return out


def _reset_model_caches(m) -> None:
    """Best-effort clear of any top-level KV-cache state Qwen2.5 may hang on
    the model instance across generate() calls."""
    for attr in ("_cache", "past_key_values", "_static_cache",
                 "_attn_implementation_internal_cache"):
        if hasattr(m, attr):
            try:
                setattr(m, attr, None)
            except Exception:
                pass


def run_text(prompt: str,
             sample: bool = False,
             temperature: float = 1.0,
             top_p: float = 1.0,
             seed: int = 0,
             max_new_tokens: int = 256) -> str:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    _reset_model_caches(model)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(max_new_tokens=int(max_new_tokens),
                       pad_token_id=tok.eos_token_id)
    if sample:
        gen_kwargs.update(do_sample=True,
                           temperature=float(temperature),
                           top_p=float(top_p))
    else:
        gen_kwargs.update(do_sample=False)

    with torch.inference_mode():
        gen = model.generate(**inputs, **gen_kwargs)

    out = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                     skip_special_tokens=True)
    del inputs, gen
    return out


for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    if raw == "__SHUTDOWN__":
        break
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"id": "?", "ok": False,
                           "error": f"bad request json: {e}"}), flush=True)
        continue

    rid = req.get("id", "?")
    prompt = req.get("prompt", "")
    image_path = req.get("image_path")

    try:
        if WORKER_TYPE == "vl":
            out = run_vl(prompt, image_path)
        else:
            out = run_text(
                prompt,
                sample=bool(req.get("sample", False)),
                temperature=float(req.get("temperature", 1.0)),
                top_p=float(req.get("top_p", 1.0)),
                seed=int(req.get("seed", 0)),
                max_new_tokens=int(req.get("max_new_tokens", 256)),
            )
        print(json.dumps({"id": rid, "ok": True, "output": out}),
              flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"[worker {rid}] error: {tb}")
        print(json.dumps({"id": rid, "ok": False,
                           "error": f"{type(e).__name__}: {e}"}), flush=True)
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
