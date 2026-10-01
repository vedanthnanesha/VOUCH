"""Verification cost analysis.

Reports per-condition cost-per-correct-decision in two units:

  - inferences:  total subprocess inference calls
  - tokens_out:  output token count (proxy from len(raw output))

Computed per (dataset, regime, condition) from the JSONL ProtocolResult
records. ``contracts*`` conditions include the contract-negotiation calls
(propose + accept + verify), centralized-coordinator includes the
coordinator call, etc.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _count_calls_per_example(rec: dict) -> int:
    """Best-effort inference count: 1 per agent initial pred + per
    contract: 1 propose + 1 accept (if reached verification, +1 for the
    re-prediction) + per deliberation log entry."""
    n = len(rec.get("initial_preds") or {})
    contracts = rec.get("contracts") or []
    for c in contracts:
        # propose + accept
        n += 2
        # verify implies re-prediction (single call). Approximate.
        if c.get("verified") in (True, False):
            n += 1
    deli = rec.get("deliberation_log") or []
    if deli:
        # round-1 entries reuse initial_preds; only round>=2 are new calls
        n += sum(1 for e in deli if e.get("round", 1) >= 2)
    coord = (rec.get("meta") or {}).get("coordinator_reasoning")
    if coord:
        n += 1
    return n


def _tokens_per_example(rec: dict) -> int:
    """Sum of raw output lengths across all model calls captured in the
    record (rough token-count proxy: 1 token ≈ 4 chars)."""
    chars = 0
    for v in (rec.get("initial_preds") or {}).values():
        chars += len(str(v.get("reasoning") or ""))
    for c in rec.get("contracts") or []:
        chars += len(str(c.get("my_reasoning") or ""))
        chars += len(str(c.get("acceptance_reasoning") or ""))
    coord = (rec.get("meta") or {}).get("coordinator_reasoning") or ""
    chars += len(str(coord))
    deli = rec.get("deliberation_log") or []
    for e in deli:
        chars += len(str(e.get("reasoning") or ""))
    return max(1, chars // 4)


def cost_table(jsonl: Path) -> dict[str, dict]:
    """Per-condition cost summary: total / per-example / per-correct."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    with Path(jsonl).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_cond[r.get("condition", "?")].append(r)

    out: dict[str, dict] = {}
    for cond, recs in by_cond.items():
        n_ex = len(recs)
        n_correct = sum(1 for r in recs if r.get("correct"))
        total_calls = sum(_count_calls_per_example(r) for r in recs)
        total_tokens = sum(_tokens_per_example(r) for r in recs)
        out[cond] = {
            "n_examples": n_ex,
            "n_correct": n_correct,
            "calls_total": total_calls,
            "calls_per_example": total_calls / max(n_ex, 1),
            "calls_per_correct": total_calls / max(n_correct, 1),
            "tokens_total": total_tokens,
            "tokens_per_correct": total_tokens / max(n_correct, 1),
        }
    return out
