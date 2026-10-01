"""Longitudinal trust-weight trajectories.

Reconstruct, example-by-example, the cumulative voting weight an agent
would carry if we ran VOUCH sequentially over a run's examples.
The weight update rule mirrors the protocol: × 0.5 on a failed contract
that the agent proposed; unchanged otherwise. The trajectory tells us
whether shortcut-sensitive agents' weights collapse and truthful agents'
weights stay near 1 across the run.

Input: a JSONL of contracts-condition ProtocolResult records, in the
order they were evaluated. Output: list of dicts keyed by example index
with cumulative per-agent weights.
"""
from __future__ import annotations

import json
from pathlib import Path


def weight_trajectory(jsonl: Path,
                      condition: str = "contracts",
                      decay_failed: float = 0.5
                      ) -> dict[str, list[tuple[int, float]]]:
    """Returns {agent_id: [(example_index, cumulative_weight), ...]} for
    the requested condition."""
    weights: dict[str, float] = {}
    trajectories: dict[str, list[tuple[int, float]]] = {}

    i = 0
    with Path(jsonl).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("condition") != condition:
                continue

            # initialise weights for all observed agents
            initial = r.get("initial_preds") or {}
            for aid in initial:
                if aid not in weights:
                    weights[aid] = 1.0
                    trajectories[aid] = [(0, 1.0)]

            # apply weight updates from this example's contracts
            for c in r.get("contracts") or []:
                p = c.get("proposer")
                if p not in weights:
                    continue
                if c.get("verified") is False:
                    weights[p] *= decay_failed
            i += 1
            for aid, w in weights.items():
                trajectories[aid].append((i, w))
    return trajectories
