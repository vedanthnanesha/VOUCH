#!/usr/bin/env bash
# Multimodal classification: accuracy and contract quality for every
# condition, same-family (Qwen text agent) and cross-family (Llama text
# agent), clean and adversarial regimes.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Fixes Python's hash seed so VOUCH-Random draws the same interventions on every run.
export PYTHONHASHSEED=0

CONDS="independent ensemble centralized_single centralized_coordinator contracts contracts_random contracts_adaptive"

# dataset:n
for spec in hateful_memes:400 crisismmd:300 mm_imdb:400; do
  ds=${spec%%:*}; n=${spec##*:}
  python -m experiments.main_evaluation --dataset "$ds" --n "$n" --conditions $CONDS
  python -m experiments.main_evaluation --dataset "$ds" --n "$n" --adversarial --conditions $CONDS
done

for spec in hateful_memes:400 crisismmd:300 mm_imdb:500; do
  ds=${spec%%:*}; n=${spec##*:}
  python -m experiments.cross_family_eval --dataset "$ds" --n "$n" --conditions $CONDS
  python -m experiments.cross_family_eval --dataset "$ds" --n "$n" --adversarial --conditions $CONDS
done
