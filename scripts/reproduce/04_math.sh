#!/usr/bin/env bash
# Math reasoning (GSM8K, MATH algebra, SVAMP).
set -euo pipefail
cd "$(dirname "$0")/../.."

# Single-side (self-form) protocol.
for spec in gsm8k:500 math:300 svamp:300; do
  ds=${spec%%:*}; n=${spec##*:}
  python -m experiments.main_eval_math --dataset "$ds" --n "$n"
  python -m experiments.main_eval_math --dataset "$ds" --n "$n" --adversarial
done

# Dual-side protocol (Qwen2.5-7B proposer, Llama-3.1-8B acceptor).
for ds in gsm8k svamp math; do
  python -m experiments.main_eval_math_dual --dataset "$ds" --n 300
  python -m experiments.main_eval_math_dual --dataset "$ds" --n 300 --adversarial
done
