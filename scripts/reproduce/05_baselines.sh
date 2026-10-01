#!/usr/bin/env bash
# Single-judge and self-consistency baselines on math. The judge reads the
# dual-side outputs of 04_math.sh, so run that first.
set -euo pipefail
cd "$(dirname "$0")/../.."

for f in gsm8k_math_dual_n300 gsm8k_math_dual_adversarial_n300 \
         svamp_math_dual_n300 svamp_math_dual_adversarial_n300 \
         math_math_dual_n300 math_math_dual_adversarial_n300; do
  python -m experiments.baseline_judge --in-file "results/$f.jsonl"
done

# Judge with an adversarial framing, on the clean runs.
for f in gsm8k_math_dual_n300 svamp_math_dual_n300 math_math_dual_n300; do
  python -m experiments.baseline_judge --in-file "results/$f.jsonl" --adversarial-judge
done

# Self-consistency (k=5, temperature 0.7).
python -m experiments.baseline_sc_math --dataset gsm8k --n 50
python -m experiments.baseline_sc_math --dataset math --n 50
