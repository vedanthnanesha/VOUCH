#!/usr/bin/env bash
# Code generation (MBPP, HumanEval). Generated code is executed locally;
# run this in an isolated environment.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Single-side (self-form) protocol on the full samples.
python -m experiments.main_eval_codegen --dataset mbpp --n 500 \
  --out results/mbpp_codegen_clean_n500.jsonl --summary results/summary_mbpp_codegen_clean_n500.json
python -m experiments.main_eval_codegen --dataset mbpp --n 500 --adversarial \
  --out results/mbpp_codegen_adv_n500.jsonl --summary results/summary_mbpp_codegen_adv_n500.json
python -m experiments.main_eval_codegen --dataset humaneval --n 164 \
  --out results/humaneval_codegen_clean_n164.jsonl --summary results/summary_humaneval_codegen_clean_n164.json
python -m experiments.main_eval_codegen --dataset humaneval --n 164 --adversarial \
  --out results/humaneval_codegen_adv_n164.jsonl --summary results/summary_humaneval_codegen_adv_n164.json

# Dual-side protocol; each record also scores the single-side check on the same contract.
python -m experiments.main_eval_codegen_dual --dataset mbpp --n 200
python -m experiments.main_eval_codegen_dual --dataset mbpp --n 200 --adversarial
python -m experiments.main_eval_codegen_dual --dataset humaneval --n 164
python -m experiments.main_eval_codegen_dual --dataset humaneval --n 164 --adversarial
