#!/usr/bin/env bash
# Weight-schedule ablation on Hateful Memes adversarial (n=100). Each
# schedule sets the multipliers for the (proposer side, acceptor side)
# verification outcomes: pass/pass, pass/fail, fail/pass, fail/fail.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONHASHSEED=0

CONDS="contracts contracts_random contracts_adaptive ensemble independent"

run() {
  local tag=$1
  CONCORD_WEIGHT_PP=$2 CONCORD_WEIGHT_PM=$3 CONCORD_WEIGHT_MP=$4 CONCORD_WEIGHT_MM=$5 \
    python -m experiments.main_evaluation --dataset hateful_memes --n 100 --adversarial \
      --conditions $CONDS --results-dir "results/weight_ablation/$tag"
}

run aggressive 1.50 0.80 0.70 0.20
run default    1.20 0.95 0.90 0.50
run gentle     1.05 0.99 0.97 0.80
