# VOUCH: Counterfactual Contracts for Multi-Agent Coordination

Code for **VOUCH: Counterfactual Contracts for Multi-Agent Coordination** (Abhay Bhandarkar and Vedanth Nanesha), accepted to Findings of AACL-IJCNLP 2026.

VOUCH is a coordination protocol for multi-agent LLM systems in which every agent claim is coupled to a deterministic test that the protocol runs itself. When two agents disagree, the proposing agent picks an intervention from a fixed menu and commits to how the intervention will change its own output (self-side) and, in the dual-side construction, how it will change the other agent's output (cross-side). The protocol applies the intervention, re-queries the agents, and verifies the contract only if every commitment holds. A self-coherent but biased agent can satisfy a self-side commitment by construction; the cross-side commitment closes that gap.

The experiments cover multimodal classification (Hateful Memes, CrisisMMD, MM-IMDb), code generation (MBPP, HumanEval), and math reasoning (GSM8K, MATH algebra, SVAMP).

## Repository contents

- `ccontracts/`: the protocol implementation.
  - `contracts/`: contract data structures, the protocol and its weight updates (`protocol.py`), single- and dual-side verification (`verification.py`), and the classification, code, and math intervention menus.
  - `agents/`: model wrappers. Every model runs in a persistent subprocess worker (`_persistent_worker.py`, managed by `_worker_pool.py`).
  - `baselines/`: Independent, Ensemble, Centralized-Single, and Centralized-Coordinator (plus a free-deliberation baseline that the paper does not report).
  - `datasets/`: loaders that build seeded manifests for the eight benchmarks.
  - `task_config.py`: per-dataset prompts and label sets.
- `experiments/`: experiment entry points (classification, cross-family, single- and dual-side code and math, judge and self-consistency baselines) and post-hoc analyses (`format_results.py`, `analysis/`).
- `scripts/`: data preparation, derived tables and statistics, and figures. `scripts/reproduce/` runs the whole pipeline in order.
- `tests/`: a model-free unit test of the dual-side verification logic.

## Requirements

- Linux with an NVIDIA GPU. The paper's experiments ran on a single A100 (80 GB), and the agents run in full precision (fp32) by default. The model workers poll subprocess pipes, which is not supported on Windows.
- Python 3.10 or later.
- Access to the models on Hugging Face: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, and `meta-llama/Llama-3.1-8B-Instruct`. Llama 3.1 is gated: accept its license on Hugging Face and authenticate locally (`huggingface-cli login`).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build (`torch` and `torchvision`) for your system first if the command above would select a CPU-only build. The original experiment environment was not captured as a lockfile, so `requirements.txt` records the required packages and minimum versions rather than an exact environment.

## Data

No data is distributed with this repository. `scripts/download_data.py` downloads each benchmark, draws a fixed seed-0 sample, and writes `data/<dataset>/manifest.jsonl`. Experiments evaluate the first *n* rows of a manifest:

| Dataset | Source used by the loader | Manifest size | Evaluated |
|---|---|---|---|
| Hateful Memes | Hugging Face mirror `neuralcatcher/hateful_memes` (validation split) | 500 | first 400 |
| MM-IMDb | Hugging Face mirror `Aurorace1/MM-IMDb` | 500 | first 400 (cross-family: 500) |
| CrisisMMD v2.0 | official archive from `crisisnlp.qcri.org` (about 1.9 GB) | 300 | 300 |
| GSM8K | `openai/gsm8k` (test) | 500 | 500 single-side, first 300 dual-side |
| MATH (algebra) | `EleutherAI/hendrycks_math` (algebra, test) | 300 | 300 |
| SVAMP | `ChilleD/SVAMP` (test) | 300 | 300 |
| MBPP | `google-research-datasets/mbpp` (test) | 500 | 500 single-side, first 200 dual-side |
| HumanEval | `openai/openai_humaneval` (test) | 164 | 164 |

Build all manifests before running any experiment, then mask the burned-in text in the Hateful Memes and MM-IMDb images (EasyOCR plus inpainting) so that the image-only agent cannot read the caption:

```bash
python scripts/download_data.py --dataset all
python scripts/build_masked_views.py --dataset hateful_memes
python scripts/build_masked_views.py --dataset mm_imdb
```

The experiment scripts also build a missing manifest on first use, but only at the size they request, so building every manifest up front is what reproduces the paper's samples.

Each benchmark remains subject to its own license and terms of use. Hateful Memes is distributed by Meta under its own dataset license, and the loader fetches a community mirror for convenience; make sure you have accepted the original license terms before downloading it. Hateful Memes contains hateful and offensive content by construction.

## Running the experiments

`scripts/reproduce/` contains one script per group of experiments. Run them from the repository root:

```bash
bash scripts/reproduce/00_data.sh             # manifests and masked images
bash scripts/reproduce/01_classification.sh   # classification, same- and cross-family
bash scripts/reproduce/02_weight_ablation.sh  # weight-schedule ablation
bash scripts/reproduce/03_code.sh             # code generation, single- and dual-side
bash scripts/reproduce/04_math.sh             # math reasoning, single- and dual-side
bash scripts/reproduce/05_baselines.sh        # single-judge and self-consistency baselines
bash scripts/reproduce/06_analysis.sh         # derived tables, statistics and figures (CPU only)
```

The full pipeline took roughly 170 to 200 A100 GPU-hours. Every entry point can also be run on its own (see `--help`); the reproduce scripts list the exact arguments used for the paper.

| Experiments | Entry point | Outputs in `results/` |
|---|---|---|
| Classification accuracy and contract quality, all seven conditions | `experiments/main_evaluation.py` | `main_eval_<dataset>_<regime>.jsonl`, `summary_<dataset>_<regime>.json` |
| Cross-family classification (Llama text agent) | `experiments/cross_family_eval.py` | `cross_family_eval_<dataset>_<regime>.jsonl`, `summary_<dataset>_<regime>.cross_family.json` |
| Weight-schedule ablation | `experiments/main_evaluation.py --results-dir ...` | `weight_ablation/<schedule>/` |
| Code generation, single-side | `experiments/main_eval_codegen.py` | `<dataset>_codegen_{clean,adv}_n<N>.jsonl`, `summary_...json` |
| Code generation, dual-side | `experiments/main_eval_codegen_dual.py` | `<dataset>_codegen_dual[_adversarial]_n<N>.jsonl`, `summary_...json` |
| Math reasoning, single-side | `experiments/main_eval_math.py` | `<dataset>_math[_adversarial].jsonl`, `summary_...json` |
| Math reasoning, dual-side | `experiments/main_eval_math_dual.py` | `<dataset>_math_dual[_adversarial]_n300.jsonl`, `summary_...json` |
| Single-judge and self-consistency baselines | `experiments/baseline_judge.py`, `experiments/baseline_sc_math.py` | `baselines/` |

## Analyses and figures

`scripts/reproduce/06_analysis.sh` regenerates the following from `results/` without a GPU:

| Script | Writes | Contents |
|---|---|---|
| `scripts/derive_headline_stats.py` | `results/derived/headline_table.csv`, `abstention_disentangled.csv`, `heterogeneity_panel.csv`, `intervention_informativeness.csv`, `calibration_metrics.csv` | helped/hurt/caught/missed counts, action versus catch rates, same- versus cross-family accuracy, per-intervention change rates, per-agent confidence calibration |
| `scripts/derive_supplementary_tables.py` | `results/derived/judge_baseline_table.csv`, `intervention_fire_rate.csv` | judge-baseline table, per-intervention fire rates on math |
| `scripts/bootstrap_ci.py` | `results/derived/ci_wilson.csv`, `ci_bootstrap.csv` | Wilson intervals and paired-bootstrap CIs on the dual-side reduction |
| `scripts/mcnemar_cells.py` | `results/derived/per_cell_mcnemar.json` | per-cell and pooled exact McNemar tests, VOUCH versus Ensemble |
| `python -m experiments.format_results --analyses` | `results/analysis/summary.json` | per-agent self- and cross-form CC, per-intervention informativeness, cost, weight trajectories |
| `python -m experiments.render_figures` | `figures/` | CC, informativeness, sample-size stability, weight-trajectory, and summary figures |
| `scripts/make_revision_figures.py`, `scripts/make_fig_heterogeneity.py` | `figures/` | abstention, CI forest plot, per-intervention change, heterogeneity |
| `scripts/make_figures_better.py` | `figures_better/` | alternative styling of the same figures |

`experiments/reanalyze_multilabel.py` reports additional multi-label metrics for MM-IMDb (subset accuracy, Jaccard, macro- and micro-F1). `experiments/rank_interventions.py` computes the per-dataset informativeness ranking I(ι) from completed runs and prints it in the form used by `intervention_ranking` in `ccontracts/task_config.py` (VOUCH-Adaptive).

## Configuration

Environment variables (the `CONCORD_` prefix is the project's working name):

- `CONCORD_WEIGHT_PP`, `CONCORD_WEIGHT_PM`, `CONCORD_WEIGHT_MP`, `CONCORD_WEIGHT_MM`: multiplicative weight updates for the (proposer side, acceptor side) verification outcomes pass/pass, pass/fail, fail/pass, and fail/fail. Defaults: 1.20, 0.95, 0.90, 0.50.
- `CONCORD_WORKER_RECYCLE`, `CONCORD_WORKER_RECYCLE_VL`, `CONCORD_WORKER_RECYCLE_TEXT`: restart a model worker every *K* calls (defaults: 50 for the vision-language worker, 1 for text workers).
- `CONCORD_ALLOW_PARALLEL=1`: allow more than one `main_evaluation.py` process on the GPU.
- `CONCORD_ADV_PROMPT_OVERRIDE`: replace the adversarial system prompt.
- `CONCORD_DEBUG=1`: verbose worker logging.

Model calls are cached in `results/inference_cache/`, keyed by model, prompt, and image hash; delete it to force recomputation. Decoding is greedy everywhere except in the self-consistency baseline. The reproduce scripts set `PYTHONHASHSEED=0` so that VOUCH-Random draws the same interventions across runs.

## Responsible use

- The code-generation experiments execute model-generated Python in a subprocess with a timeout. This is not a security sandbox; run these experiments in an isolated environment, such as a container without network access or credentials.
- The adversarial system prompts in the experiment scripts deliberately bias an agent toward a shortcut decision rule. They exist to evaluate the protocol on fixed benchmarks and are not intended for deployment.
- The code is intended for research on multi-agent coordination, consistent with the research-only terms of the underlying benchmarks.

## Tests

```bash
python -m pytest tests
```

## Citation

Citation metadata is available in `CITATION.cff`. Proceedings metadata will be added after publication.

## License

Code is released under the MIT License; see `LICENSE`. Datasets and model weights are not included and remain subject to their own licenses and terms of use.
