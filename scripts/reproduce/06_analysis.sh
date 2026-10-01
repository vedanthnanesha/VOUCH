#!/usr/bin/env bash
# Derived tables, statistics and figures. CPU only; reads results/.
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/derive_headline_stats.py
python scripts/derive_supplementary_tables.py
python scripts/bootstrap_ci.py
python scripts/mcnemar_cells.py
python -m experiments.format_results --analyses

python -m experiments.render_figures
python scripts/make_revision_figures.py
# Overwrites the fig_heterogeneity.pdf written by make_revision_figures.py.
python scripts/make_fig_heterogeneity.py
python scripts/make_figures_better.py
