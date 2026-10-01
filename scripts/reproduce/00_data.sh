#!/usr/bin/env bash
# Build every dataset manifest at the size used in the paper, then the
# caption-masked image views used by the image-only agent.
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/download_data.py --dataset all
python scripts/build_masked_views.py --dataset hateful_memes
python scripts/build_masked_views.py --dataset mm_imdb
