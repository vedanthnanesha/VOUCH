"""Build the dataset manifests used in the paper.

Each loader downloads its source (Hugging Face mirrors, or the official
CrisisMMD archive), draws a fixed seed-0 sample, and writes
data/<name>/manifest.jsonl. The experiment scripts evaluate the first n
rows of a manifest and only build one on demand when it is missing, so
build every manifest at the sizes below before running any experiment.

Usage:
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset hateful_memes [--n 500]
    python scripts/download_data.py --dataset crisismmd     [--n 300]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CRISISMMD_URL = "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz"
CRISISMMD_RAW = REPO_ROOT / "data" / "crisismmd" / "raw"

# Manifest sizes used in the paper. Each experiment evaluates the first n
# rows (for example, Hateful Memes uses n=400 and MBPP dual-side n=200).
DEFAULT_N = {
    "hateful_memes": 500,
    "mm_imdb": 500,
    "crisismmd": 300,
    "gsm8k": 500,
    "math": 300,
    "svamp": 300,
    "mbpp": 500,
    "humaneval": 164,
}


def fetch_hateful_memes(n: int) -> None:
    from ccontracts.datasets import hateful_memes
    hateful_memes.download(n=n)


def fetch_mm_imdb(n: int) -> None:
    from ccontracts.datasets import mm_imdb
    mm_imdb.download(n=n)


def fetch_crisismmd(n: int) -> None:
    from ccontracts.datasets import crisismmd

    CRISISMMD_RAW.mkdir(parents=True, exist_ok=True)
    archive = CRISISMMD_RAW / "CrisisMMD_v2.0.tar.gz"
    extracted = CRISISMMD_RAW / "CrisisMMD_v2.0"

    if not extracted.exists():
        if not archive.exists():
            print(f"[crisismmd] downloading {CRISISMMD_URL} (~1.9 GB)")
            subprocess.run(["wget", "-O", str(archive), CRISISMMD_URL], check=True)
        print(f"[crisismmd] extracting to {CRISISMMD_RAW}")
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(CRISISMMD_RAW)], check=True)

    crisismmd.build(n=n)


def fetch_gsm8k(n: int) -> None:
    from ccontracts.datasets import gsm8k
    gsm8k.build(n=n)


def fetch_math(n: int) -> None:
    from ccontracts.datasets import math_bench
    math_bench.build(n=n)


def fetch_svamp(n: int) -> None:
    from ccontracts.datasets import svamp
    svamp.build(n=n)


def fetch_mbpp(n: int) -> None:
    from ccontracts.datasets import mbpp_codegen
    mbpp_codegen.build(n=n)


def fetch_humaneval(n: int) -> None:
    from ccontracts.datasets import humaneval
    humaneval.build(n=n)


FETCHERS = {
    "hateful_memes": fetch_hateful_memes,
    "mm_imdb": fetch_mm_imdb,
    "crisismmd": fetch_crisismmd,
    "gsm8k": fetch_gsm8k,
    "math": fetch_math,
    "svamp": fetch_svamp,
    "mbpp": fetch_mbpp,
    "humaneval": fetch_humaneval,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=[*FETCHERS, "all"])
    p.add_argument("--n", type=int, default=None,
                   help="Number of examples to keep (default: the paper's manifest size)")
    args = p.parse_args()

    targets = list(FETCHERS) if args.dataset == "all" else [args.dataset]
    for t in targets:
        n = args.n if args.n is not None else DEFAULT_N[t]
        print(f"\n=== {t} (n={n}) ===")
        FETCHERS[t](n)


if __name__ == "__main__":
    main()
