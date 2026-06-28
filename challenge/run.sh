#!/usr/bin/env bash
# One command: install deps, generate labelled synthetic data, run the ablation,
# produce submission.csv, and validate it. Fully offline / CPU-only.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
[ -d ".venv" ] || "$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r challenge/requirements.txt

N="${N:-100000}"
echo "==> generating $N synthetic candidates + ground truth"
python -m challenge.synth --n "$N" --out data/synth

echo "==> ablation study (proof each component earns its place)"
python -m challenge.evaluate --in data/synth/candidates.jsonl --gt data/synth/ground_truth.csv

echo "==> producing submission.csv"
python -m challenge.rank --in data/synth/candidates.jsonl --out submission.csv

echo "==> validating submission.csv"
python -m challenge.validate_submission submission.csv --pool data/synth/candidates.jsonl

echo "==> done. Launch the demo with:  streamlit run challenge/app.py"
