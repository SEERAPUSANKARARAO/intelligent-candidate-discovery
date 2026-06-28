#!/usr/bin/env bash
# One-command bootstrap: create venv, install deps, generate data, launch server.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  echo "==> creating virtualenv (.venv)"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "==> generating synthetic data (idempotent)"
python -m backend.data_gen

echo "==> launching API + UI at http://localhost:8000"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
