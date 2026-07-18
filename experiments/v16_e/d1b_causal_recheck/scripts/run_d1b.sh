#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-2}"
N="${N:-1200}"
SEEDS=$(seq 16200 16223)
python "$ROOT/analysis/v16e_d1b_causal_recheck.py" \
  --root "$ROOT/workspace" --device "$DEVICE" --threads "$THREADS" --n "$N" --seeds $SEEDS
python "$ROOT/analysis/summarize_d1b.py" --root "$ROOT"
