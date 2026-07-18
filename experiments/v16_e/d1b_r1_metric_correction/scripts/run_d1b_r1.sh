#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-2}"
N="${N:-1200}"
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SEEDS=$(seq 16200 16223)
"$PYTHON" "$SCRIPT_DIR/analysis/v16e_d1b_r1_corrected.py" \
  --root "$ROOT/workspace" --seeds $SEEDS --n "$N" --device "$DEVICE" --threads "$THREADS" \
  | tee "$SCRIPT_DIR/reports/V16-E.D1b-R1_CONSOLE.txt"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar -czf "$HOME/V16-E.D1b-R1_RESULTS_${STAMP}.tar.gz" \
  -C "$ROOT/workspace" analysis/V16-E.D1b-R1 \
  -C "$SCRIPT_DIR" analysis/v16e_d1b_r1_corrected.py protocol README_KO.md reports
printf '%s\n' "$HOME/V16-E.D1b-R1_RESULTS_${STAMP}.tar.gz"
