#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${DEVICE:-auto}"
SEEDS=$(seq -s ' ' 16200 16223)
python3 "$ROOT/analysis/v16e_c1_causal_analysis.py" --root "$ROOT" --seeds $SEEDS --n 1200 --device "$DEVICE" --threads "${THREADS:-2}" | tee "$ROOT/logs/causal_analysis.log"
