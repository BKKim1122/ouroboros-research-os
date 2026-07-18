#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${DEVICE:-auto}"
JOBS="${JOBS:-1}"
THREADS="${THREADS:-2}"
python3 "$ROOT/scripts/run_matrix.py" --root "$ROOT" --device "$DEVICE" --jobs "$JOBS" --threads "$THREADS"
