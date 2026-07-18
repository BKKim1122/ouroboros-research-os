#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${DEVICE:-auto}"
THREADS="${THREADS:-2}"
SEED=19999
for MODE in compositional packed; do
  python3 "$ROOT/src/v16e_c1_confirmatory.py" --root "$ROOT" --seed "$SEED" --rule-mode "$MODE" --steps 100 --eval-n 200 --threads "$THREADS" --device "$DEVICE" > "$ROOT/logs/smoke_${MODE}_${SEED}.log" 2>&1
done
python3 "$ROOT/analysis/v16e_c1_causal_analysis.py" --root "$ROOT" --seeds "$SEED" --n 100 --device "$DEVICE" --threads "$THREADS" > "$ROOT/logs/smoke_causal.log" 2>&1
echo "Smoke test complete. Seed 19999 is preregistered as excluded."
