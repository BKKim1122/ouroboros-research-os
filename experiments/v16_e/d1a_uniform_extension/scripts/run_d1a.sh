#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
C1_ROOT="${1:-$(pwd)}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-2}"
TARGET_STEPS="${TARGET_STEPS:-2600}"
JOBS="${JOBS:-1}"

if [[ ! -f "$C1_ROOT/src/v16e_c1_confirmatory.py" ]]; then
  echo "ERROR: C1 root not found. Run as: $0 /path/to/V16-E.C1_DGX_CONFIRMATORY_20260712" >&2
  exit 2
fi
if [[ ! -d "$C1_ROOT/raw/V16-E.C1/compositional_seed16200" ]]; then
  echo "ERROR: completed C1 raw results are missing under $C1_ROOT/raw/V16-E.C1" >&2
  exit 2
fi

WORK="$HERE/workspace"
mkdir -p "$WORK/src" "$WORK/raw/V16-E.C1" "$HERE/logs"
cp "$C1_ROOT/src/v16e_c1_confirmatory.py" "$WORK/src/"

for seed in $(seq 16200 16223); do
  src="$C1_ROOT/raw/V16-E.C1/compositional_seed${seed}"
  dst="$WORK/raw/V16-E.C1/compositional_seed${seed}"
  if [[ ! -d "$dst" ]]; then
    cp -a "$src" "$dst"
  fi
done

run_seed() {
  local seed="$1"
  python "$WORK/src/v16e_c1_confirmatory.py" \
    --root "$WORK" --seed "$seed" --rule-mode compositional \
    --steps "$TARGET_STEPS" --eval-n 4000 --threads "$THREADS" \
    --device "$DEVICE" --resume \
    > "$HERE/logs/seed${seed}.log" 2>&1
  echo "completed seed $seed"
}
export -f run_seed
export HERE C1_ROOT DEVICE THREADS TARGET_STEPS WORK

if [[ "$JOBS" -le 1 ]]; then
  for seed in $(seq 16200 16223); do run_seed "$seed"; done
else
  seq 16200 16223 | xargs -n1 -P "$JOBS" -I{} bash -c 'run_seed "$@"' _ {}
fi

python "$HERE/analysis/summarize_d1a.py" | tee "$HERE/logs/summary.log"
(
  cd "$HERE"
  tar -czf "../V16-E.D1a_RESULTS_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
    protocol analysis logs workspace/raw/V16-E.C1 workspace/src
)
echo "D1a complete. Result archive created beside $HERE"
