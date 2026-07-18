#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-/tmp/v16_meta_smoke}"
rm -rf "$OUT"
mkdir -p "$OUT"
python "$(dirname "$0")/src/v16_meta_awareness.py" --root "$OUT" --mode all --steps 40 --eval-size 200 --batch-size 96 --seeds 16990 --conditions full_monitor
