#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/tmp/v16e_smoke}"
rm -rf "$OUT"
mkdir -p "$OUT"
python "$ROOT/src/v16e_emergent_factorization.py" --root "$OUT" --seeds 16990 --conditions unified_behavior_only --steps 80 --eval-n 300
python - "$OUT" <<'PY'
import sys, pandas as pd
from pathlib import Path
r=Path(sys.argv[1])
p=r/'raw/pilot/unified_behavior_only_seed16990/base_metrics.csv'
assert p.exists(), p
d=pd.read_csv(p)
assert len(d)==1
print('SMOKE PASS', d[['action_acc','world_bacc']].to_dict('records')[0])
PY
