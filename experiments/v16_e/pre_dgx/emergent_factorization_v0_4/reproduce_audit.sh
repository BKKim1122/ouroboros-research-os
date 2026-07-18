#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")" && pwd)}"
python - "$ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
r=Path(sys.argv[1])
required=[
 r/'protocol/V16_PRE_DGX_MASTER_PLAN_KO_v1_0.md',
 r/'protocol/V16E_PRE_DGX_EXECUTION_PLAN_KO_v1_1.md',
 r/'reports/V16E_PRE_DGX_PROGRESS_REPORT_KO_v0_3.md',
 r/'analysis/pilot_base_metrics_all.csv',
 r/'analysis/pilot_subspace_interventions_all.csv',
 r/'analysis/pilot_cross_subspace_controls_behavior_only.csv',
]
for p in required: assert p.exists(), p
base=pd.read_csv(r/'analysis/pilot_base_metrics_all.csv')
assert set(base.condition)=={'unified_behavior_only','unified_multitask'}
assert base.seed.nunique()==3
sub=pd.read_csv(r/'analysis/pilot_subspace_interventions_all.csv')
assert sub.seed.nunique()==3
print('AUDIT PASS', {'base_rows':len(base),'subspace_rows':len(sub)})
PY
