#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/analysis/v16e_c1_confirmatory_analysis.py" --root "$ROOT" | tee "$ROOT/logs/confirmatory_analysis.log"
python3 "$ROOT/scripts/audit_results.py" --root "$ROOT" | tee "$ROOT/logs/audit.log"
