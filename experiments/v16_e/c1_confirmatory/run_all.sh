#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/scripts/00_preflight.sh"
"$ROOT/scripts/01_smoke_test.sh"
"$ROOT/scripts/02_run_confirmatory.sh"
"$ROOT/scripts/03_run_causal_analysis.sh"
"$ROOT/scripts/04_finalize_analysis.sh"
"$ROOT/scripts/05_collect_results.sh"
