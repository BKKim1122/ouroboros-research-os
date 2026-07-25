#!/usr/bin/env bash
# V20 프롬프트 다양성 파일럿 — 모델×뱅크(orig/para/ko) 병합 재현. DGX Spark에서.
#   bash run_v20_pilot.sh --smoke                 # 1모델×전체뱅크, R=60
#   bash run_v20_pilot.sh                          # 6모델×3뱅크, R=1000
#   bash run_v20_pilot.sh --models qwen25_1p5b qwen3_4b --null-R 300
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
SMOKE=0; EXTRA=()
for a in "$@"; do case "$a" in --smoke) SMOKE=1;; *) EXTRA+=("$a");; esac; done
if [ "$SMOKE" -eq 1 ]; then
  echo "▶ V20 스모크: 1모델×3뱅크, R=60"
  "$PY" experiments/v20/discover_prompts.py --mode pilot --models qwen25_1p5b --null-R 60 ${EXTRA[@]+"${EXTRA[@]}"}
else
  echo "▶ V20 파일럿: 6모델×3뱅크, R=1000"
  "$PY" experiments/v20/discover_prompts.py --mode pilot ${EXTRA[@]+"${EXTRA[@]}"}
fi
echo; echo "✅ 리포트: experiments/v20/results/discover_prompts_report.json"
