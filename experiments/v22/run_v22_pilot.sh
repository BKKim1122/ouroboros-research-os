#!/usr/bin/env bash
# V22 표면 환원 통제 파일럿 — DGX Spark.
#   bash run_v22_pilot.sh --smoke                  # 1모델, R=100
#   bash run_v22_pilot.sh                           # 6모델, R=300
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # 캐시만 사용(네트워크 멈춤 방지)
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
SMOKE=0; EXTRA=()
for a in "$@"; do case "$a" in --smoke) SMOKE=1;; *) EXTRA+=("$a");; esac; done
if [ "$SMOKE" -eq 1 ]; then
  "$PY" experiments/v22/surface_probe.py --mode pilot --models qwen25_1p5b --null-R 100 ${EXTRA[@]+"${EXTRA[@]}"}
else
  "$PY" experiments/v22/surface_probe.py --mode pilot ${EXTRA[@]+"${EXTRA[@]}"}
fi
echo; echo "✅ 리포트: experiments/v22/results/ (버저닝)"
