#!/usr/bin/env bash
# V19 파일럿 — 다중모델×층 자기관련 축 발견 + ben↔conc 병합 확증. DGX Spark에서 실행.
#   bash run_v19_pilot.sh --smoke            # 1모델(qwen25_0p5b), R=60 (~2-3분)
#   bash run_v19_pilot.sh                     # spec 전체 모델, R=1000 (모델당 ~7분)
#   bash run_v19_pilot.sh --models qwen25_0p5b qwen25_1p5b   # 부분집합
#   추가 인자는 discover_multi.py로 전달(--device cpu 등)
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
SMOKE=0; EXTRA=()
for a in "$@"; do case "$a" in --smoke) SMOKE=1;; *) EXTRA+=("$a");; esac; done
if [ "$SMOKE" -eq 1 ]; then
  echo "▶ V19 스모크: 1모델(qwen25_0p5b), R=60"
  "$PY" experiments/v19/discover_multi.py --mode pilot --models qwen25_0p5b --null-R 60 ${EXTRA[@]+"${EXTRA[@]}"}
else
  echo "▶ V19 파일럿: 전체 모델, R=1000"
  "$PY" experiments/v19/discover_multi.py --mode pilot ${EXTRA[@]+"${EXTRA[@]}"}
fi
echo; echo "✅ 리포트: experiments/v19/results/discover_multi_report.json  ← 공유해줘"
