#!/usr/bin/env bash
# V18 파일럿 원클릭 러너 — DGX Spark(edgexpert)에서 실행.
# 실모델(config mock:false) seeds 0 1 2로 파이프라인·null 분포 검증. 승격 없음.
# 확증(seeds 20-27)은 freeze+거버너(discovery) 배선 후 별도 러너로.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # repo 루트
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"

MODE="full"; [ "${1:-}" = "--smoke" ] && MODE="smoke"
if [ "$MODE" = "smoke" ]; then
  echo "▶ 스모크(파이프라인만, null R=50, 빠름):"
  "$PY" experiments/v18/discover.py --mode pilot --seeds 0 1 2 --null-R 50
else
  echo "▶ 파일럿(실모델, null R=1000):"
  "$PY" experiments/v18/discover.py --mode pilot --seeds 0 1 2
fi
echo
echo "✅ 리포트: experiments/v18/results/discover_report.json  ← 이 파일을 공유해줘."
