#!/usr/bin/env bash
# V21 나/남 축 개입 파일럿 — DGX Spark. 먼저 --check-hook 로 hook 자가검증 권장.
#   bash run_v21_pilot.sh --check --model Qwen/Qwen2.5-1.5B    # hook 작동만 확인
#   bash run_v21_pilot.sh --sweep --model Qwen/Qwen2.5-1.5B    # α 스윕(깨끗한 구간 탐색)
#   bash run_v21_pilot.sh --model Qwen/Qwen2.5-1.5B            # 파일럿(R=300)
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # 캐시만 사용(네트워크 멈춤 방지)
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
CHECK=0; SWEEP=0; EXTRA=()
for a in "$@"; do case "$a" in --check) CHECK=1;; --sweep) SWEEP=1;; *) EXTRA+=("$a");; esac; done
if [ "$CHECK" -eq 1 ]; then
  "$PY" experiments/v21/steer.py --check-hook ${EXTRA[@]+"${EXTRA[@]}"}
elif [ "$SWEEP" -eq 1 ]; then
  "$PY" experiments/v21/steer.py --sweep ${EXTRA[@]+"${EXTRA[@]}"}
else
  "$PY" experiments/v21/steer.py --mode pilot ${EXTRA[@]+"${EXTRA[@]}"}
fi
echo; echo "✅ 리포트: experiments/v21/results/ (버저닝: *_타임스탬프.json + *_latest.json)"
