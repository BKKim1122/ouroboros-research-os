#!/usr/bin/env bash
# V21R — 역방향 steering.  pilot: 배선점검 / confirm: freeze 후 정식(3모델×8seed, 수시간)
#   bash run_v21r.sh --pilot --models qwen25_1p5b --null-R 100
#   bash run_v21r.sh --confirm
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # 캐시만 사용(네트워크 멈춤 방지)
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
MODE=pilot; EXTRA=()
for a in "$@"; do case "$a" in --pilot) MODE=pilot;; --confirm) MODE=confirm;; *) EXTRA+=("$a");; esac; done
"$PY" experiments/v21r/steer_r.py --mode "$MODE" ${EXTRA[@]+"${EXTRA[@]}"}
