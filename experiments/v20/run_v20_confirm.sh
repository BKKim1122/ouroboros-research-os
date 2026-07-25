#!/usr/bin/env bash
# V20 확증 — freeze 검증 → confirm 실행(orig/para/ko × seed 20-27) → 거버너 판정.
#   bash run_v20_confirm.sh --by 김병관
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # 캐시만 사용(네트워크 멈춤 방지)
cd "$(dirname "$0")"
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
"$PY" confirm_v20.py "$@"
