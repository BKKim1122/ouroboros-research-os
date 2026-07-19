#!/usr/bin/env bash
# V18 확증 — freeze 검증 → 실모델 confirm(seeds 20-27) → 거버너(discovery) 판정 → 원장 기록.
# 인간이 실행. 등급 천장 승격은 인간 게이트(cli approve --gate claim_promotion) 필요.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
"$PY" experiments/v18/confirm.py "$@"
