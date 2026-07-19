#!/usr/bin/env bash
# V18 freeze — 확증 전 코드+spec 동결(+거버너/freeze/ledger 공용코드 해시 기록). 인간이 실행.
# 동결 후 experiments/v18 의 코드/spec을 바꾸면 confirm 이 무효 처리한다.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
"$PY" - << 'PY'
import os, json, sys, yaml
sys.path.insert(0, os.getcwd())
from ouroboros import freeze as fz
exp = "experiments/v18"
spec = yaml.safe_load(open(os.path.join(exp, "spec.yaml"), encoding="utf-8"))
p = fz.freeze(exp, spec, prompt_dir="ouroboros")  # 공용 거버너/freeze/ledger도 해시에 기록
rec = json.load(open(p, encoding="utf-8"))
print("✓ freeze 완료:", p)
print(f"  experiment 파일 {len(rec['file_hashes'])}개 · 공용코드 {len(rec['prompt_hashes'])}개 해시 봉인")
print(f"  확증 seed: {rec['confirmatory_seeds']}  (파일럿 {rec['pilot_seeds']} 과 disjoint)")
print("  이후 experiments/v18 코드/spec 변경 시 confirm 이 무효 처리.")
PY
