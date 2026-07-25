#!/usr/bin/env bash
# V20 freeze — 확증 전 코드+spec 동결(공용코드 해시 포함). v18 러너와 동일 호출 규약.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
"$PY" - << 'PY'
import os, json, sys, yaml
sys.path.insert(0, os.getcwd())
from ouroboros import freeze as fz
exp = "experiments/v20"
spec = yaml.safe_load(open(os.path.join(exp, "spec.yaml"), encoding="utf-8"))
p = fz.freeze(exp, spec, prompt_dir="ouroboros")
rec = json.load(open(p, encoding="utf-8"))
print("✓ freeze 완료:", p)
print(f"  experiment 파일 {len(rec['file_hashes'])}개 · 공용코드 {len(rec['prompt_hashes'])}개 해시 봉인")
print(f"  확증 seed: {rec['confirmatory_seeds']}  (파일럿 {rec['pilot_seeds']} 과 disjoint)")
print("  이후 experiments/v20 코드/spec 변경 시 confirm 이 무효 처리.")
PY
