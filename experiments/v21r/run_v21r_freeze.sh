#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f "$HOME/.venv/bin/activate" ]; then source "$HOME/.venv/bin/activate"; fi
PY="$(command -v python || command -v python3)"
"$PY" - << 'PY'
import os, json, sys, yaml
sys.path.insert(0, os.getcwd())
from ouroboros import freeze as fz
exp = "experiments/v21r"
spec = yaml.safe_load(open(os.path.join(exp, "spec.yaml"), encoding="utf-8"))
p = fz.freeze(exp, spec, prompt_dir="ouroboros")
rec = json.load(open(p, encoding="utf-8"))
print("✓ freeze 완료:", p)
print(f"  파일 {len(rec['file_hashes'])}개 · 공용 {len(rec['prompt_hashes'])}개 봉인 · 확증 seed {rec['confirmatory_seeds']}")
PY
