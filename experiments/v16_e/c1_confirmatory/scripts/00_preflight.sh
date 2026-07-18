#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/environment"
{
  echo "UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "HOST=$(hostname)"
  uname -a
  echo "--- OS ---"
  cat /etc/os-release 2>/dev/null || true
  echo "--- CPU/MEMORY ---"
  lscpu 2>/dev/null || true
  free -h 2>/dev/null || true
  echo "--- GPU ---"
  nvidia-smi 2>/dev/null || true
  echo "--- PYTHON ---"
  python3 - <<'PY'
import json, platform, sys
import numpy, pandas, torch
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
}, indent=2))
PY
} | tee "$ROOT/environment/PREFLIGHT.txt"
python3 -m py_compile "$ROOT/src/v16e_c1_confirmatory.py" "$ROOT/analysis/v16e_c1_causal_analysis.py" "$ROOT/analysis/v16e_c1_confirmatory_analysis.py"
echo "Preflight complete."
