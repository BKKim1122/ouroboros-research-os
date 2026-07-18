#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/../V16-E.C1_DGX_RESULTS_$(date -u +%Y%m%dT%H%M%SZ).tar.gz}"
tar -czf "$OUT" -C "$ROOT" \
  protocol src analysis reports environment logs raw README.md FROZEN_FILES_SHA256.txt MANIFEST.csv
echo "$OUT"
