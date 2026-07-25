#!/usr/bin/env bash
# place.sh — 다운로드 폴더의 중복 파일((1)(2)…) 중 '내용 마커가 맞는 최신(mtime)' 파일을 repo에 배치.
# 알파벳순 head -1 의 스테일 픽 문제를 제거. 공백 파일명("spec (10).yaml") 안전.
#
# 사용:  source tools/place.sh
#        place '마커문자열' '목적지경로' 후보파일들...
# 예:    place 'V22-surface-reduction' experiments/v22/spec.yaml "$DL"/spec*.yaml
#        place 'local_files_only'      experiments/v21/steer.py  "$DL"/steer*.py

place() {
  local marker="$1" dest="$2"; shift 2
  local newest="" f
  for f in "$@"; do
    [ -f "$f" ] || continue
    grep -q "$marker" "$f" 2>/dev/null || continue
    if [ -z "$newest" ] || [ "$f" -nt "$newest" ]; then newest="$f"; fi
  done
  if [ -n "$newest" ]; then
    cp "$newest" "$dest"
    echo "✓ $dest  ←  $newest"
    grep -q "$marker" "$dest" && echo "  (배치 검증: 마커 확인됨)"
  else
    echo "✗ '$marker' 매칭 파일 없음 — 다운로드 안 됐거나 옛 버전만 있음"
    return 1
  fi
}
