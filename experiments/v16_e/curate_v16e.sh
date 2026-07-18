#!/usr/bin/env bash
# ==========================================================================
# V16-E 과거 실험 이력 → repo 정리 스크립트 (안전 모드)
#
# 하는 일:
#   홈의 V16-E.* 폴더들에서 '재현 레시피'(코드·스펙·문서 + 대표 리포트)만 골라
#   ~/ouroboros/experiments/v16_e/<갈래>/ 아래로 복사한다.
#
# 안전 원칙:
#   - 복사만 한다. 원본과 기존 repo 파일은 절대 수정/삭제하지 않는다.
#   - 대용량/실행산물 제외: .venv, raw/, logs/, *.tar.gz, *.db, *.ckpt,
#     seed_*.json, __pycache__, *.pyc, *.npy, *.npz, *.pt, *.safetensors
#   - 대표 리포트(reports/의 .md/.txt/.csv, 소용량 .json)는 포함.
#   - dry-run 기본. 실제 복사는 --apply 플래그를 줄 때만.
#
# 사용:
#   bash curate_v16e.sh            # 미리보기(무엇이 복사될지만 출력)
#   bash curate_v16e.sh --apply    # 실제 복사
# ==========================================================================
set -euo pipefail

HOME_DIR="$HOME"
REPO="$HOME/ouroboros"
DEST_ROOT="$REPO/experiments/v16_e"
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

# 원본 폴더 → repo 하위 갈래 이름 매핑
declare -A MAP=(
  ["V16-E.C1_DGX_CONFIRMATORY_20260712"]="c1_confirmatory"
  ["V16-E.8_identifiable_composition_20260712"]="e8_identifiable_composition"
  ["V16-E.D1a_uniform_extension_20260712"]="d1a_uniform_extension"
  ["V16-E.D1b_causal_recheck_20260712"]="d1b_causal_recheck"
  ["V16-E.D1b-R1_metric_correction_20260712"]="d1b_r1_metric_correction"
)

# 포함할 상위 디렉토리(존재할 때만) — 코드·스펙·문서
INCLUDE_DIRS=(src scripts protocol environment analysis reports docs)
# 포함할 최상위 개별 파일 패턴
INCLUDE_FILES=(README.md PROVENANCE.md MANIFEST.csv PLANNED_RUNS.csv \
  FROZEN_FILES_SHA256.txt "*_QUICKSTART*.txt" requirements.txt run_all.sh run.sh)

# 어떤 경로든 제외할 패턴(대용량/산물)
EXCLUDES=(--exclude='.venv' --exclude='__pycache__' --exclude='*.pyc'
  --exclude='raw' --exclude='logs' --exclude='*.tar.gz' --exclude='*.tgz'
  --exclude='*.db' --exclude='*.ckpt' --exclude='*.pt' --exclude='*.safetensors'
  --exclude='*.npy' --exclude='*.npz' --exclude='seed_*.json'
  --exclude='*.log' --exclude='*.bin')

# reports/ 안에서는 대용량 json도 컷(리포트 요약만): 10MB 초과 파일 제외
MAXBYTES=$((10*1024*1024))

echo "== V16-E 정리 (apply=$APPLY) =="
copy_one () {
  local src="$1" destname="$2"
  local dest="$DEST_ROOT/$destname"
  echo; echo "── $src  →  experiments/v16_e/$destname"
  [[ -d "$HOME_DIR/$src" ]] || { echo "   (원본 없음, 건너뜀)"; return; }
  [[ $APPLY -eq 1 ]] && mkdir -p "$dest"

  # 개별 파일
  for pat in "${INCLUDE_FILES[@]}"; do
    for f in "$HOME_DIR/$src"/$pat; do
      [[ -e "$f" ]] || continue
      echo "   file: $(basename "$f")"
      [[ $APPLY -eq 1 ]] && cp -n "$f" "$dest/"
    done
  done
  # 디렉토리
  for d in "${INCLUDE_DIRS[@]}"; do
    [[ -d "$HOME_DIR/$src/$d" ]] || continue
    echo "   dir : $d/"
    if [[ $APPLY -eq 1 ]]; then
      rsync -a "${EXCLUDES[@]}" --max-size=$MAXBYTES \
        "$HOME_DIR/$src/$d" "$dest/"
    fi
  done
}

for src in "${!MAP[@]}"; do
  copy_one "$src" "${MAP[$src]}"
done

# 세대 전체를 설명하는 README (없을 때만 생성)
if [[ $APPLY -eq 1 ]]; then
  readme="$DEST_ROOT/README.md"
  if [[ ! -e "$readme" ]]; then
    cat > "$readme" <<'MD'
# experiments/v16_e — V16-E 세대 (2026-07-12, DGX 확증)

이전 세션에서 edgexpert(DGX Spark)에서 수행한 V16-E 계열 실험의 **재현 레시피**.
코드·스펙·문서·대표 리포트만 포함하며, 대용량 실행 산물(원자료·체크포인트·
tarball·가상환경)은 git에 넣지 않는다(별도 백업/GitHub Release로 보관).

## 갈래
- `c1_confirmatory/`          — C1 본확증
- `e8_identifiable_composition/`
- `d1a_uniform_extension/`
- `d1b_causal_recheck/`
- `d1b_r1_metric_correction/`

## 재현
각 갈래의 README / PROVENANCE / FROZEN_FILES_SHA256 참조. 원자료가 필요하면
결과 tarball(별도 보관)을 내려받아 각 폴더 옆에 풀어 사용한다.

## 현행 연구와의 관계
V16-E 세대의 후속으로 V17(사전학습 LLM 자기표상 자연발생)이 진행 중이며,
설계 경위는 `docs/history/2026-07_v17_decision_log.md` 참조.
MD
    echo "   + README.md 생성: experiments/v16_e/README.md"
  fi
fi

echo; echo "== 완료 =="
[[ $APPLY -eq 0 ]] && echo "미리보기였습니다. 실제 복사하려면:  bash curate_v16e.sh --apply"
