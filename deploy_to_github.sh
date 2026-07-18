#!/usr/bin/env bash
# Ouroboros Research OS — GitHub 원클릭 배포
#
# 준비 (딱 한 번):
#   1) gh CLI 설치            → https://cli.github.com  (예: sudo apt install gh)
#   2) gh auth login          → 브라우저/토큰으로 GitHub 로그인 (한 번만)
#
# 이후:
#   ./deploy_to_github.sh     → 최초엔 public repo 생성+업로드, 이후엔 변경분 커밋+푸시
#
# 옵션:
#   REPO_NAME 환경변수로 저장소 이름 변경 가능
#     REPO_NAME=my-repo ./deploy_to_github.sh
set -euo pipefail

REPO_NAME="${REPO_NAME:-ouroboros-research-os}"
DESC="A deterministic, governor-gated research OS for computational self-representation experiments."

cd "$(dirname "$0")"

# 1) gh CLI + 로그인 확인
if ! command -v gh >/dev/null 2>&1; then
  echo "❌ gh CLI가 없습니다. 설치: https://cli.github.com  (예: sudo apt install gh)"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "❌ GitHub 로그인이 필요합니다. 한 번만 실행하세요:  gh auth login"
  exit 1
fi

# 2) git 초기화 (최초 1회)
if [ ! -d .git ]; then
  git init -q
  git branch -M main
  echo "✓ git 초기화"
fi

# 3) 변경분 스테이지 + 커밋 (변경 없으면 건너뜀)
git add -A
if git diff --cached --quiet; then
  echo "· 커밋할 변경 없음"
else
  git commit -q -m "update: $(date +%Y-%m-%d_%H%M)"
  echo "✓ 커밋 생성"
fi

# 4) 원격이 있으면 푸시, 없으면 repo 생성 후 푸시
if git remote get-url origin >/dev/null 2>&1; then
  git push -q -u origin main
  echo "✓ 푸시 완료"
else
  echo "· 원격이 없어 새 public repo를 만듭니다: $REPO_NAME"
  gh repo create "$REPO_NAME" --public --source=. --remote=origin \
     --description "$DESC" --push
fi

URL="$(gh repo view --json url -q .url 2>/dev/null || true)"
echo "🎉 완료${URL:+: $URL}"
