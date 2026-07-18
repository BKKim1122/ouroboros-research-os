# V16-E.C1 DGX Confirmatory Package

## 목적

V16-E.5~E.7의 탐색 결과를 독립 seed 24개로 확증한다. Primary claim은 compositional encoding의 우월성이 아니라 다음 세 항목이다.

1. factor-specific slot/gain/route/report/loss가 없는 unified behavior-only Transformer에서 episode-local identity, beneficiary, concern 관계가 학습되는가
2. 미학습 symbol, factor 조합, operator 조합에 재사용되는가
3. matched input-token intervention이 지정 policy/outcome을 counterfactual 방향으로 바꾸면서 비목표 정책을 보존하는가

## 명명

- `V16-E.7`: 마지막 탐색 파일럿
- `V16-E.C1`: 첫 확증실험(Confirmatory 1)

확증 단계이므로 `V16-E.8`이 아니라 `V16-E.C1`로 분리했다.

## 동결 항목

- Primary seeds: 16200–16223
- 두 모델: compositional, parameter-count-matched packed
- 각 모델 75,102 parameters
- 1,100 training steps
- seed/condition당 base evaluation 4,000 episodes
- causal target/seed당 1,200 matched episodes
- FP32, TF32 off, dropout 0
- 성공기준과 해석 제한은 `protocol/V16-E.C1_CONFIRMATORY_PROTOCOL_FROZEN.json`에 고정

## DGX Spark 실행

압축을 해제한 폴더에서 실행한다.

```bash
cd V16-E.C1_DGX_CONFIRMATORY_20260712
chmod +x run_all.sh scripts/*.sh

# 1. 환경 점검
./scripts/00_preflight.sh

# 2. 제외 seed를 이용한 짧은 점검
DEVICE=cuda THREADS=2 ./scripts/01_smoke_test.sh

# 3. 24 seeds × 2 models 본실험
DEVICE=cuda JOBS=1 THREADS=2 ./scripts/02_run_confirmatory.sh

# 4. 사전 고정 인과분석
DEVICE=cuda THREADS=2 ./scripts/03_run_causal_analysis.sh

# 5. Gate 판정 및 무결성 감사
./scripts/04_finalize_analysis.sh

# 6. 반환용 결과 패키지 생성
./scripts/05_collect_results.sh
```

전체를 한 번에 실행할 때:

```bash
DEVICE=cuda JOBS=1 THREADS=2 ./run_all.sh
```

## CPU 병렬 대안

이 모델은 매우 작아 GPU보다 CPU 다중 실행이 빠를 수도 있다. CUDA smoke가 비정상적으로 느릴 때만 본실험 시작 전에 아래 대안을 선택한다.

```bash
DEVICE=cpu JOBS=8 THREADS=2 ./scripts/02_run_confirmatory.sh
DEVICE=cpu THREADS=2 ./scripts/03_run_causal_analysis.sh
./scripts/04_finalize_analysis.sh
./scripts/05_collect_results.sh
```

본실험을 시작한 뒤 backend를 seed별로 혼합하지 않는다. 중단 시 같은 명령을 다시 실행하면 저장된 model/optimizer/batch-generator 상태에서 재개한다.

## 주요 결과 파일

- `analysis/V16-E.C1/CONFIRMATORY_SUMMARY.json`
- `analysis/V16-E.C1/CONFIRMATORY_DECISIONS.csv`
- `analysis/V16-E.C1/AUDIT.json`
- `reports/V16-E.C1_DGX_CONFIRMATORY_RESULTS_KO.md`
- `V16-E.C1_DGX_RESULTS_*.tar.gz`

## 금지사항

결과를 본 뒤 seed, threshold, step 수, architecture, loss, held-out 조합을 변경하지 않는다. 실패 seed도 제외하지 않는다. Packed 비교와 integrated policy는 secondary analysis이며 primary Gate를 대체하지 않는다.
