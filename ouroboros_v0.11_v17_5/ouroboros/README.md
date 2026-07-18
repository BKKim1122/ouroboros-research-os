# Ouroboros Research OS v0.1

AI 계산구조 연구(V16~)와 인간 체험 연구를 하나의 폐루프로 운영하기 위한
**결정론적 연구 상태기계**. 에이전트 군집이 아니라 상태기계가 중심이며,
LLM 에이전트는 제한된 역할(설계·반증·연결)만 수행한다.

## 원안 대비 수정사항 (설계 근거)

1. **자동화 % 표 삭제** — 자동화 수준은 게이트 구조로만 정의 (허구적 정밀도 배제)
2. **에이전트 13종 → 5종** — Architect / Adversary(Minimalist+Shortcut Hunter 통합) /
   North-Star / Bridge(Phase 3) + Runner·Auditor·Governor는 **LLM이 아닌 순수 코드**
3. **2축 주장 격자** — 기계론 E0–E7 × 대응 H0–H3 분리
   (E8을 같은 사다리에 두면 기계론 증거 강도가 인간 대응 강도로 오인됨)
4. **불변식 I2** — LLM 리뷰는 증거수준을 절대 올릴 수 없음. 강등/플래그만 가능.
   승격은 규칙 기반 `governor.py`만 수행하며 E5 초과는 인간 게이트 필수.
5. **N=1 자기실험 보정** — 인간 측 예측은 관찰 **이전 봉인**(sealed_predictions)만
   H2 후보. 사후 대응은 전부 약한 비유로 강등. 기대효과 리스크를 매 판정에 명시.
6. **통계 사전 등록** — min_confirmatory_seeds / effect_size_min /
   multiple_comparison / gpu_tolerance가 spec 필수 필드. 동결 후 변경 불가.
7. **스택 최소화** — v0.1은 stdlib + PyYAML + SQLite + Git.
   LangGraph/Prefect/MLflow/DVC는 병목 확인 후 단계적 도입 (아래 로드맵).

## 시스템 불변식

- **I1** FREEZE_GATE, HUMAN_LOOP, E5 초과 승격은 ledger의 인간 승인 기록 필수
- **I2** LLM 출력은 증거수준을 올릴 수 없다
- **I3** Adversary의 blocking_issues가 있으면 파일럿 진행 불가
- **I4** protocol_freeze 이후 파일 해시가 변하면 확증실험 무효 (`freeze.verify`)
- **I5** 파일럿 데이터는 확증 분석에서 제외

## 상태 흐름

```
IDLE → OBSERVATION → MODEL_UPDATE → DESIGN → ADVERSARIAL_REVIEW
     → PILOT → PILOT_AUDIT → [FREEZE_GATE 인간승인] → CONFIRMATORY
     → ANALYSIS → CAUSAL_AUDIT → CLAIM_ADJUDICATION → ARCHIVE
     → [HUMAN_LOOP 인간승인] → OBSERVATION (다음 순환)
```

## 빠른 시작

```bash
pip install pyyaml
python cli.py loop --experiment demo_experiment      # FREEZE_GATE에서 정지 (정상)
python cli.py approve --gate protocol_freeze --experiment V16E-demo --by 김병관
python cli.py loop --experiment demo_experiment      # 확증→감사→판정→아카이브
python cli.py status                                 # 주장 원장 확인
export ANTHROPIC_API_KEY=...                         # 설정 시 Adversary 등 실제 LLM 검토
```

## 실제 V16 실험으로 교체하는 방법

1. `demo_experiment/`를 복제해 `experiments/v16e/` 생성
2. `run_seed.py`를 실제 학습·개입 코드로 교체 —
   단, 출력 JSON 계약(`factors / effects 행렬 / controls`)만 유지하면
   audit·governor·freeze가 그대로 작동
3. `spec.yaml`의 seeds/stats/kill_criteria/claim_ceiling을 실제 값으로 등록
4. DGX에서 seed 병렬화가 필요해지면 `cli.py`의 `run_seeds`만
   Prefect flow 또는 단순 `concurrent.futures`로 교체 (계약 동일)

## 단계별 도입 로드맵

| 단계 | 추가하는 것 | 도입 조건(병목 신호) |
|---|---|---|
| Phase 1 (현재) | 상태기계+원장+동결+감사+판정 | — |
| Phase 1.5 | MLflow(run 비교), DVC(대형 checkpoint) | 결과 폴더 수동 비교가 고통스러울 때 |
| Phase 2 | Adversary 상시화, claim lattice 자동 Go/No-Go, 판별실험 후보 3개 자동 생성 | 파일럿 1회전이 안정화된 후 |
| Phase 2.5 | Prefect(재시도·중단재개), LangGraph(장기 세션 checkpoint) | seed 실행이 수 시간 단위가 될 때 |
| Phase 3 | Human Mirror + sealed predictions + Bridge | E4 이상 주장이 2개 이상 확보된 후 |

## Phase 3 인간 루프의 필수 안전장치

- 관찰 과제 제시 **이전에** AI 측 예측을 해시와 함께 봉인 (ledger에 기록)
- 봉인 예측 일부를 가짜 예측과 섞어 블라인드 제시 (기대효과 통제)
- "AI 이론과 불일치한 체험" 항목을 필수 응답으로 강제
- 생리·EEG 자기실험은 별도 protocol freeze + 중단기준 문서화 후 human_study 게이트
