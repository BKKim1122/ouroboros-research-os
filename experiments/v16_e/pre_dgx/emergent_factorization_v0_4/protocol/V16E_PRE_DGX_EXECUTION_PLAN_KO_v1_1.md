# V16-E DGX 연결 전 상세 실행계획 v1.1

## 목적

DGX 자원을 사용하기 전에 연구질문, 과제 식별성, 인과개입 방법, 통제군을 CPU에서 충분히 검증하여 대규모 확증실험의 실패비용을 최소화한다.

## 최적 경로 판정

V16-E는 현재 후보 중 **기대 정보량/비용 비율이 가장 높다.**

- 현재 핵심 약점인 construction objection을 직접 검증한다.
- V14/V15 자산을 재사용할 수 있다.
- 소형 CPU pilot으로 가설의 식별 가능성을 빠르게 판정할 수 있다.
- 성공·부분성공·실패 모두 V17 이후 설계를 좁혀준다.

이는 인간 봄과 가장 직접적으로 닮은 실험이라는 뜻이 아니다. 인간 단계로 가기 위한 가장 강한 선행증거를 가장 저렴하게 확보하는 경로라는 뜻이다.

## Workstream 1 — 과제 식별성

### 1.1 단일 통합행동 감사 — 완료

- 세 factor가 동일 action-value manifold로 합쳐질 수 있음을 확인
- cross-factor subspace가 서로 대체되어 factor-specificity 부재 확인

### 1.2 다중 행동맥락 — 1차 파일럿 완료

- identity-control, beneficiary-allocation, common-protection, integrated-policy
- 하나의 shared Transformer와 policy head
- factor label supervision 없음

### 1.3 동적 식별환경 — 다음 구현

각 factor에 시간적으로 독립적인 causal footprint를 부여한다.

| Factor | 독립적 장기 결과 |
|---|---|
| Identity | private memory/energy continuity, action-origin bookkeeping |
| Beneficiary | reward/utility가 귀속되는 entity의 장기 outcome |
| Concern | 두 entity harm에 대한 대칭 비용 |
| World | factor와 무관한 전이예측 |

직접 factor report나 factor-specific head는 사용하지 않는다.

## Workstream 2 — 표현 위치와 기하

### 2.1 층별·토큰별 probe

- CLS, entity tokens, role/context tokens
- within-context와 cross-context BACC 분리
- linear와 shallow nonlinear probe 비교

### 2.2 subspace overlap

- factor별 matched-difference PCA
- principal angle
- projection overlap
- shared policy manifold 비율

### 2.3 causal path patching

- upstream role token
- intermediate entity binding
- final CLS
- intervention-by-outcome matrix

목표는 factor 정보가 어디까지 분리되어 있고 어느 층에서 공통 action-value로 합쳐지는지 찾는 것이다.

## Workstream 3 — 통제군

필수 통제:

1. random norm-matched subspace
2. shuffled counterfactual pairing
3. cross-factor subspace
4. same-action/different-factor pairs
5. same-factor/different-action pairs
6. lexical relabeling
7. random entity order
8. symbol-only relational binding
9. held-out factor combination
10. world-state matched counterfactual

## Workstream 4 — 모델 조건

| 조건 | 학습목표 | 역할 |
|---|---|---|
| Factorized oracle | explicit factors | 기존 construction 상한 |
| Unified multitask | action/world + shared auxiliary heads | 표현가능성 기준 |
| Unified behavior-only | action/world only | 핵심 자연학습 조건 |
| Dynamic behavior-only | 장기 reward/world only | 강한 자연학습 조건 |

## Workstream 5 — Go/No-Go 기준

### Gate A: Task validity

- 각 factor flip이 충분한 counterfactual conflict case 생성
- factor를 제거한 ablation policy가 성능 저하
- action class imbalance 허용범위 내

### Gate B: Representation

- cross-context/OOD에서 factor 정보가 chance 이상
- lexical/position 통제 후 유지

### Gate C: Causal specificity

- own-factor subspace effect > cross-factor and shuffled controls
- bootstrap CI가 0 초과
- 비목표 outcome 하락이 사전 기준 이하
- world information 보존

### Gate D: Robustness

- pilot 3~5 seed에서 방향 일관
- held-out combination과 새 transition rule에서 재현

Gate C가 통과되지 않으면 DGX 확증을 시작하지 않는다.

## 실행 순서와 예상 공수

### Phase P2-1: 분석 확장

- 기간: 2~4 작업일
- 산출물: token/layer map, principal-angle table, path-patching matrix
- 연산: 현재 CPU 충분

### Phase P2-2: 동적 환경 v0.4

- 기간: 4~7 작업일
- 산출물: environment, behavior-only model, shortcut audit
- 연산: 현재 CPU 파일럿 3 seed 가능

### Phase P2-3: OOD·held-out

- 기간: 3~5 작업일
- 산출물: held-out combinations, symbol-only robustness, shuffled controls
- 연산: CPU 가능, 대규모 반복은 DGX 권장

### Phase P2-4: Freeze candidate

- 기간: 2~3 작업일
- 산출물: confirmatory protocol draft, source snapshot, environment lock, threshold rationale

총 DGX 전 예상: 약 2~4주 달력기간. AI 실행·분석 중심이며 사용자는 주요 설계변경과 동결 승인만 수행한다.

## DGX 연결 후

- 신규 seed 24~32개
- 3개 모델 규모
- robustness 12 seed 이상
- full activation storage
- multi-layer causal tracing
- exact source/environment freeze
- evidence package와 후속 논문

## 중단 규칙

다음이면 natural factorization 강한 주장을 중단한다.

- factor 정보는 읽히지만 모든 causal subspace가 공통
- factor-specificity가 action matching 후 사라짐
- OOD에서 표현·개입 붕괴
- behavior-only에서 반복적으로 실패하고 auxiliary supervision에서만 성공

이 경우 결과는 "upstream role information with downstream shared control geometry"라는 진단 논문 또는 V17 설계근거로 전환한다.
