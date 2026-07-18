# V16-E DGX 연결 전 마스터 계획 v1.0

## 1. 고정 연구 질문

명시적인 identity, beneficiary, privilege, concern 전용 slot·gain·경로가 없는 하나의 통합 Transformer가 행동과 세계예측을 학습하는 과정에서 다음 계산축을 구별되는 인과적 표현으로 조직하는가?

- 기능적 identity attribution
- beneficiary priority
- common protective responsiveness
- 상대적 privilege
- observer-neutral world information

V16의 목적은 인간의 봄을 직접 구현하는 것이 아니라, V10~V15에서 명시적으로 구성한 축들이 통합 학습계에서 task-induced latent organization으로 형성되는지 검증하는 것이다.

## 2. 객관적 우선순위 근거

이 단계는 다음 반론을 가장 낮은 비용으로 직접 제거한다.

> 연구자가 경로를 미리 나누었기 때문에 분리가 관찰된 것 아닌가?

메타알아차림, 신체화, LLM 적용은 모두 가치가 있으나 이 공백을 해결하지 않은 채 먼저 수행하면 동일한 construction objection이 반복된다. 따라서 연구 의존관계상 V16-E가 선행한다.

## 3. 단계 구분

### Pre-DGX P0: 설계 감사

- V14/V15 task와 주장 경계 재검토
- 입력 누출, 표면 cue, action imbalance 점검
- factorized oracle, unified multitask, behavior-only의 구분 확정

### Pre-DGX P1: CPU 파일럿

- Unified multitask 모델
- Unified behavior-only 모델
- action/world 성능 확인
- layerwise held-out probe
- cross-context probe
- 1차원 latent projection swap
- random norm-matched control
- lexical relabeling test

### Pre-DGX P2: 실패 진단·설계 고정 준비

- probe만 성공하고 intervention이 실패하는지 구분
- factor별 최적 layer가 지나치게 seed 의존적인지 확인
- action conflict sample 수 확인
- random control 대비 효과크기 산정
- OOD와 held-out combination 설계
- DGX용 확증 기준 초안 작성

### DGX P3: 확증 동결

Pilot 결과를 본 뒤 다음을 고정한다.

- exact source snapshot
- primary/secondary outcomes
- seed range
- model sizes
- intervention layer 선택 규칙
- OOD environments
- 제외 기준
- equivalence margin 사용 여부

### DGX P4: 확증·강건성

- 신규 seed 24개 이상
- 3개 모델 규모
- factorized oracle / unified multitask / behavior-only
- held-out role combination
- symbol/position/risk OOD
- activation patching, steering, causal scrubbing
- 실패 seed 전량 공개

## 4. 모델 조건

### A. Factorized oracle

기존 V14/V15의 명시적 구성. 성능 상한과 인과개입 기준점으로만 사용한다.

### B. Unified multitask

하나의 shared Transformer와 shared CLS. factor별 latent·gain은 없다. 학습 안정성 확인을 위해 동일 CLS에서 factor auxiliary heads만 허용한다.

### C. Unified behavior-only

행동과 observer-neutral world objective만 제공한다. Identity, beneficiary, concern label은 학습 loss에 사용하지 않고 사후 probe와 intervention에만 사용한다. V16의 핵심 조건이다.

## 5. 성공 수준

1. **Decodability:** cross-context probe로 factor가 읽힌다.
2. **Causal separability:** factor 방향 개입이 counterfactual policy를 선택적으로 따른다.
3. **Compositional generalization:** 학습하지 않은 조합과 OOD에서도 유지된다.

Probe 성공만으로 emergent causal factorization을 주장하지 않는다.

## 6. Pre-DGX Go/No-Go

### Go

- behavior-only action accuracy가 실용 수준에 도달
- 최소 2개 factor의 cross-context BACC가 chance보다 안정적으로 높음
- random control보다 factor intervention의 counterfactual following이 높음
- world prediction과 비목표 probe가 대체로 보존

### 제한적 성공

- probe는 높으나 인과개입이 약함
- multitask만 성공하고 behavior-only는 실패
- seed별 layer 위치가 불안정

이 경우 DGX 전 모델/환경을 재설계한다.

### No-Go

- lexical relabeling에서 붕괴
- factor 개입이 random control과 같음
- 전체 정책·세계정보가 함께 붕괴
- action imbalance로 factor 효과를 판정할 conflict case가 부족

## 7. Pre-DGX 산출물

- CPU pilot source/config/snapshot
- per-seed checkpoint·history·metadata
- base/probe/intervention CSV
- run registry
- pilot 진단 보고서
- DGX confirmatory protocol 초안
- 재현 스크립트와 manifest/checksum

## 8. 후속 단계와의 연결

V16-E 성공 후에만 자연학습된 축을 대상으로 메타접근과 비강제적 policy decoupling을 V17에서 검증한다. 기존 attention-recovery pilot은 V17 탐색자료로 보존한다. 기존 embodied handoff의 ownership·agency·control은 V18로 이관한다.
