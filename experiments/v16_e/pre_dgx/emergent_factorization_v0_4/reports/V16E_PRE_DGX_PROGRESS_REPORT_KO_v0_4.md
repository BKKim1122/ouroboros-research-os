# V16-E DGX 연결 전 진행 보고서 v0.4

## 1. 이번 단계의 목적

V15까지의 명시적 계산 분해가 unified behavior-only Transformer에서 어느 위치와 형태로 학습되는지 판별하기 위해 다음 두 작업을 수행했다.

1. **P2-A 층·토큰 표현지도:** v0.3 다중 행동맥락 모델 3개 seed에서 layer/token probe, factor counterfactual subspace geometry, token path patching 수행
2. **P2-B 동적 식별환경 v0.4:** identity, beneficiary, concern에 서로 다른 시간적·결과적 footprint를 부여하고 shared policy/outcome heads만으로 3개 seed 학습 및 causal tracing 수행

모든 결과는 exploratory pilot이며 확증 결과가 아니다.

## 2. P2-A: v0.3 층·토큰 지도

### 2.1 기본 결과

Seed 16130~16132의 unified behavior-only 모델은 모든 query에서 action/world 성능을 거의 완벽하게 유지했다.

통합정책 query(q3)에서 각 factor 정보는 해당 relation token에서 3개 seed 모두 BACC 1.0으로 해독됐다. 그러나 이는 입력 relation token에 body binding이 직접 포함되어 있으므로 자연발생 증거가 아니라 **입력 정보 지속성 확인**이다.

CLS에서는 concern과 beneficiary가 부분적으로 해독됐고 identity는 더 약하고 seed 변동이 컸다. 이는 최종 정책표현이 factor label을 균일한 선형축으로 유지하지 않음을 재확인한다.

### 2.2 Counterfactual subspace geometry

- identity-role 및 beneficiary-role token의 factor별 counterfactual subspace는 중간층에서 대체로 낮은 projection overlap을 보였다.
- CLS에서는 identity-beneficiary overlap이 상대적으로 높았고, 기존 v0.2/v0.3 cross-subspace intervention처럼 서로 다른 factor subspace가 동일한 정책전환을 광범위하게 유발했다.
- layer-0에서 변화가 없는 factor-token 조합의 SVD basis는 임의적이므로 해당 overlap은 해석 대상에서 제외한다.

### 2.3 Token path patching

3개 seed에서 공통적으로 관찰된 흐름:

1. **Embedding(layer 0):** counterfactual relation token patch가 해당 factor의 home/integrated policy를 선택적으로 전환
2. **중간층(layer 1):** factor 정보가 relation token과 CLS/출력 위치에 분산되며 seed별 경로 차이 발생
3. **마지막 block 출력(layer 2):** relation token만 바꿔서는 정책이 변하지 않고 CLS 또는 output token에 이미 전달된 표현을 바꿔야 정책이 전환

따라서 v0.3의 가장 적절한 계산지도는 다음이다.

```text
입력 relation binding
    -> 중간층의 분산 causal coalition
    -> 최종 shared action-value control representation
```

이는 factor-specific 최종 latent의 자연발생이 아니라 **상류 역할정보와 하류 공통 제어기하의 결합**을 지지한다.

## 3. P2-B: 동적 식별환경 v0.4

### 3.1 환경 변경

각 factor가 서로 다른 결과를 갖도록 네 phase를 구성했다.

| Phase | 주된 causal footprint |
|---|---|
| continuity | identity-linked private memory continuity |
| allocation | beneficiary-linked terminal energy |
| protection | concern 기반 양측 위해 보호 |
| integrated | identity + beneficiary + concern 통합정책 |

내부에는 factor-specific slot, gain, route, factor report head를 두지 않았다. 하나의 Transformer와 모든 phase에 공통인 policy head/outcome head만 사용했다.

### 3.2 3-seed 기본 성능

| Phase | 평균 action accuracy | 평균 outcome MAE |
|---|---:|---:|
| continuity | 1.000 | 0.0087 |
| allocation | 1.000 | 0.0543 |
| protection | 1.000 | 0.0343 |
| integrated | 1.000 | 0.0456 |

따라서 동적 과제는 세 factor를 사용해 안정적으로 학습 가능했다.

### 3.3 Layerwise causal routing

- layer 0에서는 identity, beneficiary, concern relation token patch가 각 지정 phase 및 integrated phase를 3 seed 모두 완전 전환했다. 이는 입력 cue의 기능적 타당성을 확인하지만 그 자체로 emergent factorization은 아니다.
- layer 1에서는 relation token 단독 효과가 seed별로 달랐고, 정보가 matching phase token으로 상당 부분 이동했다.
- layer 2에서는 relation token 효과가 사라지고 matching phase token patch가 counterfactual policy를 완전 전달했다.

### 3.4 Distributed coalition 결과

Layer 1에서 own-role token과 matching-phase token을 함께 patch했다.

| Target | 전용 phase joint follow | integrated joint follow |
|---|---:|---:|
| identity | 1.000 | 0.921 |
| beneficiary | 1.000 | 0.799 |
| concern | 1.000 | 0.922 |

전용 phase에서는 3 seed 모두 joint patch가 1.0이었다. unrelated role을 matching phase와 함께 patch한 결과는 phase-only와 거의 같아, own-role 정보가 추가적인 factor-specific causal contribution을 가짐을 보여준다.

그러나 role-only 효과는 seed에 따라 크게 달랐다. 따라서 자연스럽게 하나의 독립 factor slot이 형성됐다고 볼 수 없으며, **role source와 phase destination의 분산 causal coalition**으로 해석하는 것이 타당하다.

## 4. 현재 판정

### 직접 지지되는 내용

1. factor-specific 내부 route 없이도 unified model은 identity, beneficiary, concern을 서로 다른 시간적 결과에 사용한다.
2. factor 정보는 입력 relation token에서 시작해 중간층의 role/phase coalition을 거쳐 최종 phase-specific action representation으로 전달된다.
3. 잘 설계된 식별환경에서는 own-role 정보가 unrelated role보다 선택적인 인과기여를 보인다.
4. 최종 정책표현은 여전히 factor별 독립축보다 phase/action-value 표현으로 통합된다.

### 아직 지지되지 않는 내용

1. identity, beneficiary, concern이 명시적 task 구조 없이 독립적인 latent factor로 자연발생했다는 주장
2. 하나의 seed-independent factor-specific 경로나 단일 causal direction이 존재한다는 주장
3. 인간의 자기구조 또는 봄과 직접 동형이라는 주장

## 5. 연구적 의미

v0.2~v0.4 결과는 단순한 성공/실패보다 다음 구조를 제안한다.

```text
factor 정보의 국소적·관계적 표상
    -> 여러 token에 분산된 causal coalition
    -> 행동시점별 policy representation
    -> 공통 action-value geometry
```

즉 자연학습되는 것은 명시적 slot과 동일한 분해라기보다, **필요한 계산시점에 역할정보를 전달하는 동적 causal routing**일 수 있다. 이는 V12의 distributed causal coalition과 V13의 conditional routing 결과를 자연학습형 통합모델로 연결하는 더 정합적인 후보다.

## 6. 다음 단계: Pilot v0.5

v0.4는 식별성은 높지만 relation token에 canonical body embedding을 포함하므로 factor 정보가 입력에서 지나치게 직접적이다. 다음 단계에서는 내부 분해를 더 강하게 시험한다.

1. role token의 canonical body embedding 제거
2. entity symbol과 relation cue만으로 binding하되, factor label head 없이 transition/outcome prediction으로 grounding
3. curriculum: 쉬운 binding -> randomized symbol/order -> held-out symbol permutation
4. same-action/different-factor 및 same-factor/different-action 대조
5. action-matched counterfactual path patching
6. held-out identity x beneficiary x concern 조합
7. 새 transition rule OOD

### DGX 진입 판정 유지

다음이 충족되기 전 confirmatory freeze를 하지 않는다.

- symbol/position 통제 후 factor 정보 유지
- own-factor causal contribution이 cross-factor/shuffled보다 큼
- held-out 조합에서 재현
- 3~5 seed 방향 일관성
- world/outcome 기능 보존

## 7. 상태

- P2-A layer/token mapping: 완료
- P2-B dynamic v0.4 implementation: 완료
- v0.4 3-seed training: 완료
- v0.4 path patching/coalition analysis: 완료
- 강한 emergent factorization: 아직 미확인
- 현재 최선의 후보: distributed relational routing -> phase-specific policy -> shared control geometry
