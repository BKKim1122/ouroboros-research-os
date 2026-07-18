# V16-E DGX 연결 전 진행 보고서 v0.3

## 1. 연구방향 판정

현재 시점에서 가장 효과적이고 효율적인 선행경로는 다음이다.

> V10~V15에서 명시적으로 구성한 identity attribution, beneficiary priority, relative privilege, common protective responsiveness의 계산적 구분이 factor-specific slot·gain·route가 없는 통합 학습계에서도 형성되는지 먼저 검증한다.

이 경로는 사용자의 마지막 제안이라서 선택한 것이 아니라, 다음 연구 의존관계를 기준으로 선택했다.

1. 현재 가장 큰 반론은 "분리해 설계했으므로 분리됐다"는 construction objection이다.
2. meta-awareness, decoupling, embodiment, LLM 적용은 이 반론을 해결하지 않은 채 먼저 수행하면 같은 문제가 반복된다.
3. 기존 V14/V15 환경과 코드를 재사용할 수 있어 비용이 가장 낮다.
4. 성공·부분성공·실패 모두 다음 연구의 설계정보가 된다.

따라서 V16-E를 먼저 수행하고, 자연학습된 축이 확인된 뒤 V17에서 meta-access와 비강제적 policy decoupling을 검증하는 순서가 고정 경로다.

## 2. DGX 연결 전 목표

DGX 이전에는 대규모 확증을 하지 않는다. 다음을 완료한다.

- unified behavior-only task의 식별 가능성 확인
- shortcut과 action imbalance 제거
- layerwise probe와 인과개입 방법 비교
- factor-specific subspace와 shared policy manifold 구분
- OOD·held-out 설계 확정
- confirmatory protocol 초안과 exact source-freeze 절차 완성

## 3. Pilot v0.1: relational-symbol-only 실패

초기 설계는 entity와 role을 임의 symbol matching으로만 연결했다.

- 행동 정확도 약 0.75
- identity/beneficiary auxiliary head와 probe는 chance
- concern만 거의 완벽히 해독
- identity/beneficiary intervention은 counterfactual policy를 따르지 못함

행동 정확도는 역할변수를 학습한 결과가 아니라 risk와 concern 중심의 부분정책으로 달성된 것이었다.

**판정:** 자연분화의 부정 결과가 아니라, relational binding 난이도와 partial-policy shortcut이 결합된 pilot 설계 실패. 결과는 archive_v0_1에 보존했다.

## 4. Pilot v0.2: 통합 모델 학습 성공, 1D 인과축 실패

Canonical body embedding을 entity/role token에 추가하되 내부 factor slot·gain·route는 만들지 않았다.

3개 seed, 두 조건을 실행했다.

- unified behavior-only
- unified multitask(shared CLS auxiliary heads만 사용)

### 기본 성능 평균

| 조건 | Action | World | Identity head | Beneficiary head | Concern head |
|---|---:|---:|---:|---:|---:|
| Behavior-only | 1.000 | 1.000 | 미학습 | 미학습 | 미학습 |
| Multitask | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Behavior-only의 사후 probe에서는 같은 분포 내 identity·beneficiary·concern이 거의 완벽히 해독됐다. 그러나 identity와 beneficiary의 cross-context BACC는 약 0.50~0.51로, 맥락을 넘는 단일 선형축은 형성되지 않았다. Concern은 약 0.90이었다.

1차원 probe 방향을 교환하는 개입은 정책을 충분히 바꾸지 못했다. 이는 decodability와 causal control이 다르다는 점을 다시 확인했다.

## 5. 다차원 matched-difference subspace

별도 calibration set에서 factor 하나만 뒤집은 paired activation difference를 수집하고, 상위 k차원 PCA subspace의 projection만 counterfactual 값으로 교환했다.

Behavior-only 3-seed 평균:

- k=2: counterfactual following 약 0.78~0.96
- k=4: identity 0.996, beneficiary 0.991, concern 1.000
- random norm-matched subspace: 0.000
- world prediction stability: 약 1.000

이는 통합 모델 내부에 factor 변화와 정책변화를 전달하는 저차원 구조가 존재함을 보여주는 강한 pilot 신호다.

그러나 factor-specificity 대조에서 중요한 반대 결과가 나왔다.

- identity subspace뿐 아니라 beneficiary/concern subspace도 identity counterfactual을 거의 전달
- beneficiary와 concern에도 같은 현상

즉 세 factor의 paired-difference subspace가 서로 독립적이라기보다, 최종 CLS에서 공통의 A/B action-value manifold에 크게 겹쳤다.

**판정:** 정보와 causal policy direction은 형성됐지만, factor-specific emergent causal factorization은 아직 입증되지 않았다.

## 6. Pilot v0.3: 다중 행동맥락으로 식별성 강화

단일 통합 정책은 identity·beneficiary·concern의 원인을 따로 보존할 필요가 없고 최종 A/B 가중치만 계산하면 된다. 이는 과제 수준의 비식별성이다.

따라서 factor label을 직접 감독하지 않으면서 독립적인 행동적 결과를 추가했다.

- q0: identity-conditioned control target
- q1: beneficiary-conditioned allocation target
- q2: common concern protection
- q3: 세 요인의 통합정책

모든 맥락은 하나의 shared Transformer와 하나의 policy head를 사용한다.

Seed 16130의 기본결과:

- q0~q3 action accuracy: 모두 1.000
- world BACC: 모두 1.000
- home-query probe: 모두 1.000
- integrated q3로의 probe 일반화: identity 0.641, beneficiary 0.842, concern 0.986

그러나 final CLS의 k=4 subspace는 여전히 상당히 겹쳤다. 다른 factor에서 학습한 subspace도 counterfactual policy를 광범위하게 전달했다.

**해석:** 독립적 행동 결과를 추가해도 최종 정책표현은 원인별 표현을 하나의 공통 제어공간으로 압축할 수 있다. 자연분화는 단일 최종 policy vector에서만 찾을 것이 아니라, 상류 role/context representation과 하류 shared control manifold를 층별로 분리해 측정해야 한다.

## 7. 현재 직접 결론

현재 pilot은 다음을 지지한다.

1. Factor-specific pathway 없이도 통합 Transformer가 세 역할변수를 사용해 완전한 행동과 세계예측을 학습할 수 있다.
2. 세 변수는 hidden representation에서 해독 가능하다.
3. matched counterfactual difference의 저차원 subspace는 정책을 인과적으로 전환할 수 있다.
4. 그러나 final CLS에서 이 subspace들은 강하게 겹치므로, factor-specific causal axes가 자연발생했다고 말할 수 없다.
5. 현재 관찰된 것은 `upstream factor information + downstream shared action-value manifold`일 가능성이 가장 높다.

이는 실패가 아니라 V16의 핵심 판별이다. V15까지의 명시적 factorization이 통합모델의 최종 정책공간에 그대로 자연발생한다는 단순 가설은 현재 pilot에서 지지되지 않았다.

## 8. DGX 전 남은 작업

### P2-A. 층·토큰별 위치 분석

- CLS뿐 아니라 entity/role/context token 전체 분석
- layer별 cross-context probe
- token-level causal patching
- attention pattern과 role-to-entity binding 분석
- upstream factor specificity와 downstream integration의 전이시점 식별

### P2-B. 식별 가능한 동적 환경

단순 query classification 대신 각 factor가 시간적으로 다른 결과를 갖게 한다.

- identity: 어느 상태가 private memory/energy continuity를 갖는가
- beneficiary: 보상이 어느 entity의 장기 outcome에 귀속되는가
- concern: 양쪽 harm 변화에 대한 대칭적 비용
- world: 독립적인 전이예측

모델은 단일 장기 policy와 world model만 학습하고 factor label은 받지 않는다.

### P2-C. 통제군

- shuffled paired-difference subspace
- cross-factor subspace
- random/norm-matched subspace
- action-matched but factor-different pairs
- factor-matched but action-different pairs
- lexical/symbol-only relational OOD
- held-out identity×beneficiary×concern 조합

### P2-D. 분석법

- principal-angle/subspace-overlap
- multi-dimensional causal patching
- nonlinear probe와 linear probe 분리
- causal mediation과 path patching
- layerwise intervention-by-outcome matrix

## 9. DGX 확증 진입조건

다음이 충족되기 전 확증 protocol을 동결하지 않는다.

- task가 세 factor를 실제로 필요로 함
- 부분정책 shortcut이 없음
- 적어도 한 층에서 own-factor subspace가 cross-factor보다 선택적임
- random/shuffled control보다 인과효과가 큼
- world information과 비목표 기능 보존
- held-out 조합에서 동일 경향
- seed별 효과 방향 안정

이 조건을 충족하면 DGX에서 신규 24 seed, 3개 모델크기, OOD 12 seed 이상으로 확증한다.

## 10. 계획 고정

현재 고정 순서는 다음과 같다.

1. V16-E: 자연학습 정보구조와 shared/분리 causal geometry 판별
2. V17: 자연학습된 상태에 대한 meta-access와 비강제적 policy decoupling
3. V18: ownership·agency·control을 포함한 embodied agent
4. V19: 공개 LLM·장기에이전트 전이
5. 인간 행동·EEG 구조동형성 검증

기존 attention-recovery pilot은 V17 탐색자료로 보존하고, 기존 embodied handoff는 V18 설계자료로 유지한다.
