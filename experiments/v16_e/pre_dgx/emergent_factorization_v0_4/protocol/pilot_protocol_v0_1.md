# V16-E CPU Pilot Protocol v0.1

**상태:** 탐색 파일럿. 확증 사전등록 아님.

## 목적

통합 Transformer에서 V15까지의 계산축이 사후적으로 해독되고, 선형 subspace 개입이 정책에 선택적 영향을 줄 가능성이 있는지 저비용으로 확인한다.

## 환경

- 두 canonical entity A/B
- entity별 risk bits와 observer-neutral bits
- episode별 서로 다른 symbol을 A/B에 배정
- entity token 순서를 무작위화
- identity role token과 beneficiary role token은 target entity의 symbol을 제공
- common concern은 별도 context token으로 제공
- 행동은 none/protect A/protect B/protect both
- 정답 행동은 common concern, identity privilege, beneficiary priority, risk, cost의 결합으로 결정

## 금지된 구조

Unified 조건에는 다음을 두지 않는다.

- factor-specific slot
- factor-specific route
- factor gain
- 정책식 내부의 학습 가능한 별도 privilege/concern branch
- ground-truth candidate index 입력

## 조건

1. unified_behavior_only: action + world + neutral objective
2. unified_multitask: 위 objective + shared CLS factor auxiliary heads

## 분석

- layerwise factor probe
- cross-context probe
- probe direction 1D projection swap
- paired counterfactual input과 policy following
- norm-matched random direction control
- off-target probe stability
- world prediction stability
- deterministic symbol relabeling

## Seed와 실행량

- Pilot seed: 16120, 16121, 16122
- Condition당 650 steps
- Confirmatory seed에는 재사용하지 않음

## 해석 경계

- 높은 probe 성능은 정보 존재만 의미한다.
- counterfactual following과 off-target preservation이 함께 있어야 인과적 분화 후보로 본다.
- 입력 역할변수는 과제가 요구하므로 task-induced emergence이지 무목적 자연발생이 아니다.
- 인간의 봄, 자아경험, 도덕적 concern을 주장하지 않는다.
