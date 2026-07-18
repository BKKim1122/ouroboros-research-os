# V16-E Pilot v0.4 — Dynamic Identifiability Environment

## 목적

v0.2~v0.3에서 확인된 `상류 factor 정보 + 하류 shared action-value manifold` 가 단일 행동결과의 비식별성 때문인지 검증한다. identity, beneficiary, concern이 서로 다른 시간적·결과적 footprint를 갖는 하나의 unified behavior-only Transformer를 사용한다.

## 구조

- factor-specific latent, gain, route, factor-report head 없음
- 하나의 shared Transformer
- 모든 시점에 동일한 shared policy head와 shared outcome head 적용
- entity 순서와 symbol 무작위화
- identity/beneficiary는 relation token으로 entity와 결합
- concern은 공통 위해 민감도 context

## 네 시점

1. continuity: identity-linked private memory의 지속 여부에 따른 보존 행동
2. allocation: beneficiary-linked terminal energy 결과에 따른 자원배분
3. protection: concern 조건과 양쪽 risk에 따른 대칭 보호
4. integrated: V15형 identity/beneficiary/concern 통합정책

각 시점은 별도 head가 아니라 phase token에 동일한 policy/outcome head를 적용한다.

## Pilot 판정

- 모든 시점 action accuracy >= .95
- outcome prediction BACC >= .95
- layer-0 relation-token counterfactual patch가 지정 시점에 선택적 효과
- 중간층에서 own-factor patch 효과가 비지정 factor patch보다 큼
- 최종층 phase-token 표현은 통합될 수 있으므로 공유 manifold 자체는 실패로 간주하지 않음
- world/outcome stability를 함께 기록

## 한계

이 환경은 task-induced identifiability를 강화한 탐색적 환경이다. 자연발생의 강한 증거가 아니며, dynamic behavior-only confirmatory candidate를 고르기 위한 pilot이다.
