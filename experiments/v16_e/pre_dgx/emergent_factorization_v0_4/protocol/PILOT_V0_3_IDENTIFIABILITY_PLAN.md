# Pilot v0.3: Multi-context identifiability

v0.2의 단일 통합 행동은 identity, beneficiary, concern이 모두 동일한 A/B action-value manifold에 합쳐져도 최적 행동을 수행할 수 있었다. 실제로 factor별 matched-difference subspace뿐 아니라 다른 factor의 subspace도 거의 같은 counterfactual transfer를 일으켰다. 이는 자연분화 성공이 아니라 **과제 수준 비식별성**을 뜻한다.

v0.3은 factor label을 직접 감독하지 않으면서 각 변수가 서로 다른 행동 맥락에서 독립적인 결과를 갖게 한다.

- q0 control context: identity에 따라 motor target 선택
- q1 allocation context: beneficiary에 따라 recipient 선택
- q2 common protection context: concern과 risk에 따라 대칭 보호
- q3 integrated context: 세 요인의 결합정책

모든 맥락은 하나의 shared Transformer와 하나의 policy head를 사용한다. factor slot, gain, factor head는 없다.

성공 후보는 다음 intervention-by-query 구조다.

- identity subspace: q0와 q3에 효과, q1/q2에는 작음
- beneficiary subspace: q1와 q3에 효과, q0/q2에는 작음
- concern subspace: q2와 q3에 효과, q0/q1에는 작음
- cross-factor subspace와 random subspace는 target-specific transfer가 낮음

이 단계는 자연발생을 보장하는 장치가 아니라, latent factorization이 식별 가능하도록 필요한 독립적 causal footprint를 제공한다.
