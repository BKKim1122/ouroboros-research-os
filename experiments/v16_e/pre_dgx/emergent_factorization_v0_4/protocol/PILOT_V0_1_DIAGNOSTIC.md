# Pilot v0.1 진단

- Random symbol relation만으로 identity/beneficiary target을 지정한 초기 unified 모델은 650 step에서도 두 factor auxiliary head와 linear probe가 chance 수준에 머물렀다.
- Concern은 거의 완벽히 해독됐지만 identity/beneficiary는 cross-context에서 반전 또는 붕괴했다.
- 행동 정확도 약 0.75는 concern과 risk 중심의 부분정책으로 달성 가능해, 역할변수 학습 성공을 의미하지 않았다.
- factor intervention도 counterfactual policy를 따르지 못했다.

**판정:** emergent factorization의 부정 결과가 아니라, pilot task가 relational binding 난이도와 부분정책 shortcut을 동시에 포함한 설계 실패다. 결과는 삭제하지 않고 archive_v0_1에 보존한다.

v0.2에서는 canonical body embedding을 role/entity token에 추가하여 역할변수 자체가 학습 가능한지 먼저 확인한다. 이 조치는 내부 factor-specific pathway를 추가하지 않으며, 모든 정보는 여전히 하나의 shared Transformer를 통과한다. Symbol-only relational binding은 이후 robustness 조건으로 재도입한다.
