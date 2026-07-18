# V16-E DGX 연결 전 실행계획 v1.2

## 완료

- v0.2 unified behavior-only/multitask
- v0.3 multi-context identifiability
- P2-A 3-seed layer/token probe, subspace geometry, path patching
- v0.4 dynamic identifiability environment
- v0.4 3-seed causal routing and role+phase coalition analysis

## 현재 핵심 결과

최종 factor-specific latent보다는 다음 계산흐름이 반복적으로 나타난다.

`relation binding -> distributed intermediate coalition -> phase/action representation -> shared action-value geometry`

## 다음 실행: v0.5 relational OOD

### 환경

- canonical body embedding을 role token에서 제거
- entity는 episode별 random symbol로만 식별
- relation cue는 symbol을 통해 entity에 결합
- identity/beneficiary/concern 직접 report loss 없음
- shared policy/outcome heads 유지

### 단계

1. easy relational curriculum smoke test
2. fully randomized symbol/order 3-seed pilot
3. unseen symbol permutation test
4. held-out factor combination
5. action-matched counterfactual intervention
6. shuffled relation-pair control

### 성공 후보 기준

- in-distribution action >= .95
- unseen symbol/order action >= .85
- own-role + destination coalition effect > cross-role + destination
- action-matched factor intervention이 random/shuffled control보다 큼
- outcome/world MAE 증가 <= 사전 기준

### No-Go

- canonical binding 제거 후 shortcut policy로 회귀
- OOD에서 chance 수준
- causal specificity가 action matching 후 소멸

No-Go 시 강한 emergent factorization 주장을 중단하고, V16의 결과를 `distributed conditional routing with downstream shared control geometry`로 정리한다.
