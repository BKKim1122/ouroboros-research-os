# AI Attention–Awareness Research Repository

업데이트: 2026-07-12

## 1. 연구 목적

AI attention 및 self-related routing을 분석하여 인간의 ‘봄/알아차림’과 비교 가능한 기능적 계산구조를 구축하고, AI 안전·에이전트 신뢰성·메타인지 및 향후 인간 행동/신경과학 검증으로 확장한다.

## 2. 동결 기준선

- 최종 논문: `identity_privilege_concern_final_manuscripts_v4.1.zip`
- 근거 패키지: `identity_privilege_concern_evidence_package_v1.4.zip`
- 투고 패키지: `identity_privilege_concern_submission_kit_v4.1.zip`
- 상태: V1–V15 동결. V16 결과로 자동 수정하지 않음.

## 3. V1–V15 핵심 계보

- V1–V9: 측정 실패·우회경로·정보소실과 성능저하 혼동 수정
- V10: gain calibration
- V11: 규모·구조·OOD 일반화
- V12: B/S/V 분산 causal coalition
- V13: 내부 사전고정 확증 + generic conditional-routing 경계 확인
- V14: identity attribution과 beneficiary priority의 설계된 이중해리
- V15: relative privilege와 operational concern의 분리; privilege 0에서 대칭적 보호와 무관심 구분

## 4. 현재 V16 고정 방향

**V16-E: Emergent Factorization**

명시적인 identity/beneficiary/concern 전용 slot·gain·route 없이 통합 Transformer가 행동과 세계예측을 학습하는 과정에서 해당 계산축을 자연학습하고 인과적으로 분화하는지 검증한다.

### 현재 파일럿 상태

- unified behavior-only 및 multitask 모델 학습 완료
- 동일분포 성능과 world prediction은 높음
- factor 정보는 내부표현에서 해독됨
- 저차원 counterfactual subspace는 정책을 이동시킴
- 그러나 cross-factor control에서 공통 action-value manifold 중첩 확인
- 강한 factor-specific causal factorization은 아직 미확인

### DGX 전 다음 작업

1. layer/token별 표현지도
2. same-action/different-factor 및 same-factor/different-action 통제
3. 동적 식별환경 설계
4. held-out 조합·symbol-only·transition OOD
5. own-factor vs cross-factor causal specificity 확보
6. Gate 통과 후 DGX 확증 protocol/source snapshot 동결

## 5. 보존된 후속 트랙

- V17: meta-access, awareness without compulsory correction, policy decoupling
- V18: identity/ownership/agency/control의 embodied extension
- V19: 공개 LLM·장기 에이전트 전이
- 인간 단계: 행동과제 → EEG/생리 → 인과개입/neurofeedback

## 6. 주요 파일

### V1–V15 동결본
- `identity_privilege_concern_evidence_package_v1.4.zip`
- `identity_privilege_concern_final_manuscripts_v4.1.zip`
- `identity_privilege_concern_submission_kit_v4.1.zip`

### V16 인수·이전 파일럿
- `V16_handoff_package_v1.1.zip`
- `v16_attention_meta_awareness_pilot_v0_1.zip` (V17 탐색자료로 보존)

### 현재 V16-E
- `v16_emergent_factorization_pre_dgx_v0_3.zip`
- `V16_PRE_DGX_MASTER_PLAN_KO_v1_0.md`
- `V16E_PRE_DGX_EXECUTION_PLAN_KO_v1_1.md`
- `V16E_PRE_DGX_PROGRESS_REPORT_KO_v0_3.md`

## 7. 운영 원칙

- Pilot과 confirmatory 분리
- 확증 전 exact frozen source snapshot + protocol hash 보존
- decodability와 causal use 분리
- 상류 정보와 하류 policy representation 분리
- random/shuffled/cross-factor control 필수
- 실패 seed 및 부정 결과 공개
- 현재 v4.1 논문은 별도 승인 전 수정 금지
