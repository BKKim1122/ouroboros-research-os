# experiments/v16_e/pre_dgx — DGX 확증 이전 파일럿들

윈도우 PC에서 수행한 V16-E 계열의 **DGX 이전 파일럿** 재현 레시피.
코드·프로토콜·리포트·설정·요약 통계·figure만 포함하며, 대용량 산물
(checkpoint.pt, training_state.pt, raw/)은 git에 넣지 않았다.
원자료가 필요하면 별도 백업/Release에서 복원한다.

## 갈래 (시간순)
- `meta_awareness_pilot/`        — V16 메타어텐션 파일럿 (인덱스상 V17 탐색자료로도 보존)
- `emergent_factorization_v0_4/` — V16-E 창발분해, pre-DGX v0.1→v0.4 (pilot_snapshots 포함)
- `e5_relational_binding/`       — 관계적 결합 파일럿
- `e6_surface_transition/`       — 표면전이 강건성

## 후속
이들의 DGX 확증본은 `experiments/v16_e/`의 c1_confirmatory 등(별도 커밋)에 있고,
사전학습 LLM으로 전환한 후속이 V17(`experiments/v17/`)이다.
전체 계보는 `docs/history/AI_ATTENTION_AWARENESS_LINEAGE.md` 참조.
