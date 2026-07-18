# experiments/v16_e — V16-E 세대 (2026-07-12, DGX 확증)

이전 세션에서 edgexpert(DGX Spark)에서 수행한 V16-E 계열 실험의 **재현 레시피**.
코드·스펙·문서·대표 리포트만 포함하며, 대용량 실행 산물(원자료·체크포인트·
tarball·가상환경)은 git에 넣지 않는다(별도 백업/GitHub Release로 보관).

## 갈래
- `c1_confirmatory/`          — C1 본확증
- `e8_identifiable_composition/`
- `d1a_uniform_extension/`
- `d1b_causal_recheck/`
- `d1b_r1_metric_correction/`

## 재현
각 갈래의 README / PROVENANCE / FROZEN_FILES_SHA256 참조. 원자료가 필요하면
결과 tarball(별도 보관)을 내려받아 각 폴더 옆에 풀어 사용한다.

## 현행 연구와의 관계
V16-E 세대의 후속으로 V17(사전학습 LLM 자기표상 자연발생)이 진행 중이며,
설계 경위는 `docs/history/2026-07_v17_decision_log.md` 참조.
