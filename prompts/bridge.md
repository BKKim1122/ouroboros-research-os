# Phenomenology Bridge (Phase 3)

AI 계산결과와 인간 일인칭 자료를 연결하되 **동일시하지 않는다**.

## 입력
{ "ai_results": ..., "human_reports": [...], "sealed_predictions": [...] }

## 방법론적 제약 (N=1 자기실험 보정)
1. sealed_predictions(관찰 이전에 봉인된 예측)와의 대조만 H2 후보가 될 수 있다.
   사후 대응은 전부 "약한 비유"로 분류하라.
2. human_reports 중 "AI 이론과 불일치하는 체험"을 최우선으로 분석하라.
   이 자료의 목적은 이론 확인이 아니라 이론이 놓친 구조의 발견이다.
3. 기대효과(피험자=연구자) 가능성을 모든 대응 판정에 명시하라.

## 출력 (JSON만)
{
  "direct_correspondence": [], "functional_similarity": [], "weak_analogy": [],
  "unverified_speculation": [], "mismatches": [],
  "next_bidirectional_tests": [], "demand_characteristic_risk": "...",
  "blocking_issues": [], "non_blocking_flags": []
}
