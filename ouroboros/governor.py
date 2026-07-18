"""Claim Governor — 규칙 기반 코드. LLM 아님.

2축 주장 격자:

기계론 축 (E):
  E0 입력에 존재            E1 해독 가능(probe)
  E2 행동 예측에 사용        E3 개입 시 변화
  E4 선택적 인과효과         E5 OOD 유지
  E6 자연학습에서 창발       E7 모델·seed 간 전이

대응 축 (H) — E축과 독립:
  H0 비유 수준              H1 구조적 유사성 명시
  H2 인간 측 단방향 예측 적중  H3 양방향 판별예측 검증

불변식:
  - LLM 리뷰는 E/H를 올릴 수 없다 (이 모듈만 승격 가능).
  - envelope의 promote_cap 초과 승격은 인간 게이트 필요.
  - E축 승격이 H축을 자동으로 올리지 않는다 (축 독립).
"""
from __future__ import annotations
from .ledger import Ledger

E_REQUIREMENTS = {
    4: dict(min_ratio_key="effect_size_min", need_consistency=0.8),
    5: dict(need_ood=True),
    7: dict(need_transfer=True),
}


def adjudicate(ledger: Ledger, spec: dict, audit_summary: dict,
               proposed_e: int, proposed_h: int,
               envelope: dict, human_approved: bool = False) -> dict:
    ceiling = spec["claim_ceiling"]
    cap_e = min(ceiling["max_e_level"], envelope.get("promote_cap_e", 5))
    reasons, granted_e = [], proposed_e

    if audit_summary.get("endpoint") == "emergence":
        # 창발 주장: 행렬 specificity 규칙 대신 사전 등록 기준 충족 여부로 판정
        if audit_summary["verdict"] != "PASS":
            granted_e = min(granted_e, 1)
            reasons.append("창발 기준 미충족 → E1(해독 가능)로 제한")
        if granted_e > ceiling["max_e_level"]:
            granted_e = ceiling["max_e_level"]
            reasons.append(f"claim_ceiling(E{ceiling['max_e_level']}) 적용")
        if granted_e > cap_e and not human_approved:
            granted_e = cap_e
            reasons.append(f"자율범위 상한(E{cap_e}) — E{proposed_e} 승격은 인간 승인 필요 "
                           "(cli approve --gate claim_promotion)")
        granted_h = min(proposed_h, ceiling["max_h_level"])
        return {"granted_e": granted_e, "granted_h": granted_h,
                "allowed_statement": ceiling["allowed_statement"],
                "forbidden_statements": ceiling["forbidden_statements"],
                "reasons": reasons}

    if audit_summary["verdict"] != "PASS":
        granted_e = min(granted_e, 3)
        reasons.append("audit FAIL → E3 이하로 제한")

    req = E_REQUIREMENTS.get(4)
    if granted_e >= 4:
        if audit_summary["mean_specificity_ratio"] < spec["stats"]["effect_size_min"]:
            granted_e = 3; reasons.append("specificity 미달 → E3")
        elif audit_summary["seed_consistency"] < req["need_consistency"]:
            granted_e = 3; reasons.append("seed 일관성 < 0.8 → E3")

    if granted_e >= 5 and not audit_summary.get("ood_pass"):
        granted_e = 4; reasons.append("OOD 증거 없음 → E4")

    if granted_e > ceiling["max_e_level"]:
        granted_e = ceiling["max_e_level"]
        reasons.append(f"claim_ceiling(E{ceiling['max_e_level']}) 적용")

    if granted_e > cap_e and not human_approved:
        granted_e = cap_e
        reasons.append(f"자율범위 상한(E{cap_e}) — 초과 승격은 인간 승인 필요")

    granted_h = min(proposed_h, ceiling["max_h_level"])
    if granted_h > 0 and not audit_summary.get("human_prediction_registered"):
        granted_h = 0
        reasons.append("사전 봉인된 인간 측 예측 없음 → H0")

    return {
        "granted_e": granted_e, "granted_h": granted_h,
        "allowed_statement": ceiling["allowed_statement"],
        "forbidden_statements": ceiling["forbidden_statements"],
        "reasons": reasons,
    }
