"""V20 확증 판정기 (동결 외부) — confirm_v20.py의 원장 API 버그(ledger.claim→없음)로
판정·기록 단계만 미완인 상황을, 동결을 건드리지 않고 마무리한다.

정당성: 확증 데이터는 freeze 무결 검증 하에 이미 생성됨(discover_prompts confirm,
리포트 저장 완료). 본 스크립트는 (1) freeze 무결 재확인 (2) 그 리포트를 읽어
(3) 봉인 기준 그대로 거버너 판정 (4) 올바른 API(upsert_claim)로 원장 기록.
기준·데이터·분석은 일절 불변 — 기록 단계의 API만 교정. (결정로그 기록 대상)

사용: python tools/adjudicate_v20.py --report experiments/v20/results/discover_prompts_report_20260726_032507.json --by 김병관
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import yaml  # noqa: E402
from ouroboros.ledger import Ledger  # noqa: E402
from ouroboros import freeze as fz  # noqa: E402
from ouroboros.governor import adjudicate  # noqa: E402

V20 = os.path.join(ROOT, "experiments", "v20")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--by", default="김병관")
    ap.add_argument("--db", default=os.path.join(ROOT, "ouroboros.db"))
    args = ap.parse_args()

    spec = yaml.safe_load(open(os.path.join(V20, "spec.yaml"), encoding="utf-8"))
    exp_id = spec["experiment_id"]
    ledger = Ledger(args.db)

    # 1) freeze 무결 재확인 (데이터 생성 시점과 동일 상태임을 증빙)
    changed = fz.verify(V20)
    if changed:
        print("❌ freeze 무결성 실패 — 판정 불가:", changed); sys.exit(2)
    print("✓ freeze 무결")

    # 2) 확증 리포트 로드 + 해시(출처 고정)
    rp = args.report if os.path.isabs(args.report) else os.path.join(ROOT, args.report)
    report = json.load(open(rp, encoding="utf-8"))
    rhash = hashlib.sha256(open(rp, "rb").read()).hexdigest()[:16]
    if report.get("mock") or report.get("mode") != "confirm":
        print("❌ confirm 모드의 실데이터 리포트가 아님."); sys.exit(2)
    print(f"· 리포트: {os.path.relpath(rp, ROOT)} (sha256:{rhash})")

    # 3) 거버너 판정 (봉인 기준 그대로)
    reps = report.get("bank_merge_replication", {})
    audit = {"endpoint": "prompt_diversity",
             "replication_min": report.get("replication_min", spec.get("replication_min", 0.75)),
             "orig_replication": reps.get("orig"),
             "para_replication": reps.get("para"),
             "ko_replication": reps.get("ko"),
             "verdict_sealed": report.get("verdict"),
             "report_sha256_16": rhash}
    human_ok = ledger.gate_approved("claim_promotion", exp_id)
    result = adjudicate(ledger, spec, audit,
                        proposed_e=spec["claim_ceiling"]["max_e_level"], proposed_h=0,
                        envelope={"promote_cap_e": 3}, human_approved=human_ok)

    # 4) 원장 기록 (올바른 API)
    model = result["prompt_model"]
    claim_text = {
        "PROMPT_DEPENDENT": "beneficiary∪concern 병합은 모델의 의미 조직이 아니라 원본 프롬프트 "
                            "세트의 성질이다 — 영어 재표현(para)에서 병합이 소멸(확증 seed 20–27).",
        "PROMPT_INVARIANT": "병합은 표현을 넘는 모델 성질이다(orig·para 재현).",
    }.get(model, model)
    ledger.upsert_claim(f"{exp_id}-prompt_diversity", exp_id, claim_text,
                        result["granted_e"], result["granted_h"], model,
                        evidence={"audit_summary": audit, "reasons": result["reasons"]})
    ledger.event(args.by, "confirm_adjudicated",
                 {"exp": exp_id, "model": model, "E": result["granted_e"],
                  "H": result["granted_h"], "human_gate": human_ok,
                  "note": "adjudicated via tools/adjudicate_v20.py (recording-API fix; criteria unchanged)"})

    out = {"experiment_id": exp_id, "confirmed_by": args.by,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "freeze_verified": True,
           "prompt_model": model,
           "granted": {"E": result["granted_e"], "H": result["granted_h"]},
           "human_gate_approved": human_ok, "reasons": result["reasons"],
           "audit_summary": audit,
           "allowed_statement": result["allowed_statement"],
           "forbidden_statements": result["forbidden_statements"]}
    outdir = os.path.join(V20, "results"); os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    vp = os.path.join(outdir, f"confirm_verdict_{ts}.json")
    json.dump(out, open(vp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(out, open(os.path.join(outdir, "confirm_verdict_latest.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"[확증 판정] prompt_model = {model}")
    print(f"  등급: E{result['granted_e']} / H{result['granted_h']}")
    for r in result["reasons"]:
        print(f"   · {r}")
    print(f"  뱅크 재현율: orig {audit['orig_replication']} / para {audit['para_replication']}")
    print(f"  기록: {os.path.relpath(vp, ROOT)} + 원장(claims:{exp_id}-prompt_diversity)")
    print("=" * 60)


if __name__ == "__main__":
    main()
