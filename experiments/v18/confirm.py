"""V18 확증 드라이버 — freeze 무결성 검증 → discover confirm → 거버너(discovery) 판정 → 원장 기록.

규율:
  · 확증 전 freeze 필수(protocol_freeze.json). verify()에 변경파일 있으면 결과 무효 → 중단.
  · mock 확증 금지(discover.py가 하드 차단). 여기서도 report.mock=True면 중단.
  · seed 분리(파일럿 0-2 / 확증 20-27)는 discover.py가 검사.
  · 등급 승격은 인간 게이트: E가 자율상한 초과면 cli approve --gate claim_promotion 필요.
  · 판정은 discover.py 봉인 기준 + 거버너가 낸다. 사람이 사후에 고쳐 쓰지 않는다.

사용:
  python confirm.py --by 김병관              # freeze 검증 → 실모델 확증 → 판정
  python confirm.py --report results/discover_report.json --by 김병관   # 기존 리포트로 재판정(검증용)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import yaml  # noqa: E402
from ouroboros.ledger import Ledger  # noqa: E402
from ouroboros import freeze as fz  # noqa: E402
from ouroboros.governor import adjudicate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", default="김병관", help="확증 실행 주체(감사 기록용)")
    ap.add_argument("--report", default=None, help="기존 discover_report.json으로 재판정(검증용)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--db", default=os.path.join(ROOT, "ouroboros.db"))
    args = ap.parse_args()

    spec = yaml.safe_load(open(os.path.join(HERE, "spec.yaml"), encoding="utf-8"))
    exp_id = spec["experiment_id"]
    ledger = Ledger(args.db)

    # ── 1. freeze 무결성 (METHODS_STANDARD §2) ──
    freeze_path = os.path.join(HERE, fz.FREEZE_FILE)
    if not os.path.exists(freeze_path):
        print("❌ freeze 없음. 확증 전 freeze 필수:  bash experiments/v18/run_v18_freeze.sh")
        sys.exit(2)
    changed = fz.verify(HERE)
    if changed:
        print("❌ freeze 무결성 실패 — 동결 이후 변경된 파일:")
        for c in changed:
            print("   -", c)
        print("   확증 결과는 무효. 변경을 되돌리거나 재-freeze 후 재실행.")
        ledger.event(args.by, "confirm_aborted", {"reason": "freeze_verify_failed", "changed": changed})
        sys.exit(2)
    print("✓ freeze 무결 (동결 이후 코드/spec 변경 없음)")

    # ── 2. 확증 실행 또는 기존 리포트 로드 ──
    report_path = os.path.join(HERE, "results", "discover_report.json")
    if args.report:
        report_path = args.report if os.path.isabs(args.report) else os.path.join(HERE, args.report)
        print(f"· 기존 리포트로 재판정: {os.path.relpath(report_path, HERE)}")
    else:
        print(f"· 확증 실행: discover.py --mode confirm --seeds {spec['confirmatory_seeds']}")
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "discover.py"),
             "--mode", "confirm", "--device", args.device],
            cwd=HERE, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            print("❌ discover.py 확증 실패:\n" + r.stderr)
            ledger.event(args.by, "confirm_aborted", {"reason": "discover_failed"})
            sys.exit(2)

    report = json.load(open(report_path, encoding="utf-8"))
    if report.get("mock"):
        print("❌ 리포트가 mock — 확증 무효. 실모델로 재실행.")
        sys.exit(2)

    # ── 3. audit_summary(endpoint=discovery) 구성 → 거버너 판정 ──
    audit_summary = {
        "endpoint": "discovery",
        "verdict": report.get("verdict_sealed"),
        "struct_frac": report.get("struct_frac", 0.0),
        "ma1_frac": report.get("ma1_frac", 0.0),
        "lexical_any": report.get("lexical_any", False),
        "consistency_min": spec["stats"]["consistency_min"],
        "merge_evidence": report.get("merge_evidence"),
    }
    envelope = {"promote_cap_e": 3}   # 자율상한 E3. 천장(E4)까지의 승격은 인간 게이트.
    human_ok = ledger.gate_approved("claim_promotion", exp_id)
    result = adjudicate(ledger, spec, audit_summary,
                        proposed_e=spec["claim_ceiling"]["max_e_level"], proposed_h=0,
                        envelope=envelope, human_approved=human_ok)

    # ── 4. 판정 기록 (원장 + verdict json) ──
    model = result["discovery_model"]
    claim_text = {
        "MA1": "사전학습 LLM의 자기관련 변이 공간이 우리 4요인(identity·beneficiary·privilege·concern) 경계로 조직된다.",
        "MA2": "자기관련 구조는 실재하나 우리 4분할이 아니라 더 적은 축(예: 이득∪이해관계 병합)으로 조직된다. 4번째 경계는 부분적 부과물(H-A 약화, 반증 아님).",
        "MA3": "자기관련 변이의 군집 구조가 null과 구분되지 않는다.",
        "MA1_blocked_lexical": "군집이 요인이 아닌 템플릿 어휘 골격을 따른다(MA1 불가).",
    }.get(model, model)
    claim_id = f"{exp_id}-discovery"
    ledger.upsert_claim(claim_id, exp_id, claim_text,
                        result["granted_e"], result["granted_h"], model,
                        evidence={"audit_summary": audit_summary, "reasons": result["reasons"],
                                  "merge_evidence": report.get("merge_evidence")})
    ledger.event(args.by, "confirm_adjudicated",
                 {"model": model, "E": result["granted_e"], "H": result["granted_h"],
                  "human_gate": human_ok})

    verdict_out = {
        "experiment_id": exp_id, "confirmed_by": args.by, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "freeze_verified": True, "mock": False,
        "discovery_model": model,
        "granted": {"E": result["granted_e"], "H": result["granted_h"]},
        "human_gate_approved": human_ok,
        "reasons": result["reasons"],
        "claim_text": claim_text,
        "audit_summary": audit_summary, "merge_evidence": report.get("merge_evidence"),
        "allowed_statement": result["allowed_statement"],
        "forbidden_statements": result["forbidden_statements"],
    }
    outp = os.path.join(HERE, "results", "confirm_verdict.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(verdict_out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ── 5. 콘솔 요약 ──
    print("\n" + "=" * 60)
    print(f"[확증 판정] discovery_model = {model}")
    print(f"  등급: E{result['granted_e']} / H{result['granted_h']}   (천장 E{spec['claim_ceiling']['max_e_level']}/H{spec['claim_ceiling']['max_h_level']})")
    if not human_ok and spec["claim_ceiling"]["max_e_level"] > envelope["promote_cap_e"]:
        print(f"  ⚠ 인간 게이트 미승인 — E는 자율상한 E{envelope['promote_cap_e']}로 제한.")
        print(f"     천장까지 기록하려면: python cli.py approve --gate claim_promotion --experiment {exp_id} --by {args.by}")
        print("     승인 후 confirm.py 재실행.")
    for r in result["reasons"]:
        print("   ·", r)
    me = report.get("merge_evidence") or {}
    if me:
        print(f"  병합증거(진단): bene/conc cos={me.get('bene_conc_cos_mean')} "
              f"/ modal병합 {me.get('modal_merge_frac')} / 3라벨≥4라벨 {me.get('ari_merged_ge_split_frac')}")
    print(f"  claim: {claim_text}")
    print(f"\n  기록: results/confirm_verdict.json + 원장(claims:{claim_id})")
    print("=" * 60)


if __name__ == "__main__":
    main()
