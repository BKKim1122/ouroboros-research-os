"""V20 확증 드라이버 — freeze 무결성 검증 → discover_prompts confirm → 거버너(prompt_diversity) 판정 → 원장 기록.

규율(V18 confirm과 동일):
  · 확증 전 freeze 필수. verify() 변경 감지 시 무효 → 중단.
  · mock 확증 금지(discover_prompts.py 하드 차단 + 여기서도 이중 확인).
  · seed 분리(파일럿 0-2 / 확증 20-27).
  · 판정은 봉인 기준 + 거버너. 사람이 사후 수정 금지.
  · orig·para 두 뱅크가 판정 주지표(ko는 보고만).

사용:
  python confirm_v20.py --by 김병관
  python confirm_v20.py --report results/discover_prompts_report_XXX.json --by 김병관   # 재판정
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
    ap.add_argument("--by", default="김병관")
    ap.add_argument("--report", default=None, help="기존 리포트로 재판정(검증용)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--db", default=os.path.join(ROOT, "ouroboros.db"))
    args = ap.parse_args()

    spec = yaml.safe_load(open(os.path.join(HERE, "spec.yaml"), encoding="utf-8"))
    exp_id = spec["experiment_id"]
    ledger = Ledger(args.db)

    # ── 1. freeze 무결성 ──
    freeze_path = os.path.join(HERE, fz.FREEZE_FILE)
    if not os.path.exists(freeze_path):
        print("❌ freeze 없음. 확증 전 freeze 필수:  bash experiments/v20/run_v20_freeze.sh")
        sys.exit(2)
    changed = fz.verify(HERE)
    if changed:
        print("❌ freeze 무결성 실패 — 동결 이후 변경된 파일:")
        for c in changed:
            print("   -", c)
        ledger.event(args.by, "confirm_aborted",
                     {"exp": exp_id, "reason": "freeze_verify_failed", "changed": changed})
        sys.exit(2)
    print("✓ freeze 무결 (동결 이후 코드/spec 변경 없음)")

    # ── 2. 확증 실행 또는 기존 리포트 ──
    if args.report:
        report_path = args.report if os.path.isabs(args.report) else os.path.join(HERE, args.report)
        print(f"· 기존 리포트로 재판정: {os.path.relpath(report_path, HERE)}")
    else:
        print(f"· 확증 실행: discover_prompts.py --mode confirm (banks: orig para ko, "
              f"seeds {spec['confirmatory_seeds']})")
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, "discover_prompts.py"),
                            "--mode", "confirm", "--device", args.device],
                           cwd=HERE)
        if r.returncode != 0:
            print("❌ 확증 실행 실패"); sys.exit(r.returncode)
        print(f"  (확증 소요 {time.time()-t0:.0f}s)")
        report_path = os.path.join(HERE, "results", "discover_prompts_report_latest.json")

    report = json.load(open(report_path, encoding="utf-8"))
    if report.get("mock"):
        print("❌ mock 리포트로는 확증 판정 불가."); sys.exit(2)
    if report.get("mode") != "confirm":
        print("❌ confirm 모드 리포트가 아님 (파일럿으로는 승격 불가)."); sys.exit(2)

    # ── 3. audit_summary → 거버너 ──
    reps = report.get("bank_merge_replication", {})
    audit = {"endpoint": "prompt_diversity",
             "replication_min": report.get("replication_min",
                                           spec.get("replication_min", 0.75)),
             "orig_replication": reps.get("orig"),
             "para_replication": reps.get("para"),
             "ko_replication": reps.get("ko"),
             "verdict_sealed": report.get("verdict")}
    envelope = {"promote_cap_e": 3}  # 자율상한 E3 (V20도 관찰·기술 — 개입 없음)
    verdict = adjudicate(ledger, spec, audit,
                         proposed_e=spec["claim_ceiling"]["max_e_level"],
                         proposed_h=0, envelope=envelope, human_approved=False)

    out = {"experiment_id": exp_id, "confirmed_by": args.by,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "freeze_verified": True, "mock": False,
           "prompt_model": verdict["prompt_model"],
           "granted": {"E": verdict["granted_e"], "H": verdict["granted_h"]},
           "reasons": verdict["reasons"], "audit_summary": audit,
           "allowed_statement": verdict["allowed_statement"],
           "forbidden_statements": verdict["forbidden_statements"]}
    ledger.claim(f"{exp_id}-prompt_diversity", args.by, json.dumps(out, ensure_ascii=False))

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    vp = os.path.join(outdir, f"confirm_verdict_{ts}.json")
    json.dump(out, open(vp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(out, open(os.path.join(outdir, "confirm_verdict_latest.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"[확증 판정] prompt_model = {verdict['prompt_model']}")
    print(f"  등급: E{verdict['granted_e']} / H{verdict['granted_h']}")
    for r_ in verdict["reasons"]:
        print(f"   · {r_}")
    print(f"  뱅크 재현율: orig {audit['orig_replication']} / para {audit['para_replication']}"
          f" / ko {audit['ko_replication']}")
    print(f"  기록: {os.path.relpath(vp, HERE)} + 원장(claims:{exp_id}-prompt_diversity)")
    print("=" * 60)


if __name__ == "__main__":
    main()
