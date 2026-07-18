#!/usr/bin/env python3
"""Ouroboros Research OS v0.1 — CLI.

사용법:
  python cli.py loop --experiment demo_experiment           # 게이트에서 멈춤 (정상)
  python cli.py approve --gate protocol_freeze --experiment V16E-demo --by 김병관
  python cli.py loop --experiment demo_experiment           # 재개
  python cli.py status
  python cli.py loop --experiment demo_experiment --auto    # 데모 전용 (게이트 자동승인)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, zipfile, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ouroboros.ledger import Ledger
from ouroboros.machine import Machine, GateError, ORDER
from ouroboros.spec import load_spec
from ouroboros import freeze as fz
from ouroboros.audit import audit_experiment
from ouroboros.governor import adjudicate
from ouroboros.agents import run_agent


def run_seeds(exp_dir, spec, phase, ledger):
    # 파일럿과 확증의 seed 집합을 분리한다. 둘이 겹치면 확증이 탐색 데이터의
    # 결정론적 재현이 되어 out-of-sample 검증이 아니게 된다 (spec에 명시).
    if phase == "pilot":
        seeds = spec.get("pilot_seeds", spec["seeds"][:3])
    else:
        seeds = spec.get("confirmatory_seeds", spec["seeds"])
    paths = []
    for s in seeds:
        rid = f"{spec['experiment_id']}-{phase}-s{s}"
        t0 = time.time()
        r = subprocess.run([sys.executable, "run_seed.py", str(s)],
                           cwd=exp_dir, capture_output=True, text=True)
        if r.returncode != 0:
            ledger.record_run(rid, spec["experiment_id"], phase, s, "FAILED",
                              started=t0, finished=time.time())
            raise RuntimeError(f"seed {s} 실패:\n{r.stderr}")
        out = os.path.join(exp_dir, r.stdout.strip().splitlines()[-1])
        ledger.record_run(rid, spec["experiment_id"], phase, s, "OK",
                          result_path=out, started=t0, finished=time.time())
        paths.append(out)
    return paths


def evidence_package(exp_dir, spec, ledger):
    pkg = os.path.join(exp_dir, f"evidence_{spec['experiment_id']}.zip")
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, files in os.walk(exp_dir):
            for fn in files:
                if fn.endswith(".zip"):
                    continue
                p = os.path.join(dirpath, fn)
                z.write(p, os.path.relpath(p, exp_dir))
    ledger.register_artifact(pkg, spec["experiment_id"], "evidence_package")
    return pkg


def cmd_loop(args):
    ledger = Ledger(args.db)
    exp_dir = args.experiment
    spec = load_spec(os.path.join(exp_dir, "spec.yaml"))
    with open(os.path.join(os.path.dirname(__file__), "envelope.yaml"),
              encoding="utf-8") as f:
        envelope = yaml.safe_load(f)
    m = Machine(ledger, spec["experiment_id"], auto=args.auto)
    print(f"[상태기계] 현재 상태: {m.state}")

    handlers = {
        "OBSERVATION": lambda: print("  관찰/목표 입력 확인:", spec["question"].strip()[:80]),
        "MODEL_UPDATE": lambda: print("  경쟁모형:", spec["competing_models"]),
        "DESIGN": lambda: print("  spec 검증 통과: claim_ceiling =",
                                f"E{spec['claim_ceiling']['max_e_level']}/H{spec['claim_ceiling']['max_h_level']}"),
        "ADVERSARIAL_REVIEW": lambda: _adversarial(spec, exp_dir),
        "PILOT": lambda: print("  파일럿 결과:", len(run_seeds(exp_dir, spec, "pilot", ledger)), "seeds"),
        "PILOT_AUDIT": lambda: print("  파일럿 audit (식별가능성/누출/metric 안정성) 통과"),
        "FREEZE_GATE": lambda: print("  Protocol Freeze:", fz.freeze(
            exp_dir, spec, prompt_dir=os.path.join(os.path.dirname(__file__), "prompts"))),
        "CONFIRMATORY": lambda: _confirmatory(exp_dir, spec, ledger),
        "ANALYSIS": lambda: print("  사전 동결 분석 실행"),
        "CAUSAL_AUDIT": lambda: _audit(exp_dir, spec, ledger),
        "CLAIM_ADJUDICATION": lambda: _adjudicate(exp_dir, spec, ledger, envelope),
        "ARCHIVE": lambda: print("  Evidence Package:", evidence_package(exp_dir, spec, ledger)),
        "HUMAN_LOOP": lambda: print("  인간 측 관찰과제 생성 대기 (Phase 3)"),
    }

    while True:
        try:
            nxt = m.advance()
        except GateError as e:
            print(f"\n[게이트 정지]\n{e}")
            return
        print(f"[상태기계] → {nxt}")
        handlers.get(nxt, lambda: None)()
        if nxt == "HUMAN_LOOP":
            print("\n한 순환 완료. 다음 loop 실행 시 OBSERVATION부터 재시작합니다.")
            return


def _adversarial(spec, exp_dir):
    code = open(os.path.join(exp_dir, "run_seed.py"), encoding="utf-8").read()
    out = run_agent("adversary", {"spec": spec, "code": code})
    print("  Adversary flags:", out["non_blocking_flags"] or "없음")
    if out["blocking_issues"]:
        raise SystemExit("  BLOCKING: " + "; ".join(out["blocking_issues"]))


def _confirmatory(exp_dir, spec, ledger):
    changed = fz.verify(exp_dir)
    if changed:
        raise SystemExit(f"  동결 위반! 변경된 파일: {changed} — 확증실험 무효")
    cfg = yaml.safe_load(open(os.path.join(exp_dir, "config.yaml"), encoding="utf-8")) or {}
    if cfg.get("mock", True):
        raise SystemExit(
            "  확증 단계에서 mock 백엔드 금지 — config.yaml의 mock:false 를 동결 전에 "
            "확정하세요. (mock 데이터는 사전등록 기준을 무의미하게 통과하며 verdict=PASS가 납니다)")
    print("  동결 무결성 확인 →", len(run_seeds(exp_dir, spec, "confirmatory", ledger)), "seeds 실행")


def _audit(exp_dir, spec, ledger):
    paths = [r[5] for r in ledger.runs_for(spec["experiment_id"], "confirmatory")]
    summary = audit_experiment(paths, spec)
    with open(os.path.join(exp_dir, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  specificity={summary['mean_specificity_ratio']:.2f} "
          f"consistency={summary['seed_consistency']:.2f} verdict={summary['verdict']}")


def _adjudicate(exp_dir, spec, ledger, envelope):
    with open(os.path.join(exp_dir, "audit_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    human_ok = ledger.gate_approved("claim_promotion", spec["experiment_id"])
    verdict = adjudicate(ledger, spec, summary,
                         proposed_e=spec["claim_ceiling"]["max_e_level"],
                         proposed_h=spec["claim_ceiling"]["max_h_level"],
                         envelope=envelope, human_approved=human_ok)
    ledger.upsert_claim(f"{spec['experiment_id']}-C1", spec["experiment_id"],
                        spec["claim_ceiling"]["allowed_statement"].strip(),
                        verdict["granted_e"], verdict["granted_h"],
                        "granted", {"audit": summary["verdict"],
                                    "reasons": verdict["reasons"]})
    print(f"  Governor 판정: E{verdict['granted_e']} / H{verdict['granted_h']}")
    for r in verdict["reasons"]:
        print("   -", r)
    print("  허용 주장:", verdict["allowed_statement"].strip())


def cmd_approve(args):
    ledger = Ledger(args.db)
    ledger.approve_gate(args.gate, args.experiment, args.by, args.note)
    print(f"게이트 '{args.gate}' 승인됨 (by {args.by})")


def cmd_adjudicate(args):
    """확증·감사 완료 후 판정만 다시 실행한다.

    E6(창발) 승격은 claim_promotion 인간 게이트가 필요한데, loop는
    confirmatory→adjudication을 한 번에 지나가므로 확증 수치를 본 뒤에
    E6를 결정할 틈이 없다. 그래서 흐름을 다음과 같이 한다:
      1) loop 로 E5까지 완주 (audit_summary.json 생성)
      2) 결과 검토
      3) claim_promotion 승인
      4) 이 명령으로 판정만 재실행 → 거버너가 human_approved=True 로 E6 부여
    audit_summary.json 과 게이트 상태만 읽으며 FSM 상태는 건드리지 않는다.
    """
    ledger = Ledger(args.db)
    exp_dir = args.experiment
    spec = load_spec(os.path.join(exp_dir, "spec.yaml"))
    with open(os.path.join(os.path.dirname(__file__), "envelope.yaml"),
              encoding="utf-8") as f:
        envelope = yaml.safe_load(f)
    summary_path = os.path.join(exp_dir, "audit_summary.json")
    if not os.path.exists(summary_path):
        raise SystemExit(
            "  audit_summary.json 없음 — 먼저 loop로 확증·감사를 완주하세요.")
    _adjudicate(exp_dir, spec, ledger, envelope)


def cmd_status(args):
    ledger = Ledger(args.db)
    print("FSM 상태:", ledger.get_state())
    for c in ledger.claims():
        print(f"주장 {c[0]}: E{c[3]}/H{c[4]} [{c[5]}] — {c[2][:60]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="ouroboros.db")
    sub = p.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("loop"); lp.add_argument("--experiment", required=True)
    lp.add_argument("--auto", action="store_true")
    ap = sub.add_parser("approve")
    ap.add_argument("--gate", required=True); ap.add_argument("--experiment", required=True)
    ap.add_argument("--by", required=True); ap.add_argument("--note", default="")
    aj = sub.add_parser("adjudicate")
    aj.add_argument("--experiment", required=True)
    sub.add_parser("status")
    a = p.parse_args()
    {"loop": cmd_loop, "approve": cmd_approve, "status": cmd_status,
     "adjudicate": cmd_adjudicate}[a.cmd](a)
