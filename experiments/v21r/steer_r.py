"""V21R: 역방향 steering 확증 — V21 이상치(D<0)를 사전등록 가설로 승격해 검증.

V21과의 차이:
  · 가설이 D<0 (사전등록). D>0이 나와도 여기선 '가설 미지지'일 뿐.
  · 확증 seed(20-27) + 복수 모델(3개, 랩 2곳) — V21의 약점(단일모델·seed 비독립) 보강.
  · α 고정(0.01, 봉인). robust α는 보고만.
  · freeze 필수. mock 금지.

사용: python steer_r.py --mode confirm            # freeze 후
      python steer_r.py --mode pilot --models qwen25_1p5b --null-R 100   # 배선 점검
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from steer_core import SteerHF, build_self_axis, run, save_report  # noqa: E402
# steer_core.py = v21/steer.py의 벤더 사본. freeze 자기완결(외부 import는 무결성 구멍)을 위해 동봉.
from ouroboros import freeze as fz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--null-R", type=int, default=None)
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    args = ap.parse_args()
    import yaml
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    seal = spec["steer"]
    R = args.null_R or seal["null_R"]

    if args.mode == "confirm":
        fp = os.path.join(HERE, fz.FREEZE_FILE)
        if not os.path.exists(fp):
            print("❌ freeze 없음:  bash experiments/v21r/run_v21r_freeze.sh"); sys.exit(2)
        changed = fz.verify(HERE)
        if changed:
            print("❌ freeze 무결성 실패 — 변경 파일:", changed); sys.exit(2)
        print("✓ freeze 무결")

    seeds = spec["pilot_seeds"] if args.mode == "pilot" else spec["confirmatory_seeds"]
    models = spec["models"]
    if args.models:
        models = [m for m in models if m["tag"] in args.models]
    cons = spec["stats"]["consistency_min"]

    grid = []
    for m in models:
        tag = m["tag"]
        print(f"\n▶ {tag} …", flush=True)
        t0 = time.time()
        try:
            be = SteerHF(m["name"], device=(None if args.device == "auto" else args.device))
        except Exception as e:
            grid.append({"tag": tag, "status": "load_failed", "error": str(e)[:150]})
            print(f"   ❌ {str(e)[:120]}"); continue
        layer = max(1, int(be.n_layers * seal["layer_frac"]))
        import torch  # noqa
        scale = float(np.linalg.norm(be.hidden(["I made the decision."], layer)[0]))
        a = seal["alpha_rel"] * scale
        rows = []
        for s in seeds:
            w = build_self_axis(be, layer, s)
            r = run(be, s, layer, a, R, seal, w)
            r["reversed"] = bool(r["D_directional"] < r["null_gate_lo"])
            r["contam"] = "CONTAMINATED" in r["verdict"]
            rows.append(r)
            print(f"   s{s}: D {r['D_directional']:+.3f} vs 하한 {r['null_gate_lo']:+.3f} "
                  f"→ {'REVERSED' if r['reversed'] else 'no'}"
                  f"{' [오염]' if r['contam'] else ''}")
        clean = [r for r in rows if not r["contam"]]
        rf = round(float(np.mean([r["reversed"] for r in clean])), 4) if clean else None
        grid.append({"tag": tag, "status": "ok", "layer": layer, "alpha": a,
                     "reversed_frac": rf, "n_clean": len(clean), "per_seed": rows,
                     "secs": round(time.time() - t0, 1)})
        print(f"   → reversed seed비율 {rf}  [{grid[-1]['secs']}s]")

    ok = [g for g in grid if g.get("status") == "ok" and g.get("reversed_frac") is not None]
    model_pass = [g["tag"] for g in ok if g["reversed_frac"] >= cons]
    rep = round(len(model_pass) / len(ok), 4) if ok else None
    confirmed = rep is not None and rep >= spec["replication_min"]
    verdict = ("REVERSED_CONFIRMED — 역방향 인과가 seed·모델에서 재현"
               if confirmed else
               "NOT_REPLICATED — V21 NO-HANDLE 유지, 이상치는 노이즈/모델특이")

    print("\n" + "=" * 64)
    print(f"[V21R {'확증' if args.mode=='confirm' else '파일럿'} 판정] {verdict}")
    print(f"  모델별 reversed: " + ", ".join(f"{g['tag']}={g['reversed_frac']}" for g in ok))
    print(f"  재현 모델 {model_pass} ({rep} vs {spec['replication_min']})")
    if confirmed and args.mode == "confirm":
        print("  ⚠ E4 방향 주장은 인간 게이트 필요 (cli approve --gate claim_promotion)")
    print("=" * 64)
    save_report("steer_r_report", {"mode": args.mode, "seeds": seeds, "null_R": R,
                                    "grid": grid, "replication": rep, "verdict": verdict})
    if args.mode == "pilot":
        print("※ 파일럿은 승격 없음.")


if __name__ == "__main__":
    main()
