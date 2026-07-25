"""V20: 프롬프트 다양성 — ben∪conc 병합이 문장/언어를 바꿔도 재현되나.

V19 병합(12/12 모델)의 마지막 구멍: 영어·단일저자 프롬프트 하나였다. 여기서 같은 4요인·같은
self/other 구조를 (orig 원본) / (para 영어 다른표현) / (ko 한국어)로 다시 만들어, 모델×뱅크
격자에서 cross-transfer 병합을 잰다.

  · para(영어 재표현) = 핵심 통제. 같은 언어인데 병합이 사라지면 → '이 문장들의 성질'(주장 후퇴).
  · ko(한국어)        = 보너스. 영어전용 모델은 self_acc 게이트에서 걸러짐(능력 부족 ≠ 병합 없음).

판정(결과 전 봉인):
  PROMPT_INVARIANT : 병합이 orig AND para에서 재현(self_acc통과 모델 >= replication_min) → 모델 성질
  PROMPT_DEPENDENT : orig엔 있으나 para에서 무너짐 → 원본 문장 특유
  (ko는 별도 보고: 다국어 능력 있는 모델에서 병합 유지되나)
분석·null 기준은 V19와 동일(cross_transfer_merge 재사용). mock은 병합 미확증이 정상.
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V19 = os.path.join(HERE, "..", "v19")
sys.path.insert(0, V19)
from discover_multi import (self_acc_gate, self_acc_obs, cross_transfer_merge,  # noqa: E402
                            HFBackend, MockBackend)
sys.path.insert(0, HERE)
from prompts_v20 import get_bank_builder, FACTORS4  # noqa: E402


def collect_poles_bank(be, seed, bankfn):
    bank = bankfn(seed); poles = {}
    for f in FACTORS4:
        pairs = bank[f]["A_train"] + bank[f]["B_train"]
        S = be.acts([p for p, _ in pairs], f, +1)
        O = be.acts([n for _, n in pairs], f, -1)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-8)
        poles[f] = (S, O)
    return poles


def make_mock_bank(seed, bankfn):
    bank = bankfn(seed); truth = {}
    for f in FACTORS4:
        for p, n in bank[f]["A_train"] + bank[f]["B_train"]:
            truth[p] = (f, +1); truth[n] = (f, -1)
    return MockBackend(seed, truth=truth)


def pick_best_layer_bank(be, layer_fracs, bankfn, seed):
    n = be.model.config.num_hidden_layers; scan = []
    for fr in layer_fracs:
        be.layer = max(1, int(n * fr))
        sa, _, _ = self_acc_obs(collect_poles_bank(be, seed, bankfn))
        scan.append({"layer_frac": fr, "layer": be.layer, "self_acc": round(float(sa), 4)})
    return max(scan, key=lambda r: r["self_acc"]), scan


def analyze_cell(mspec, bankname, spec, seeds, device, mock, null_R):
    name, tag = mspec["name"], mspec["tag"]
    bankfn = get_bank_builder(bankname)
    ct = spec["cross_transfer"]
    try:
        if mock:
            best = {"layer": "mock"}; be = None
        else:
            be = HFBackend(name, spec["layer_fracs"][0],
                           device=(None if device == "auto" else device))
            best, _ = pick_best_layer_bank(be, spec["layer_fracs"], bankfn, seeds[0])
            be.layer = best["layer"]
    except Exception as e:
        return {"tag": tag, "bank": bankname, "status": "load_failed", "error": str(e)[:150]}

    rows = []
    for s in seeds:
        be_s = make_mock_bank(s, bankfn) if mock else be
        poles = collect_poles_bank(be_s, s, bankfn)
        sa, sa_gate, sa_pass = self_acc_gate(poles, min(null_R, 200), s, ct["self_acc_null_pct"])
        row = {"seed": s, "self_acc": sa, "self_acc_pass": sa_pass}
        if sa_pass:
            row.update(cross_transfer_merge(poles, null_R, s,
                                            ct["transfer_null_pct"], ct["specificity_null_pct"]))
        rows.append(row)

    passed = [r for r in rows if r["self_acc_pass"]]
    out = {"tag": tag, "bank": bankname, "status": "ok", "best_layer": best,
           "self_acc_mean": round(float(np.mean([r["self_acc"] for r in rows])), 4),
           "self_acc_pass_frac": round(float(np.mean([r["self_acc_pass"] for r in rows])), 4),
           "per_seed": rows}
    if passed:
        out.update({
            "t_ben_conc_mean": round(float(np.mean([r["t_ben_conc"] for r in passed])), 4),
            "merge_specificity_mean": round(float(np.mean([r["merge_specificity"] for r in passed])), 4),
            "merge_confirmed_frac": round(float(np.mean([r["merge_confirmed"] for r in passed])), 4),
        })
    return out




def save_report(outdir, stem, payload):
    import time as _t
    os.makedirs(outdir, exist_ok=True)
    ts = _t.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outdir, f"{stem}_{ts}.json")
    json.dump(payload, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(payload, open(os.path.join(outdir, f"{stem}_latest.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--banks", nargs="+", default=None)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--null-R", type=int, default=None)
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    args = ap.parse_args()

    import yaml
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    if args.mock and args.mode == "confirm":
        print("❌ 확증에서 mock 금지."); sys.exit(2)
    null_R = args.null_R or spec["null_R"]
    seeds = spec["pilot_seeds"] if args.mode == "pilot" else spec["confirmatory_seeds"]
    models = spec["models"];  banks = args.banks or spec["banks"]
    if args.models:
        models = [m for m in models if m["tag"] in args.models]

    cells = []
    for m in models:
        for bk in banks:
            print(f"▶ {m['tag']} × {bk} …", flush=True)
            t0 = time.time()
            r = analyze_cell(m, bk, spec, seeds, args.device, args.mock, null_R)
            r["secs"] = round(time.time() - t0, 1)
            cells.append(r)
            if r.get("status") == "load_failed":
                print(f"   ❌ 로드 실패: {r['error']}")
            else:
                line = f"   self_acc {r['self_acc_mean']} (pass {r['self_acc_pass_frac']})"
                if "merge_confirmed_frac" in r:
                    line += (f" | t(ben↔conc) {r['t_ben_conc_mean']} spec {r['merge_specificity_mean']} "
                             f"MERGE {r['merge_confirmed_frac']}")
                print(line + f"  [{r['secs']}s]")

    # 뱅크별 병합 재현율(self_acc 통과 모델 중)
    cons = spec["stats"]["consistency_min"]; rep_min = spec["replication_min"]
    def bank_merge_rep(bk):
        ok = [c for c in cells if c["bank"] == bk and c.get("status") == "ok"
              and c.get("self_acc_pass_frac", 0) >= cons]
        if not ok:
            return None, 0
        rep = float(np.mean([1.0 if (c.get("merge_confirmed_frac", 0) >= 0.5
                                     and c.get("merge_specificity_mean", 0) > 0) else 0.0 for c in ok]))
        return round(rep, 4), len(ok)
    reps = {bk: bank_merge_rep(bk) for bk in banks}

    orig_rep = reps.get("orig", (None, 0))[0]
    para_rep = reps.get("para", (None, 0))[0]
    invariant = (orig_rep is not None and para_rep is not None
                 and orig_rep >= rep_min and para_rep >= rep_min)
    verdict = ("PROMPT_INVARIANT (모델 성질 — 영어 재표현에도 병합 유지)" if invariant
               else "PROMPT_DEPENDENT (원본 문장 특유 — para에서 무너짐)"
               if (orig_rep and orig_rep >= rep_min) else "판정불가(orig 미재현/데이터부족)")

    print("\n" + "=" * 64)
    for bk in banks:
        rep, n = reps[bk]
        print(f"[{bk:5s}] 병합 재현율 {rep}  (self_acc 통과 모델 {n}개)")
    print(f"→ 판정(주지표 orig vs para): {verdict}")
    if "ko" in banks:
        print(f"  (한국어 ko: {reps['ko'][0]} — 다국어 능력 모델 한정, 능력부족은 self_acc서 제외됨)")
    print("=" * 64)

    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    report = {"mock": args.mock, "mode": args.mode, "seeds": seeds, "null_R": null_R,
              "banks": banks, "cells": cells,
              "bank_merge_replication": {bk: reps[bk][0] for bk in banks},
              "replication_min": rep_min,
              "verdict": verdict, "prompt_invariant": bool(invariant)}
    p = save_report(outdir, "discover_prompts_report", report)
    print(f"상세: {p} (+ _latest.json)")
    if args.mode == "pilot":
        print("※ 파일럿은 승격 없음.")


if __name__ == "__main__":
    main()
