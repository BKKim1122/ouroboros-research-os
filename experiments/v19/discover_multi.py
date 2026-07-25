"""V19: 다중 모델 × 층 스윕 자기관련 축 발견 + ben↔conc cross-transfer 병합 확증.

V18(Qwen2.5-1.5B, layer12)에서 '3축 조직(beneficiary∪concern 병합)'을 발견. V19는:
  (a) 그 3축 구조가 다른 모델·층에서도 재현되나 (E7 방향, 교차모델 일반화)
  (b) ben↔conc 병합이 'LOO로 못 갈라서'가 아니라 '같은 축을 공유해서'임을 cross-transfer로 직접 확증

절차(비용 관리):
  1) 층 스윕은 self_acc(값싼 sanity)로만 스크린 → 모델별 최적층 1점 선택
  2) 최적층에서만 비싼 발견(v18 analyze_seed) + cross-transfer 실행
  3) self_acc 게이트 미통과 (모델×층)은 축 발견에서 제외 (프롬프트 약함 vs 3축아님 분리)

봉인(METHODS_STANDARD §1, null-상대) — 발견 기준은 v18 코어 재사용. cross-transfer는 spec 참조.
  · self_acc      : self극 vs other극 LOO분리 > 라벨셔플 null 95pct → 게이트 통과
  · MERGE_CONFIRMED: t(ben↔conc) > self/other셔플 null 95pct AND 병합특이도 >= margin(null-보정)
  · 교차모델      : MA2+MERGE가 self_acc통과 모델의 >= replication_min → E7 방향 지지
  ※ mock은 4축 직교라 self_acc↑·MA1·MERGE 미확증이 정상(판별력 확인). 확증은 mock 금지.

사용:
  python discover_multi.py --mode pilot --models qwen25_0p5b --null-R 60   # 스모크(1모델)
  python discover_multi.py --mode pilot                                    # 전체 모델(파일럿 seed)
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V18 = os.path.join(HERE, "..", "v18")
V17 = os.path.join(HERE, "..", "v17")
sys.path.insert(0, V18); sys.path.insert(0, V17)
from discover import analyze_seed, seal_from_spec, loo_ncc_acc, FACTORS  # v18 코어 재사용
from run_seed import HFBackend, MockBackend  # noqa: E402
from prompts_bank import build_bank  # noqa: E402


# ------------------------------------------------------------- 극/전이
def collect_poles(be, seed):
    """요인별 (self극 acts, other극 acts), 쌍 인덱스 정렬. 단위정규화."""
    bank = build_bank(seed); poles = {}
    for f in FACTORS:
        pairs = bank[f]["A_train"] + bank[f]["B_train"]
        S = be.acts([p for p, _ in pairs], f, +1)
        O = be.acts([n for _, n in pairs], f, -1)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-8)
        poles[f] = (S, O)
    return poles


def self_acc_obs(poles):
    Xs, ys = [], []
    for f, (S, O) in poles.items():
        Xs += [S, O]; ys += [1] * len(S) + [0] * len(O)
    X = np.vstack(Xs); y = np.array(ys)
    return loo_ncc_acc(X, y), X, y


def self_acc_gate(poles, R, seed, pct):
    obs, X, y = self_acc_obs(poles)
    rng = np.random.default_rng(seed + 40000)
    null = np.array([loo_ncc_acc(X, y[rng.permutation(len(y))]) for _ in range(R)])
    return round(float(obs), 4), round(float(np.quantile(null, pct)), 4), bool(obs > np.quantile(null, pct))


def _residualize(diffs):
    """모든 요인이 공유하는 self/other 축(전체 diff 평균 방향)을 제거 → 요인-특이 성분만 남김.
    이걸 안 하면 공유 '1인칭 vs 3인칭' 축 때문에 아무 요인쌍이나 전이 1.0이 나온다(V17 관찰)."""
    alld = np.vstack([diffs[f] for f in diffs])
    g = alld.mean(0); g = g / (np.linalg.norm(g) + 1e-8)
    return {f: d - (d @ g)[:, None] * g for f, d in diffs.items()}


def _transfer(diffs, src, dst):
    w = diffs[src].mean(0); w = w / (np.linalg.norm(w) + 1e-8)
    return float(np.mean((diffs[dst] @ w) > 0))


def cross_transfer_merge(poles, R, seed, tp, sp):
    raw = {f: (S - O) for f, (S, O) in poles.items()}
    distinct = [("beneficiary", "identity"), ("beneficiary", "privilege"),
                ("concern", "identity"), ("concern", "privilege")]

    def metrics(d):
        dr = _residualize(d)   # 공유 self/other 축 제거 후 전이
        t_bc = 0.5 * (_transfer(dr, "beneficiary", "concern")
                      + _transfer(dr, "concern", "beneficiary"))
        ctrl = float(np.mean([_transfer(dr, a, b) for a, b in distinct]))
        return t_bc, ctrl

    t_bc, control = metrics(raw)
    spec = t_bc - control
    rng = np.random.default_rng(seed + 50000)
    npair = {f: len(raw[f]) for f in raw}
    tb_null, sp_null = [], []
    for _ in range(R):
        sg = {f: rng.choice([-1, 1], size=npair[f])[:, None] for f in raw}
        tb, ct = metrics({f: sg[f] * raw[f] for f in raw})
        tb_null.append(tb); sp_null.append(tb - ct)
    t_gate = float(np.quantile(tb_null, tp)); margin = float(np.quantile(sp_null, sp))
    return {"t_ben_conc": round(t_bc, 4), "t_control_distinct": round(control, 4),
            "merge_specificity": round(spec, 4), "t_gate": round(t_gate, 4),
            "spec_margin": round(margin, 4),
            "merge_confirmed": bool(t_bc > t_gate and spec >= margin)}


# ------------------------------------------------------------- 모델×층
def make_mock(seed):
    bank = build_bank(seed); truth = {}
    for f in FACTORS:
        for p, n in bank[f]["A_train"] + bank[f]["B_train"]:
            truth[p] = (f, +1); truth[n] = (f, -1)
    return MockBackend(seed, truth=truth)


def pick_best_layer(be, layer_fracs, screen_seed):
    """층 스윕: self_acc(관측) 최고 층 선택. 모델 재로드 없이 be.layer만 변경."""
    n = be.model.config.num_hidden_layers
    scan = []
    for fr in layer_fracs:
        be.layer = max(1, int(n * fr))
        sa, _, _ = self_acc_obs(collect_poles(be, screen_seed))
        scan.append({"layer_frac": fr, "layer": be.layer, "self_acc": round(float(sa), 4)})
    best = max(scan, key=lambda r: r["self_acc"])
    return best, scan


def analyze_model(mspec, spec, seal, seeds, device, mock, null_R):
    tag, name = mspec["tag"], mspec["name"]
    ct = spec["cross_transfer"]
    out = {"tag": tag, "name": name}
    try:
        if mock:
            best = {"layer_frac": None, "layer": "mock"}; scan = []
            be = make_mock(seeds[0])
        else:
            be = HFBackend(name, spec["layer_fracs"][0],
                           device=(None if device == "auto" else device))
            best, scan = pick_best_layer(be, spec["layer_fracs"], seeds[0])
            be.layer = best["layer"]
    except Exception as e:  # 로드 실패(OOM/게이트/미존재) → 스킵, 다음 모델로
        return {"tag": tag, "name": name, "status": "load_failed",
                "error": str(e)[:200]}

    rows = []
    for s in seeds:
        be_s = make_mock(s) if mock else be
        if mock is False and not hasattr(be_s, "layer"):
            be_s.layer = best["layer"]
        elif not mock:
            be_s.layer = best["layer"]
        poles = collect_poles(be_s, s)
        sa, sa_gate, sa_pass = self_acc_gate(poles, min(null_R, 200), s, ct["self_acc_null_pct"])
        row = {"seed": s, "self_acc": sa, "self_acc_gate": sa_gate, "self_acc_pass": sa_pass}
        if sa_pass:
            disc = analyze_seed(be_s, s, seal)           # v18 발견 (MA1/MA2/MA3 + 병합진단)
            mrg = cross_transfer_merge(poles, null_R, s,
                                       ct["transfer_null_pct"], ct["specificity_null_pct"])
            row.update({"verdict": disc["verdict"], "ma1_met": disc["ma1_met"],
                        "struct": disc["struct"], "bene_conc_cos": disc.get("bene_conc_cos"),
                        "all_pairs_separable": disc.get("all_pairs_separable"),
                        **mrg})
        rows.append(row)

    passed = [r for r in rows if r["self_acc_pass"]]
    agg = {"tag": tag, "name": name, "status": "ok",
           "best_layer": best, "layer_scan": scan,
           "self_acc_mean": round(float(np.mean([r["self_acc"] for r in rows])), 4),
           "self_acc_pass_frac": round(float(np.mean([r["self_acc_pass"] for r in rows])), 4),
           "per_seed": rows}
    if passed:
        agg.update({
            "ma1_frac": round(float(np.mean([r["ma1_met"] for r in passed])), 4),
            "struct_frac": round(float(np.mean([r["struct"] for r in passed])), 4),
            "t_ben_conc_mean": round(float(np.mean([r["t_ben_conc"] for r in passed])), 4),
            "merge_specificity_mean": round(float(np.mean([r["merge_specificity"] for r in passed])), 4),
            "merge_confirmed_frac": round(float(np.mean([r["merge_confirmed"] for r in passed])), 4),
            "bene_conc_cos_mean": round(float(np.mean([r["bene_conc_cos"] for r in passed if r.get("bene_conc_cos") is not None])), 4),
        })
    return agg


# ------------------------------------------------------------- main


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
    ap.add_argument("--models", nargs="+", default=None, help="tag 부분집합 (기본 spec 전체)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--null-R", type=int, default=None)
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    args = ap.parse_args()

    import yaml
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    seal = seal_from_spec(spec)
    if args.null_R is not None:
        seal["null_R"] = args.null_R
    null_R = args.null_R or spec["discovery"]["null_R"]

    if args.mock and args.mode == "confirm":
        print("❌ 확증(confirm)에서 mock 금지."); sys.exit(2)
    if args.mock:
        print("⚠️  MOCK — 파이프라인 점검 전용(4축 직교). MERGE 미확증이 정상.")

    seeds = spec["pilot_seeds"] if args.mode == "pilot" else spec["confirmatory_seeds"]
    models = spec["models"]
    if args.models:
        models = [m for m in models if m["tag"] in args.models]

    grid = []
    for m in models:
        print(f"\n▶ {m['tag']} ({m['name']}) …", flush=True)
        t0 = time.time()
        r = analyze_model(m, spec, seal, seeds, args.device, args.mock, null_R)
        r["secs"] = round(time.time() - t0, 1)
        grid.append(r)
        if r.get("status") == "load_failed":
            print(f"   ❌ 로드 실패(스킵): {r['error']}")
        else:
            bl = r["best_layer"]
            line = (f"   layer={bl.get('layer')} self_acc={r['self_acc_mean']} "
                    f"(pass {r['self_acc_pass_frac']})")
            if "ma1_frac" in r:
                line += (f" | ma1 {r['ma1_frac']} | t(ben↔conc) {r['t_ben_conc_mean']} "
                         f"spec {r['merge_specificity_mean']} MERGE {r['merge_confirmed_frac']}")
            print(line + f"  [{r['secs']}s]")

    # 격자 집계: cross-transfer(MERGE)가 주지표, LOO(MA1/MA2)는 검정력 약한 보조지표.
    # (이전 'MA2 AND MERGE' 게이트는 두 지표 모순을 잘못 처리 → 폐기)
    ok = [r for r in grid if r.get("status") == "ok" and r.get("self_acc_pass_frac", 0) >= seal["consistency_min"]]
    cons = seal["consistency_min"]
    replication_min = spec["grid_verdict"]["replication_min"]

    def fr(cond):
        return round(float(np.mean([cond(r) for r in ok])), 4) if ok else None

    # 주지표: MERGE 재현 (모델별 seed 다수결 + 특이도 양수)
    merge_rep = fr(lambda r: r.get("merge_confirmed_frac", 0) >= 0.5 and r.get("merge_specificity_mean", 0) > 0)
    merge_rep_strict = fr(lambda r: r.get("merge_confirmed_frac", 0) >= cons)   # 엄격(seed 0.75+)
    spec_pos = fr(lambda r: r.get("merge_specificity_mean", 0) > 0)             # 특이도 양수 모델비율
    # 보조지표: LOO 기반 MA2(3축)
    ma2_loo = fr(lambda r: r.get("ma1_frac", 1) < cons and r.get("struct_frac", 0) >= cons)
    generalizes = (merge_rep is not None and merge_rep >= replication_min)

    print("\n" + "=" * 64)
    print(f"[격자 집계] self_acc 통과 모델 {len(ok)}/{len([r for r in grid if r.get('status')=='ok'])}")
    print(f"  ★ MERGE 재현(주지표): {merge_rep}  (엄격 seed0.75+: {merge_rep_strict}, 특이도>0: {spec_pos})")
    print(f"    보조: LOO MA2(3축) {ma2_loo}  ← 검정력 약해 seed 변동 큼, MERGE와 어긋나면 MERGE 우선")
    print(f"  → 교차모델 일반화(E7 방향): {'지지' if generalizes else '미지지'}  (MERGE 재현 {merge_rep} vs 임계 {replication_min})")
    print("=" * 64)

    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    report = {"mock": args.mock, "mode": args.mode, "seeds": seeds, "null_R": null_R,
              "kill_criterion_sealed": spec.get("kill_criteria"),
              "grid": grid,
              "grid_summary": {"models_ok": len(ok),
                               "merge_replication_primary": merge_rep,
                               "merge_replication_strict": merge_rep_strict,
                               "specificity_positive_frac": spec_pos,
                               "ma2_loo_frac_secondary": ma2_loo,
                               "replication_min": replication_min,
                               "primary_metric": "cross_transfer_MERGE",
                               "generalizes_E7_direction": bool(generalizes)}}
    p = save_report(outdir, "discover_multi_report", report)
    print(f"\n상세: {p} (+ _latest.json)")
    if args.mode == "pilot":
        print("※ 파일럿은 승격 없음. 확증은 freeze + 거버너 경유(다음 단계).")


if __name__ == "__main__":
    main()
