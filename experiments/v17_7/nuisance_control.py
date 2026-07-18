"""V17.7 — nuisance 통제 (약한고리 A).

V17의 4요인 cross-template 전이(min_factor)가 자기관련 구성개념 때문인가, 아니면
주제·감정가·격식·구문·evidentiality·agency 같은 일반 담화 nuisance가 실려서인가.

방법: 각 nuisance 축을 self/요인과 무관한 일반 문장 극성쌍으로 방향 추정 → 6축이
이루는 부분공간을 4요인 활성치에서 제거 → cross-template 전이(min_factor)·gap 재측정.
동일 차원(6) random 정규직교 부분공간 제거를 대조로 병행(차원 제거 일반효과 배제).

★ 판정 기준은 spec.yaml에 봉인됨. 이 스크립트는 봉인 기준에 '대입'만 한다.
   nuis_min       = nuisance 6축 제거 후 min_factor 전이 (seed 평균)
   nuis_minus_rand_drop = (base_min − nuis_min) − (base_min − rand_min)  [nuisance-특이 하락]

   H_beyond   : nuis_min >= 0.75 AND nuis_minus_rand_drop <= 0.10  → nuisance로 환원 안 됨
   H_nuisance : nuis_min <= 0.60 AND nuis_minus_rand_drop >= 0.15  → 상당 부분 nuisance
   H_partial  : 그 사이

사용:
  python nuisance_control.py                # config대로 (실모델/mock)
  python nuisance_control.py --mock         # 파이프라인 검증 (수치 무의미)
"""
from __future__ import annotations
import argparse, os, sys, json, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V17_DIR = os.path.join(os.path.dirname(HERE), "v17")
sys.path.insert(0, V17_DIR); sys.path.insert(0, HERE)
from run_seed import FACTORS, MockBackend, HFBackend, steering_vec, centroid_probe_acc  # noqa: E402
from prompts_bank import build_bank                                                     # noqa: E402
from nuisance_prompts import NUISANCE, AXES, build_nuisance                             # noqa: E402

# ── 봉인된 kill criterion (spec.yaml과 일치, 결과 전 고정) ──
KILL = {"beyond_min": 0.75, "nuisance_min": 0.60,
        "beyond_gap": 0.10, "nuisance_gap": 0.15}


def project_out_subspace(X, U):
    if U.shape[1] == 0:
        return X
    return X - (X @ U) @ U.T


def orthonormal(D):            # D (m,dim) → (dim,m) 정규직교 기저
    Q, _ = np.linalg.qr(D.T)
    return Q[:, :D.shape[0]]


def random_subspace(dim, k, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
    return Q[:, :k]


def transfer_gap(A, B, U):
    tf = lambda X: project_out_subspace(X, U)
    diag = {f: centroid_probe_acc(tf(A[f][0]), tf(A[f][1]), tf(B[f][0]), tf(B[f][1]))
            for f in FACTORS}
    off = np.mean([centroid_probe_acc(tf(A[a][0]), tf(A[a][1]), tf(B[b][0]), tf(B[b][1]))
                   for a in FACTORS for b in FACTORS if a != b])
    dv = list(diag.values())
    return float(np.mean(dv)), float(min(dv)), float(np.mean(dv) - off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--n_rand", type=int, default=12)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    cfg = {}
    p = os.path.join(HERE, "config.yaml")
    if os.path.exists(p):
        import yaml
        cfg = yaml.safe_load(open(p, encoding="utf-8")) or {}
    mock = args.mock or cfg.get("mock", True)

    keys = ["base_min", "nuis_min", "rand_min", "base_mean", "nuis_mean",
            "nuis_minus_rand_drop", "base_gap", "nuis_gap"]
    per = {k: [] for k in keys}

    for seed in args.seeds:
        bank = build_bank(seed)
        nb = build_nuisance(seed)
        truth = {}
        for f in FACTORS:
            for pp, nn in bank[f]["A_train"] + bank[f]["B_train"]:
                truth[pp] = (f, +1); truth[nn] = (f, -1)
        be = (MockBackend(seed, truth=truth) if mock else
              HFBackend(cfg.get("model", "Qwen/Qwen2.5-1.5B"),
                        float(cfg.get("layer_frac", 0.43)), revision=cfg.get("revision")))

        def acts(texts, f="identity", pole=0):
            return be.acts(texts, f, pole)

        A = {f: (acts([p for p, _ in bank[f]["A_train"]], f, +1),
                 acts([n for _, n in bank[f]["A_train"]], f, -1)) for f in FACTORS}
        B = {f: (acts([p for p, _ in bank[f]["B_train"]], f, +1),
                 acts([n for _, n in bank[f]["B_train"]], f, -1)) for f in FACTORS}

        # nuisance 6축 방향 (self/요인 무관 문장 → mock에선 pole=0으로 무해)
        ndirs = []
        for ax in AXES:
            pairs = nb[ax]
            pa = acts([p for p, _ in pairs]); na = acts([n for _, n in pairs])
            ndirs.append(steering_vec(pa, na))
        U_n = orthonormal(np.stack(ndirs))
        dim = U_n.shape[0]

        bm, bmin, bg = transfer_gap(A, B, np.zeros((dim, 0)))     # 제거 없음
        nm, nmin, ng = transfer_gap(A, B, U_n)                    # nuisance 제거
        rng = np.random.default_rng(seed)
        rmins = [transfer_gap(A, B, random_subspace(dim, len(AXES), rng))[1]
                 for _ in range(args.n_rand)]
        rmin = float(np.mean(rmins))

        per["base_min"].append(bmin); per["nuis_min"].append(nmin); per["rand_min"].append(rmin)
        per["base_mean"].append(bm); per["nuis_mean"].append(nm)
        per["nuis_minus_rand_drop"].append((bmin - nmin) - (bmin - rmin))
        per["base_gap"].append(bg); per["nuis_gap"].append(ng)

    def ms(xs):
        return round(st.mean(xs), 3), round(st.stdev(xs) if len(xs) > 1 else 0.0, 3)

    report = {"mock": mock, "seeds": args.seeds, "nuisance_axes": AXES,
              "kill_criterion_sealed": KILL,
              "metrics": {k: {"mean": ms(v)[0], "sd": ms(v)[1]} for k, v in per.items()}}

    nuis_min = report["metrics"]["nuis_min"]["mean"]
    nmr = report["metrics"]["nuis_minus_rand_drop"]["mean"]

    # ── 봉인 기준 대입 (해석은 결과 전에 고정된 문구) ──
    if nuis_min >= KILL["beyond_min"] and nmr <= KILL["beyond_gap"]:
        verdict = ("H_beyond: 4요인 cross-template 전이는 통제된 6개 nuisance로 환원되지 "
                   f"않는다 (nuis_min={nuis_min} >= 0.75, nuisance-특이 하락={nmr} <= 0.10). "
                   "E6 강화. 단 이는 6개 nuisance에 한정된 통제이며 완전한 확증은 아니다.")
    elif nuis_min <= KILL["nuisance_min"] and nmr >= KILL["nuisance_gap"]:
        verdict = ("H_nuisance: 전이의 상당 부분이 통제된 nuisance에 기인한다 "
                   f"(nuis_min={nuis_min} <= 0.60, nuisance-특이 하락={nmr} >= 0.15). "
                   "E6를 self-related에서 상당히 후퇴시켜야 한다. (분석자 편향이 반길 수 있는 "
                   "방향이나, 환영도 축소도 없이 봉인 기준대로 보고한다.)")
    else:
        verdict = (f"H_partial: 부분 환원 (nuis_min={nuis_min}, nuisance-특이 하락={nmr}). "
                   "어느 봉인 기준에도 명확히 걸리지 않음 — 서술 필요.")
    report["verdict_sealed"] = verdict

    print(f"\n=== nuisance 통제 (mock={mock}, seeds={args.seeds}, 축 {len(AXES)}개) ===")
    print(f"봉인 축: {', '.join(AXES)}")
    m = report["metrics"]
    print(f"\n  min_factor 전이:  base={m['base_min']['mean']:.3f}  "
          f"nuisance제거={m['nuis_min']['mean']:.3f}  random제거={m['rand_min']['mean']:.3f}")
    print(f"  nuisance-특이 하락 (nuis_drop − rand_drop) = {nmr:+.3f}")
    print(f"  (참고) mean 전이 base={m['base_mean']['mean']:.3f} → nuisance={m['nuis_mean']['mean']:.3f}"
          f" | gap base={m['base_gap']['mean']:.3f} → nuisance={m['nuis_gap']['mean']:.3f}")
    print(f"\n[봉인 판정] {verdict}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    outp = os.path.join(HERE, "results", "nuisance_report.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("상세:", outp)


if __name__ == "__main__":
    main()
