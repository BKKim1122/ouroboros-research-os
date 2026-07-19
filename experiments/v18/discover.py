"""V18 파일럿/확증 엔진: 자기관련 축의 무감독 발견 (ARI 기반).

질문(결정 B의 귀결): 우리가 손으로 정의한 4요인이 모델이 *실제로* 조직하는
축과 일치하는가? V17까지는 "우리 요인이 인코딩됐나"를, V18은 역으로 "모델의
자생 구조가 우리 요인과 맞나"를 묻는다.

방법:
  1) V17 뱅크의 최소쌍(요인4 x A/B군) + 대명사 대조군에서
     차이벡터 d = act(자기극) - act(타자극) 수집 (단위정규화)
  2) PCA로 유효 차원, k=2..6 k-means(다중 재시작) + 실루엣으로 자연 군집 추정
  3) 군집 라벨 vs (a) 요인 라벨, (b) 템플릿 골격(A/B/ctrl) 라벨의 ARI 비교

봉인(METHODS_STANDARD §1, null-상대):  ── 값이 아니라 절차를 고정한다 ──
  N2 실루엣 null : 공분산매칭 가우시안 대체표본(R회) 대비 백분위
  N1 ARI    null : 라벨순열(R회) 대비 백분위
  best_k         : 실루엣이 N2의 (1-alpha/|k|) 백분위(=Bonferroni)를 넘는 k 중 실루엣 최대.
                   넘는 k 없음 → MA3.
  판정(best_k에서만):
    ARI(템플릿) > ARI(요인)                                  → 어휘교락 플래그(MA1 불가)
    ARI(요인) > N1 백분위 AND [ARI(요인)-ARI(템플릿)] >= δ    → MA1 (4요인 강화)
    그 외(구조는 있으나 4요인 정렬 아님)                      → MA2 (H-A 약화, 반증 아님)
  δ: null-보정(공동 셔플 |ARI(요인')-ARI(템플릿')|의 95pct) 또는 고정값. spec에서 봉인.

  ※ mock은 파이프라인 점검 전용(과학적 의미 0). 확증(--mode confirm)에서는 하드 차단.
  ※ 파일럿은 승격 없음(파이프라인·null 분포 검증만). 확증은 freeze+거버너(discovery) 경유.

사용:
  python discover.py --mode pilot   --seeds 0 1 2            # 딸깍 파일럿(run_v18_pilot.sh)
  python discover.py --mode confirm --seeds 20 21 ... 27     # freeze 배선 후 (다음 단계)
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V17 = os.path.join(HERE, "..", "v17")
sys.path.insert(0, V17)
from prompts_bank import build_bank, PERSON_CONTROL  # noqa: E402
from run_seed import FACTORS, MockBackend, HFBackend  # noqa: E402


# ------------------------------------------------------------------ 수치 유틸
def kmeans(X, k, seed, iters=50, restarts=10):
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)].copy()
        lab = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            D = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = D.argmin(1)
            newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = lab.copy(), inertia
    return best


def silhouette(X, lab, D=None):
    n = len(X)
    if D is None:
        D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(n):
        same = lab == lab[i]; same[i] = False
        if not same.any():
            continue
        others = [D[i][lab == j].mean() for j in set(lab.tolist()) if j != lab[i]]
        if not others:
            continue
        a = D[i][same].mean(); b = min(others)
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s)) if s else 0.0


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ua, ub = np.unique(a), np.unique(b)
    M = np.array([[(np.logical_and(a == x, b == y)).sum() for y in ub] for x in ua])
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = comb(M).sum(); sum_a = comb(M.sum(1)).sum(); sum_b = comb(M.sum(0)).sum()
    exp = sum_a * sum_b / max(comb(n), 1e-9)
    mx = (sum_a + sum_b) / 2.0
    return float((sum_ij - exp) / max(mx - exp, 1e-9))


def loo_ncc_acc(Xsub, y):
    """두 그룹(y∈{0,1})에 대한 LOO 최근접-중심 분류 정확도. '두 요인을 구분할 수 있나'."""
    m = len(Xsub); correct = 0
    for i in range(m):
        mask = np.ones(m, bool); mask[i] = False
        g0 = Xsub[mask & (y == 0)]; g1 = Xsub[mask & (y == 1)]
        if len(g0) == 0 or len(g1) == 0:
            continue
        c0, c1 = g0.mean(0), g1.mean(0)
        pred = 0 if ((Xsub[i] - c0) ** 2).sum() < ((Xsub[i] - c1) ** 2).sum() else 1
        correct += int(pred == y[i])
    return correct / m


def pair_separability(P, flab, fa, fb, R, seed, pct):
    """요인쌍(fa,fb)의 차이벡터가 분리되는가: 관측 LOO정확도 vs 라벨셔플 null 백분위.
    MA1의 핵심 판별자 — 예: beneficiary vs concern이 우연 이상 구분되면 4번째 경계 실재."""
    idx = [i for i, l in enumerate(flab) if l in (fa, fb)]
    Xs = P[idx]
    y = np.array([0 if flab[i] == fa else 1 for i in idx])
    obs = loo_ncc_acc(Xs, y)
    rng = np.random.default_rng(seed + 13000 + abs(hash((fa, fb))) % 9973)
    null = np.array([loo_ncc_acc(Xs, y[rng.permutation(len(y))]) for _ in range(R)])
    gate = float(np.quantile(null, pct))
    return round(float(obs), 4), round(gate, 4), bool(obs > gate)


# ------------------------------------------------------------------ null (대조)
def _cov_sqrt_lowrank(Xnorm):
    """cov(Xnorm)와 같은 2차통계를 갖는 저차원 sqrt 인자 M(r x d)을 **1회** 분해로 산출.
    표본 y = z @ M (z ~ N(0,I_r)) 는 cov(Xc)를 재현하고 데이터 부분공간(rank<=n-1)에 산다.
    실모델 d=1536에서 반복마다 d x d 재분해하던 것을 제거(핵심 성능 수정)."""
    Xc = Xnorm - Xnorm.mean(0)
    n = len(Xc)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # Xc = U S Vt
    scale = S / np.sqrt(max(n - 1, 1))
    return scale[:, None] * Vt                            # (r, d)


def n2_silhouette_null(Xnorm, k_grid, R, seed, restarts=5):
    """N2: 공분산매칭 가우시안 대체표본(구조 없음)에서 k별 실루엣 우연 분포.
    Sigma sqrt는 1회만 분해하고, 대체표본은 값싼 z@M 으로 생성."""
    rng = np.random.default_rng(seed + 90000)
    M = _cov_sqrt_lowrank(Xnorm)                          # (r, d) — 1회 분해
    r = M.shape[0]; n = len(Xnorm)
    null = {k: [] for k in k_grid}
    for ri in range(R):
        Y = rng.standard_normal((n, r)) @ M
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        Yc = Y - Y.mean(0)
        V = np.linalg.svd(Yc, full_matrices=False)[2]
        proj = Yc @ V[:min(10, V.shape[0])].T
        Dy = np.sqrt(((Yc[:, None, :] - Yc[None]) ** 2).sum(-1))  # 1회 계산 후 재사용
        for k in k_grid:
            lab = kmeans(proj, k, seed + ri, restarts=restarts)
            null[k].append(silhouette(Yc, lab, Dy))
    return {k: np.array(v) for k, v in null.items()}


def n1_ari_null(lab, flab, tlab, R, seed):
    """N1: 라벨순열에서 ARI(요인)·ARI(템플릿)의 우연 분포 + 공동셔플 격차 분포(δ용)."""
    rng = np.random.default_rng(seed + 70000)
    fa, ta = np.asarray(flab), np.asarray(tlab)
    a_f, a_t, gap = [], [], []
    for _ in range(R):
        fp = fa[rng.permutation(len(fa))]
        tp = ta[rng.permutation(len(ta))]
        vf, vt = ari(lab, fp), ari(lab, tp)
        a_f.append(vf); a_t.append(vt); gap.append(vf - vt)
    return np.array(a_f), np.array(a_t), np.array(gap)


# ------------------------------------------------------------------ 백엔드/데이터
def collect_diffs(be, seed):
    bank = build_bank(seed)
    diffs, flab, tlab = [], [], []
    for f in FACTORS:
        for fam in ("A_train", "B_train"):
            pairs = bank[f][fam]
            P = be.acts([p for p, _ in pairs], f, +1)
            N = be.acts([n for _, n in pairs], f, -1)
            for d in P - N:
                diffs.append(d); flab.append(f); tlab.append(fam)
    P = be.acts([p for p, _ in PERSON_CONTROL], FACTORS[0], +1)
    N = be.acts([n for _, n in PERSON_CONTROL], FACTORS[0], -1)
    for d in P - N:
        diffs.append(d); flab.append("person_ctrl"); tlab.append("ctrl")
    X = np.stack(diffs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X, flab, tlab


# ------------------------------------------------------------------ seed 분석
def analyze_seed(be, seed, seal):
    k_grid = seal["k_grid"]; R = seal["null_R"]
    X, flab, tlab = collect_diffs(be, seed)

    # 관측 파이프라인 값은 k와 무관한 부분(중심화·SVD·투영·거리)을 1회만 계산
    Xc = X - X.mean(0)
    Vt = np.linalg.svd(Xc, full_matrices=False)[2]
    S = np.linalg.svd(Xc, full_matrices=False)[1]
    var = (S ** 2) / (S ** 2).sum()
    pca_dim90 = int(np.searchsorted(np.cumsum(var), 0.90)) + 1
    proj = Xc @ Vt[:min(10, Vt.shape[0])].T
    Dobs = np.sqrt(((Xc[:, None, :] - Xc[None]) ** 2).sum(-1))

    # 관측 실루엣 + N2 게이트
    n2 = n2_silhouette_null(X, k_grid, R, seed)
    sil_gate_pct = 1.0 - seal["sil_alpha"] / len(k_grid)   # Bonferroni
    grid, passed, labs = [], [], {}
    for k in k_grid:
        lab = kmeans(proj, k, seed)
        labs[k] = lab
        sil = silhouette(Xc, lab, Dobs)
        thr = float(np.quantile(n2[k], sil_gate_pct))
        ok = sil > thr
        grid.append({"k": k, "silhouette": round(sil, 4),
                     "n2_gate": round(thr, 4), "pass": bool(ok)})
        if ok:
            passed.append((sil, k))

    if not passed:
        return {"seed": seed, "best_k": None, "pca_dim90": pca_dim90,
                "grid": grid, "verdict": "MA3", "ma1_met": False,
                "struct": False, "lexical_flag": False,
                "ari_factors": None, "ari_templates": None}

    best_sil, best_k = max(passed)
    lab_best = labs[best_k]
    ari_f_best, ari_t_best = ari(lab_best, flab), ari(lab_best, tlab)

    # ── MA1은 '가설 기수' k=4에서 평가한다 (argmax가 아니라 '4요인으로 조직되나') ──
    pct = seal["pair_sep_null_pct"]
    if 4 in labs:
        lab4 = labs[4]
        ari_f4, ari_t4 = ari(lab4, flab), ari(lab4, tlab)
        nf, _, ngap = n1_ari_null(lab4, flab, tlab, R, seed)
        ari_f4_thr = float(np.quantile(nf, seal["ari_null_pct"]))
        delta = (float(np.quantile(ngap, seal["delta_null_pct"]))
                 if seal["delta_mode"] == "null_calibrated" else float(seal["delta_fixed"]))
        k4_gate_pass = next((g["pass"] for g in grid if g["k"] == 4), False)

        def modal(f):
            idx = [i for i, l in enumerate(flab) if l == f]
            return int(np.bincount(lab4[idx]).argmax()) if idx else -1
        merge = {"beneficiary_cluster": modal("beneficiary"),
                 "concern_cluster": modal("concern")}
        merge["bene_conc_merged"] = bool(merge["beneficiary_cluster"] == merge["concern_cluster"])
    else:
        ari_f4 = ari_t4 = ari_f4_thr = delta = None
        k4_gate_pass = False; merge = {}

    # ── 핵심 판별자: 6개 요인쌍이 각각 우연 이상 분리되는가 (특히 beneficiary|concern) ──
    pairs = [(FACTORS[i], FACTORS[j]) for i in range(len(FACTORS))
             for j in range(i + 1, len(FACTORS))]
    pair_sep = {}
    for fa, fb in pairs:
        o, g, ok = pair_separability(proj, flab, fa, fb, R, seed, pct)
        pair_sep[f"{fa}|{fb}"] = {"obs": o, "gate": g, "pass": ok}
    all_pairs_sep = all(v["pass"] for v in pair_sep.values())
    bene_conc_sep = pair_sep.get("beneficiary|concern", {}).get("pass", False)

    # ── 병합 진단 (report-only, 게이트에 영향 없음): '분리 실패'를 '공유 축'의 양의 증거로 ──
    def fac_centroid(f):
        idx = [i for i, l in enumerate(flab) if l == f]
        return X[idx].mean(0) if idx else None
    bc, cc = fac_centroid("beneficiary"), fac_centroid("concern")
    bene_conc_cos = (round(float(np.dot(bc, cc) /
                     ((np.linalg.norm(bc) * np.linalg.norm(cc)) + 1e-9)), 4)
                     if bc is not None and cc is not None else None)
    if 4 in labs:
        merged3 = ["benefit_merged" if l in ("beneficiary", "concern") else l for l in flab]
        ari_merged3 = round(ari(labs[4], merged3), 4)
        ari_merged_ge_split = bool(ari_merged3 >= ari_f4)   # 병합 라벨이 4분할만큼/이상 설명 → 3축 증거
    else:
        ari_merged3 = None; ari_merged_ge_split = False

    # ── 봉인 판정 (k=4 게이트는 강등: 진단용. MA1은 3조건 — 정합·비어휘·모든쌍분리) ──
    lexical = (ari_t4 is not None) and (ari_t4 > ari_f4)
    ma1_cond = ((4 in labs) and (not lexical)
                and (ari_f4 > ari_f4_thr) and ((ari_f4 - ari_t4) >= delta)
                and all_pairs_sep)
    if lexical:
        verdict, ma1 = "MA1_blocked_lexical", False
    elif ma1_cond:
        verdict, ma1 = "MA1", True
    else:
        verdict, ma1 = "MA2", False

    return {"seed": seed, "best_k": best_k, "best_silhouette": round(best_sil, 4),
            "pca_dim90": pca_dim90, "grid": grid,
            "ari_factors": None if ari_f4 is None else round(ari_f4, 4),      # k=4 기준
            "ari_templates": None if ari_t4 is None else round(ari_t4, 4),
            "ari_factors_gate": None if ari_f4_thr is None else round(ari_f4_thr, 4),
            "delta": None if delta is None else round(delta, 4),
            "ari_factors_bestk": round(ari_f_best, 4),
            "ari_templates_bestk": round(ari_t_best, 4),
            "k4_gate_pass": bool(k4_gate_pass),   # 진단용(강등): MA1 게이트 아님
            "pair_separability": pair_sep, "all_pairs_separable": bool(all_pairs_sep),
            "bene_conc_separable": bool(bene_conc_sep),
            "bene_conc_cos": bene_conc_cos, "ari_merged3": ari_merged3,
            "ari_merged_ge_split": bool(ari_merged_ge_split),   # 병합 라벨이 4분할≥ → 3축 양의 증거
            "verdict": verdict, "ma1_met": ma1, "struct": True,
            "lexical_flag": bool(lexical), "merge": merge}


# ------------------------------------------------------------------ 집계·판정
def seal_from_spec(spec):
    d = (spec.get("discovery") or {})
    return {
        "k_grid": d.get("k_grid", [2, 3, 4, 5, 6]),
        "null_R": int(d.get("null_R", 1000)),
        "sil_alpha": float(d.get("sil_alpha", 0.05)),
        "ari_null_pct": float(d.get("ari_null_pct", 0.99)),
        "delta_mode": d.get("delta_mode", "null_calibrated"),
        "delta_null_pct": float(d.get("delta_null_pct", 0.95)),
        "delta_fixed": float(d.get("delta_fixed", 0.10)),
        "pair_sep_null_pct": float(d.get("pair_sep_null_pct", 0.95)),
        "consistency_min": float(d.get("consistency_min", 0.75)),
    }


def aggregate(rows, seal, mode):
    def col(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)) if vals else (None, None)
    metrics = {k: {"mean": col(k)[0], "sd": col(k)[1]}
               for k in ["best_k", "best_silhouette", "ari_factors",
                         "ari_templates", "pca_dim90"]}
    struct_frac = float(np.mean([r["struct"] for r in rows]))
    ma1_frac = float(np.mean([r["ma1_met"] for r in rows]))
    lexical_any = any(r["lexical_flag"] for r in rows)
    cmin = seal["consistency_min"]

    if mode == "pilot":
        verdict = "PILOT — 파이프라인/ null 분포 검증만, 승격 없음"
    else:
        if lexical_any:
            verdict = "FLAG_LEXICAL — 어휘교락, MA1 불가"
        elif ma1_frac >= cmin:
            verdict = f"MA1 지지 (4요인 강화) — ma1 {ma1_frac:.2f} ≥ {cmin}"
        elif struct_frac >= cmin:
            verdict = f"MA2 (H-A 약화, 반증 아님) — 구조 {struct_frac:.2f}, ma1 {ma1_frac:.2f}<{cmin}"
        else:
            verdict = f"MA3 (구조 없음) — 구조 {struct_frac:.2f} < {cmin}"
    return metrics, struct_frac, ma1_frac, lexical_any, verdict


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="실모델 디바이스. GPU OOM이면 cpu로 폴백(파일럿은 CPU로도 수분).")
    ap.add_argument("--null-R", type=int, default=None, help="spec의 null_R 덮어쓰기(파일럿 속도조절용)")
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    args = ap.parse_args()

    import yaml
    spec = yaml.safe_load(open(args.spec, encoding="utf-8")) or {}
    seal = seal_from_spec(spec)
    if args.null_R is not None:
        seal["null_R"] = args.null_R

    seeds = args.seeds or spec.get(
        "pilot_seeds" if args.mode == "pilot" else "confirmatory_seeds", [0, 1, 2])

    # config는 v17에서 읽되(모델/층/rev), mock 기본은 False(원본의 True 함정 제거)
    cfg = {}
    cfg_path = os.path.join(V17, "config.yaml")
    if os.path.exists(cfg_path):
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    mock = args.mock or bool(cfg.get("mock", False))

    # ── mock 가드 (METHODS_STANDARD §2): 확증에서 mock 하드 차단
    if mock and args.mode == "confirm":
        print("❌ 확증(confirm)에서 mock 금지 — MockBackend는 truth로 ARI≈1을 순환 생성한다.")
        print("   실모델(config mock:false)로 freeze+거버너 경유해 실행할 것. 중단.")
        sys.exit(2)
    if mock:
        print("⚠️  MOCK 백엔드 — 파이프라인 점검 전용. 과학적 의미 0. 판정에 쓰지 말 것.")

    # seed 분리 검증(§4): pilot ∩ confirmatory = ∅
    ps = set(spec.get("pilot_seeds", [])); cs = set(spec.get("confirmatory_seeds", []))
    if ps & cs:
        print(f"❌ seed 분리 위반: pilot ∩ confirmatory = {sorted(ps & cs)}. 중단."); sys.exit(2)

    rows = []
    for s in seeds:
        if mock:
            bank = build_bank(s); truth = {}
            for f in FACTORS:
                for p, n in bank[f]["A_train"] + bank[f]["B_train"]:
                    truth[p] = (f, +1); truth[n] = (f, -1)
            be = MockBackend(s, truth=truth)
        else:
            dev = None if args.device == "auto" else args.device
            be = HFBackend(cfg.get("model", "Qwen/Qwen2.5-1.5B"),
                           float(cfg.get("layer_frac", 0.43)),
                           device=dev, revision=cfg.get("revision"))
            if args.layer is not None:
                be.layer = args.layer
        rows.append(analyze_seed(be, s, seal))

    metrics, struct_frac, ma1_frac, lexical_any, verdict = aggregate(rows, seal, args.mode)

    # ── 리포트 §3: 콘솔(지표표 → 대조 → [봉인 판정])
    layer = "mock" if mock else getattr(be, "layer", "?")
    print(f"\n=== V18 축 발견 [{args.mode}] layer={layer} seeds={seeds} R={seal['null_R']} ===")
    print("[지표]        mean ± sd")
    for k, v in metrics.items():
        print(f"  {k:16s} {v['mean']} ± {v['sd']}")
    print("[대조]  ARI 우연≈0(라벨순열) / 실루엣 게이트=N2 공분산매칭 "
          f"{1.0 - seal['sil_alpha']/len(seal['k_grid']):.3f}pct / δ={seal['delta_mode']}")
    print("[per-seed] " + " | ".join(
        f"s{r['seed']}:{r['verdict']}"
        f"(bestk={r['best_k']},k4gate={'T' if r.get('k4_gate_pass') else 'F'}*,"
        f"ARIf4={r['ari_factors']},allpairs={'T' if r.get('all_pairs_separable') else 'F'},"
        f"beneconc={'분리' if r.get('bene_conc_separable') else '병합'})" for r in rows))
    # 병합 증거 요약(report-only)
    cos_vals = [r["bene_conc_cos"] for r in rows if r.get("bene_conc_cos") is not None]
    merge_frac = np.mean([r.get("merge", {}).get("bene_conc_merged", False) for r in rows])
    ge_frac = np.mean([r.get("ari_merged_ge_split", False) for r in rows])
    print(f"[병합진단*] bene/conc cos={np.mean(cos_vals):.3f} / modal병합 {merge_frac:.2f} "
          f"/ 3라벨ARI≥4라벨 {ge_frac:.2f}  (*진단용, 게이트 아님)")
    print(f"[봉인 판정] {verdict}")

    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    report = {
        "mock": mock, "mode": args.mode, "seeds": seeds,
        "kill_criterion_sealed": {
            "primary": "null-상대(METHODS_STANDARD §1), MA1은 가설기수 k=4에서 평가",
            "best_k": f"실루엣 > N2 {1.0-seal['sil_alpha']/len(seal['k_grid']):.3f}pct 인 k 중 최대(진단용)",
            "MA1": ("아래 3조건 전부(k=4 기준): "
                    "(1)구조존재 "
                    f"(2)ARI(요인)>N1 {seal['ari_null_pct']}pct AND ARI(요인)-ARI(템플릿)>=δ({seal['delta_mode']}) "
                    f"(3)6개 요인쌍 전부 LOO분리 > null {seal['pair_sep_null_pct']}pct (특히 beneficiary|concern)"),
            "k4_gate_role": "진단용(강등) — MA1 게이트 아님",
            "MA2": "구조는 존재하나 위 조건 미충족(예: bene∪conc 병합 / 쌍 분리 실패) — H-A 약화, 반증 아님",
            "MA3": "어느 k도 실루엣 게이트 미통과",
            "lexical_flag": "ARI(템플릿) > ARI(요인) @k=4 → MA1 불가",
            "merge_diagnostics_report_only": "bene/conc 코사인 · 3라벨(병합)ARI≥4라벨 · modal병합 (게이트 불변)",
            "consistency_min": seal["consistency_min"],
            "pair_sep_null_pct": seal["pair_sep_null_pct"],
            "delta": {k: seal[k] for k in ["delta_mode", "delta_null_pct", "delta_fixed"]},
        },
        "metrics": metrics, "per_seed": rows,
        "merge_evidence": {
            "bene_conc_cos_mean": round(float(np.mean(cos_vals)), 4) if cos_vals else None,
            "modal_merge_frac": round(float(merge_frac), 4),
            "ari_merged_ge_split_frac": round(float(ge_frac), 4),
        },
        "struct_frac": round(struct_frac, 4), "ma1_frac": round(ma1_frac, 4),
        "lexical_any": lexical_any, "verdict_sealed": verdict,
    }
    with open(os.path.join(outdir, "discover_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n상세: {os.path.relpath(os.path.join(outdir, 'discover_report.json'), HERE)}")
    print("판독: MA1=모델 축이 4요인과 정합 / MA2=병합 등 다른 조직(약화) / MA3=군집 구조 없음")
    if args.mode == "pilot":
        print("※ 파일럿은 승격 없음. 확증은 seed 분리 + freeze + 거버너(discovery) 경유(다음 단계).")


if __name__ == "__main__":
    main()
