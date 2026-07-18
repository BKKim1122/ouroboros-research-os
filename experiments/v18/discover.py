"""V18 파일럿: 자기관련 축의 무감독 발견.

방법:
  1) V17 뱅크의 모든 최소쌍(요인 4 x A/B군)과 대명사 대조군에서
     차이벡터 d = act(자기극) - act(타자극) 수집
  2) PCA로 유효 차원 추정
  3) k=2..6 k-means(다중 재시작) + 실루엣으로 자연 군집 수 추정
  4) 군집 라벨 vs (a) 우리의 요인 라벨, (b) 템플릿 골격 라벨의 ARI 비교
     → 군집이 자기구조를 따르는지, 어휘 골격을 따르는지 판별
  5) 요인 쌍별 병합 경향 보고 (bene-conc 병합 = MA2 예측)

사용:
  python discover.py --layer 12
  python discover.py --mock
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V17 = os.path.join(HERE, "..", "v17")
sys.path.insert(0, V17)
from prompts_bank import build_bank, PERSON_CONTROL  # noqa: E402
from run_seed import FACTORS, MockBackend, HFBackend  # noqa: E402


def kmeans(X, k, seed, iters=50, restarts=10):
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)].copy()
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


def silhouette(X, lab):
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(n):
        same = lab == lab[i]; same[i] = False
        if not same.any():
            continue
        a = D[i][same].mean()
        b = min(D[i][lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s))


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ua, ub = np.unique(a), np.unique(b)
    M = np.array([[(np.logical_and(a == x, b == y)).sum() for y in ub] for x in ua])
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = comb(M).sum(); sum_a = comb(M.sum(1)).sum(); sum_b = comb(M.sum(0)).sum()
    exp = sum_a * sum_b / comb(n)
    mx = (sum_a + sum_b) / 2.0
    return float((sum_ij - exp) / max(mx - exp, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    cfg = {}
    cfg_path = os.path.join(V17, "config.yaml")
    if os.path.exists(cfg_path):
        import yaml
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    mock = args.mock or cfg.get("mock", True)
    bank = build_bank(args.seed)
    truth = {}
    for f in FACTORS:
        for p, n in bank[f]["A_train"] + bank[f]["B_train"]:
            truth[p] = (f, +1); truth[n] = (f, -1)

    be = MockBackend(args.seed, truth=truth) if mock else HFBackend(
        cfg.get("model", "Qwen/Qwen2.5-1.5B"), float(cfg.get("layer_frac", 0.43)))
    if args.layer is not None and not mock:
        be.layer = args.layer

    # 1) 차이벡터 수집 (+ 요인 라벨, 템플릿군 라벨)
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

    # 2) PCA 유효 차원
    Xc = X - X.mean(0)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    print(f"\n=== V18 축 발견 (layer={getattr(be,'layer','mock')}, n={len(X)}) ===")
    print("[PCA] 상위 성분 설명분산:", [round(float(v), 3) for v in var[:8]])
    print(f"      90% 설명에 필요한 차원 수: {int(np.searchsorted(np.cumsum(var), 0.90)) + 1}")

    # 3-4) 군집 + ARI
    print("\n[군집]  k | 실루엣 | ARI(요인) | ARI(템플릿골격)")
    report = {"pca_var": [round(float(v), 4) for v in var[:10]], "grid": []}
    for k in range(2, 7):
        lab = kmeans(Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:10].T, k, args.seed)
        sil = silhouette(Xc, lab)
        a_f, a_t = ari(lab, flab), ari(lab, tlab)
        print(f"        {k} | {sil:6.3f} | {a_f:9.3f} | {a_t:9.3f}")
        report["grid"].append({"k": k, "silhouette": round(sil, 3),
                               "ari_factors": round(a_f, 3), "ari_templates": round(a_t, 3)})

    # 5) 요인 쌍 병합 경향: 요인별 평균 차이벡터 간 cos
    cents = {f: X[[i for i, l in enumerate(flab) if l == f]].mean(0) for f in set(flab)}
    for f in cents:
        cents[f] = cents[f] / (np.linalg.norm(cents[f]) + 1e-8)
    keys = FACTORS + ["person_ctrl"]
    print("\n[요인 중심 간 cos] (병합 후보 = 높은 값)")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            c = float(np.dot(cents[a], cents[b]))
            print(f"   {a:12s}–{b:12s}: {c:+.3f}")
            report[f"cos_{a}_{b}"] = round(c, 3)

    with open(os.path.join(HERE, "discover_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n판독 가이드: ARI(요인) 높음→MA1 / bene-conc 병합된 낮은 k 우세→MA2 /")
    print("            실루엣 전반 무의미→MA3 / ARI(템플릿)>ARI(요인)이면 어휘 교란 경고")
    print("상세: discover_report.json")


if __name__ == "__main__":
    main()
