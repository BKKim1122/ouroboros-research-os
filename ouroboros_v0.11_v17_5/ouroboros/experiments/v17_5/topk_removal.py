"""V17.5 — 공유 부분공간 top-k 제거 강건성 분석 (탐색적, V17 동결 밖).

검토 지적(약한고리 C) 대응: V17의 projection_gap(+0.397)은 '4요인 방향의
산술평균 1축(rank-1)'만 제거하고 얻은 값이다. 공유구조가 2~3차원이면 평균축에
직교하는 공통성분이 남아 gap이 factor-specific 잔여가 아닐 수 있다.

이 스크립트는 공유축을 rank-k로 일반화해 gap(k) 곡선을 그린다. 핵심 방법:
  - 방향 추출은 V17과 동일(요인별 A_train 극성 평균차, 정규화).
  - 교차검증: 공유 부분공간 U_k를 '다른 seed들'의 방향으로 추정하고,
    held-out test seed의 활성치에서 제거 후 gap 측정
    (같은 데이터로 추정·평가하는 낙관 편향 제거).
  - random 대조: 같은 차원 k의 무작위 정규직교 부분공간 제거와 비교
    (아무 k차원이나 지워도 gap이 준다는 대안 배제).

경합 해석 (spec.yaml 참조):
  MC1 rank-1 공유축   → gap이 k=1에서 대부분 소멸.
  MC2 저차원 공유공간 → k=2,3에서 소멸.
  MC3 요인 잔여 구조   → top-k 제거 후에도 비영점 plateau + random 대비 유의 우위.

사용:
  python topk_removal.py --kmax 6            # config.yaml 설정대로 (실모델/mock)
  python topk_removal.py --mock --kmax 6     # 파이프라인 검증 (수치 무의미)
출력: results/topk_report.json + 콘솔 요약표
"""
from __future__ import annotations
import argparse, os, sys, json, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V17_DIR = os.path.join(os.path.dirname(HERE), "v17")
sys.path.insert(0, V17_DIR)  # V17의 백엔드·프롬프트·probe 재사용
from prompts_bank import build_bank                       # noqa: E402
from run_seed import (FACTORS, MockBackend, HFBackend,     # noqa: E402
                      steering_vec, centroid_probe_acc)


def project_out_subspace(X, U):
    """활성치 X(n,dim)에서 정규직교 기저 U(dim,k)가 이루는 부분공간을 제거."""
    if U.shape[1] == 0:
        return X
    return X - (X @ U) @ U.T


def orthonormal_from(D):
    """방향 행렬 D(m,dim)의 top 특이벡터들을 정규직교 기저로. 반환 (dim, r)."""
    # 평균을 빼지 않는다: '방향들이 공통으로 뻗는 축'(공유 성분)을 원함.
    # (평균 제거 시 공유 성분이 사라져 잘못된 부분공간이 됨)
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt.T, s  # 열이 우측특이벡터, s는 특이값


def gap_after_removal(A_acts, B_acts, U):
    """U 부분공간 제거 후 cross-template 판별 대각-비대각 격차."""
    def tf(X):
        return project_out_subspace(X, U)
    diag = np.mean([centroid_probe_acc(tf(A_acts[f][0]), tf(A_acts[f][1]),
                                       tf(B_acts[f][0]), tf(B_acts[f][1]))
                    for f in FACTORS])
    off = np.mean([centroid_probe_acc(tf(A_acts[a][0]), tf(A_acts[a][1]),
                                      tf(B_acts[b][0]), tf(B_acts[b][1]))
                   for a in FACTORS for b in FACTORS if a != b])
    return float(diag - off), float(diag), float(off)


def random_subspace(dim, k, rng):
    if k == 0:
        return np.zeros((dim, 0))
    M = rng.normal(size=(dim, k))
    Q, _ = np.linalg.qr(M)
    return Q[:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--n_rand", type=int, default=20)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    cfg = {}
    cfg_path = os.path.join(HERE, "config.yaml")
    if os.path.exists(cfg_path):
        import yaml
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    mock = args.mock or cfg.get("mock", True)

    # 각 seed: 4요인 방향 + A/B 활성치를 한 번 계산해 캐시
    dirs, acts = {}, {}
    for s in args.seeds:
        bank = build_bank(s)
        truth = {}
        for f in FACTORS:
            for p, n in bank[f]["A_train"] + bank[f]["B_train"]:
                truth[p] = (f, +1); truth[n] = (f, -1)
        be = (MockBackend(s, truth=truth) if mock else
              HFBackend(cfg.get("model", "Qwen/Qwen2.5-1.5B"),
                        float(cfg.get("layer_frac", 0.43)),
                        revision=cfg.get("revision")))
        A_acts, B_acts, vv = {}, {}, {}
        for f in FACTORS:
            ap_ = be.acts([p for p, _ in bank[f]["A_train"]], f, +1)
            an_ = be.acts([n for _, n in bank[f]["A_train"]], f, -1)
            bp_ = be.acts([p for p, _ in bank[f]["B_train"]], f, +1)
            bn_ = be.acts([n for _, n in bank[f]["B_train"]], f, -1)
            A_acts[f], B_acts[f] = (ap_, an_), (bp_, bn_)
            vv[f] = steering_vec(ap_, an_)
        dirs[s] = vv
        acts[s] = (A_acts, B_acts)
    dim = next(iter(dirs[args.seeds[0]].values())).shape[0]
    rng = np.random.default_rng(0)

    # 교차검증 LOO: test seed 제외한 방향들로 공유 부분공간 추정 → test에서 제거
    per_k = {k: {"shared": [], "random": []} for k in range(args.kmax + 1)}
    diag_off = {0: {"diag": [], "off": []}}
    for test_s in args.seeds:
        train_D = np.stack([dirs[s][f] for s in args.seeds if s != test_s
                            for f in FACTORS])
        U_full, _ = orthonormal_from(train_D)
        A_acts, B_acts = acts[test_s]
        for k in range(args.kmax + 1):
            Uk = U_full[:, :k]
            g, d, o = gap_after_removal(A_acts, B_acts, Uk)
            per_k[k]["shared"].append(g)
            if k == 0:
                diag_off[0]["diag"].append(d); diag_off[0]["off"].append(o)
            # random 대조 (동일 차원 k)
            rg = [gap_after_removal(A_acts, B_acts, random_subspace(dim, k, rng))[0]
                  for _ in range(args.n_rand)]
            per_k[k]["random"].append(float(np.mean(rg)))

    def ms(xs):
        return (round(st.mean(xs), 3), round(st.stdev(xs) if len(xs) > 1 else 0.0, 3))

    rows, report = [], {"mock": mock, "seeds": args.seeds, "kmax": args.kmax,
                        "layer_frac": cfg.get("layer_frac"), "by_k": {}}
    for k in range(args.kmax + 1):
        sm, ssd = ms(per_k[k]["shared"])
        rm, rsd = ms(per_k[k]["random"])
        rows.append((k, sm, ssd, rm, rsd, round(sm - rm, 3)))
        report["by_k"][k] = {"shared_gap_mean": sm, "shared_gap_sd": ssd,
                             "random_gap_mean": rm, "random_gap_sd": rsd,
                             "shared_minus_random": round(sm - rm, 3)}
    report["k0_diag_mean"] = round(st.mean(diag_off[0]["diag"]), 3)
    report["k0_off_mean"] = round(st.mean(diag_off[0]["off"]), 3)

    # 사전 약속한 해석 라벨 (양방향 — 사후 스핀 방지)
    g0 = report["by_k"][0]["shared_gap_mean"]
    g1 = report["by_k"][1]["shared_gap_mean"]
    plateau_k = max(0, args.kmax)
    tail = [report["by_k"][k]["shared_gap_mean"] for k in range(2, args.kmax + 1)]
    tail_min = min(tail) if tail else g1
    tail_vs_rand = min(report["by_k"][k]["shared_minus_random"]
                       for k in range(2, args.kmax + 1)) if args.kmax >= 2 else (g1 - report["by_k"][1]["random_gap_mean"])
    if g1 <= 0.05:
        verdict = "MC1: rank-1 공유축 — gap이 k=1에서 소멸. C3는 평균축 artifact."
    elif tail_min <= 0.05:
        verdict = "MC2: 저차원 공유공간 — gap이 k>=2에서 소멸. C3 약화."
    elif tail_min > 0.05 and tail_vs_rand > 0.05:
        verdict = ("MC3: 요인 잔여 구조 — top-k 제거 후에도 비영점 plateau + "
                   "random 대비 우위. C3(공유축 제거 후 요인정보 잔여) 강건.")
    else:
        verdict = ("판정 보류: plateau는 있으나 random 대조 대비 우위가 불충분 "
                   "(shared-random <= 0.05). gap이 저차원 제거 일반효과일 수 있음.")
    report["verdict_exploratory"] = verdict

    print(f"\n=== top-k 공유부분공간 제거 (mock={mock}, seeds={args.seeds}) ===")
    print(f"k=0 원본: 대각={report['k0_diag_mean']}  비대각={report['k0_off_mean']}  "
          f"격차={report['by_k'][0]['shared_gap_mean']}")
    print(f"{'k':>2} | shared gap(±sd) | random gap(±sd) | shared−random")
    for k, sm, ssd, rm, rsd, d in rows:
        print(f"{k:>2} |  {sm:+.3f} (±{ssd:.3f}) |  {rm:+.3f} (±{rsd:.3f}) |  {d:+.3f}")
    print("\n[탐색적 판정]", verdict)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    outp = os.path.join(HERE, "results", "topk_report.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("상세:", outp)


if __name__ == "__main__":
    main()
