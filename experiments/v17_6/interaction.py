"""V17.6 — privilege가 self-특이인가 일반 epistemic 축인가 (상호작용 검정).

self/other × privilege 2×2에서, self 내부의 privilege 방향과 other 내부의
privilege 방향이 같은지(서로 전이되는지) 본다. knower/referent 대명사 성분은
PERSON_CONTROL로 추정한 대명사 축을 투영 제거해 통제한다.

사전 약속 해석 (결과 전 봉인 — spec.yaml):
  H_general      : 대명사 통제 후 cross-referent transfer >= 0.75 & cos(priv 축) >= 0.5
                   → privilege는 self/other 공통 일반 epistemic-access 축. self-특이 아님.
  H_selfspecific : 통제 후 transfer <= 0.60  또는  (Δ_self − Δ_other) >= 0.15
                   → privilege가 self에서 다르게 조직됨. self-indexed→self-specific 근거.
  중간           : 부분 전이. 서술로 보고.

사용:
  python interaction.py                 # config.yaml 설정대로 (실모델/mock)
  python interaction.py --mock          # 파이프라인 검증 (수치 무의미)
출력: results/interaction_report.json + 콘솔 요약
"""
from __future__ import annotations
import argparse, os, sys, json, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V17_DIR = os.path.join(os.path.dirname(HERE), "v17")
sys.path.insert(0, V17_DIR)
sys.path.insert(0, HERE)
from run_seed import MockBackend, HFBackend, centroid_probe_acc, FACTORS  # noqa: E402
from so_prompts import build_so_bank, PERSON_CONTROL, CELLS               # noqa: E402


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def project_out(X, u):
    u = norm(u)
    return X - np.outer(X @ u, u)


def centroid_dir(pos, neg):
    return norm(pos.mean(0) - neg.mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    cfg = {}
    p = os.path.join(HERE, "config.yaml")
    if os.path.exists(p):
        import yaml
        cfg = yaml.safe_load(open(p, encoding="utf-8")) or {}
    mock = args.mock or cfg.get("mock", True)

    M = {k: [] for k in [
        "cos_priv_axes", "cos_priv_self_ref", "cos_priv_other_ref",
        "transfer_s2o", "transfer_o2s", "transfer_mean",
        "transfer_s2o_ctrl", "transfer_o2s_ctrl", "transfer_mean_ctrl",
        "within_self", "within_other", "delta_self_minus_other",
        "shuffle_baseline"]}

    for seed in args.seeds:
        bank = build_so_bank(seed)
        # mock: privilege pole만 심는다(direct=+1, inferred=-1). self/other는 구분 안 함
        #       → 파이프라인 검증용. 실모델은 truth 무시.
        truth = {}
        for split in ("train", "eval"):
            for c in CELLS:
                pole = +1 if c in ("sp", "op") else -1
                for t in bank[split][c]:
                    truth[t] = ("privilege", pole)
        be = (MockBackend(seed, truth=truth) if mock else
              HFBackend(cfg.get("model", "Qwen/Qwen2.5-1.5B"),
                        float(cfg.get("layer_frac", 0.43)), revision=cfg.get("revision")))

        def acts(texts):
            return be.acts(texts, "privilege", 0)

        A = {sp: {c: acts(bank[sp][c]) for c in CELLS} for sp in ("train", "eval")}

        # 대명사 축 (knower/referent 대명사 성분)
        pp = acts([x for x, _ in PERSON_CONTROL])
        pn = acts([y for _, y in PERSON_CONTROL])
        v_person = centroid_dir(pp, pn)

        # privilege 방향 (각 referent 내부, train)
        v_ps = centroid_dir(A["train"]["sp"], A["train"]["si"])
        v_po = centroid_dir(A["train"]["op"], A["train"]["oi"])
        # referent 방향
        v_ref = centroid_dir(np.vstack([A["train"]["sp"], A["train"]["si"]]),
                             np.vstack([A["train"]["op"], A["train"]["oi"]]))

        M["cos_priv_axes"].append(float(v_ps @ v_po))
        M["cos_priv_self_ref"].append(abs(float(v_ps @ v_ref)))
        M["cos_priv_other_ref"].append(abs(float(v_po @ v_ref)))

        # cross-referent transfer: self priv 방향으로 other priv(eval) 판별, 반대도
        t_s2o = centroid_probe_acc(A["train"]["sp"], A["train"]["si"],
                                   A["eval"]["op"], A["eval"]["oi"])
        t_o2s = centroid_probe_acc(A["train"]["op"], A["train"]["oi"],
                                   A["eval"]["sp"], A["eval"]["si"])
        M["transfer_s2o"].append(t_s2o); M["transfer_o2s"].append(t_o2s)
        M["transfer_mean"].append((t_s2o + t_o2s) / 2)

        # 대명사 통제판: v_person 투영 제거 후 재측정
        def po(X):
            return project_out(X, v_person)
        c_s2o = centroid_probe_acc(po(A["train"]["sp"]), po(A["train"]["si"]),
                                   po(A["eval"]["op"]), po(A["eval"]["oi"]))
        c_o2s = centroid_probe_acc(po(A["train"]["op"]), po(A["train"]["oi"]),
                                   po(A["eval"]["sp"]), po(A["eval"]["si"]))
        M["transfer_s2o_ctrl"].append(c_s2o); M["transfer_o2s_ctrl"].append(c_o2s)
        M["transfer_mean_ctrl"].append((c_s2o + c_o2s) / 2)

        # within-referent 판별 (상한): 같은 referent train→eval
        w_s = centroid_probe_acc(A["train"]["sp"], A["train"]["si"],
                                 A["eval"]["sp"], A["eval"]["si"])
        w_o = centroid_probe_acc(A["train"]["op"], A["train"]["oi"],
                                 A["eval"]["op"], A["eval"]["oi"])
        M["within_self"].append(w_s); M["within_other"].append(w_o)
        M["delta_self_minus_other"].append(w_s - w_o)

        # 대조: 라벨 셔플 (cross-referent transfer의 우연선 ~0.5 확인)
        rng = np.random.default_rng(seed)
        mixed = np.vstack([A["train"]["sp"], A["train"]["si"]])
        idx = rng.permutation(len(mixed)); half = len(mixed) // 2
        sh = centroid_probe_acc(mixed[idx[:half]], mixed[idx[half:]],
                                A["eval"]["sp"], A["eval"]["si"])
        M["shuffle_baseline"].append(sh)

    def ms(xs):
        return round(st.mean(xs), 3), round(st.stdev(xs) if len(xs) > 1 else 0.0, 3)

    report = {"mock": mock, "seeds": args.seeds, "layer_frac": cfg.get("layer_frac"),
              "metrics": {k: {"mean": ms(v)[0], "sd": ms(v)[1]} for k, v in M.items()}}

    tm_ctrl = report["metrics"]["transfer_mean_ctrl"]["mean"]
    cos_ax = report["metrics"]["cos_priv_axes"]["mean"]
    d_so = report["metrics"]["delta_self_minus_other"]["mean"]
    if tm_ctrl >= 0.75 and cos_ax >= 0.5:
        verdict = ("H_general: privilege는 self/other 공통 일반 epistemic-access 축. "
                   f"대명사 통제 후 cross-referent transfer={tm_ctrl}, cos(priv축)={cos_ax}. "
                   "→ self-특이 아님. 'self-related'까지, 'self-specific' 미지지.")
    elif tm_ctrl <= 0.60 or d_so >= 0.15:
        verdict = ("H_selfspecific: privilege가 self에서 다르게 조직됨 (통제 후 transfer="
                   f"{tm_ctrl}, Δself−Δother={d_so}). → self-indexed에서 self-specific로 "
                   "넘어갈 직접 근거. 단 nuisance 완전통제(약한고리 A)는 여전히 필요.")
    else:
        verdict = (f"중간/부분 전이 (통제 후 transfer={tm_ctrl}, cos={cos_ax}, "
                   f"Δself−Δother={d_so}). privilege는 부분적으로만 self/other 공통. 서술 필요.")
    report["verdict_exploratory"] = verdict

    print(f"\n=== self/other × privilege 상호작용 (mock={mock}, seeds={args.seeds}) ===")
    order = ["cos_priv_axes", "cos_priv_self_ref", "cos_priv_other_ref",
             "transfer_mean", "transfer_mean_ctrl", "within_self", "within_other",
             "delta_self_minus_other", "shuffle_baseline"]
    for k in order:
        mn, sd = report["metrics"][k]["mean"], report["metrics"][k]["sd"]
        print(f"  {k:26s} = {mn:+.3f} (±{sd:.3f})")
    print("\n[탐색적 판정]", report["verdict_exploratory"])

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    outp = os.path.join(HERE, "results", "interaction_report.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("상세:", outp)


if __name__ == "__main__":
    main()
