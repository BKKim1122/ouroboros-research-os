"""V17 공용축 분해 분석 (파일럿 전용, 동결 전).

질문 세 가지:
  Q1. 요인들이 공유하는 축이 단순 '1인칭 대명사' 표면축인가?
      → 자기관련성 없는 중립 1인칭/3인칭 대조군(PERSON_CONTROL)으로 대명사축 추출,
        요인 판별을 얼마나 대신하는지 측정
  Q2. 공용축을 제거하면 요인별 잔여 구조가 남는가?
      → 대명사축/공유축 투영 제거 후 probe 혼동행렬 재계산
  Q3. 직교화된 벡터로 steering하면 선택성이 생기는가?
      → v_f ⊥ 공용축 벡터로 개입행렬 대각/비대각 재측정

사용:
  python decompose.py --layer 12 --alpha 8
  python decompose.py --mock            # 계약 검증 (수치는 무의미)
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_bank import build_bank, PERSON_CONTROL
from run_seed import (FACTORS, MockBackend, HFBackend,
                      steering_vec, centroid_probe_acc)



def project_out(X, u):
    u = u / (np.linalg.norm(u) + 1e-8)
    return X - np.outer(X @ u, u)


def confusion(A_acts, B_acts, transform=lambda x: x):
    return {a: {b: round(centroid_probe_acc(
        transform(A_acts[a][0]), transform(A_acts[a][1]),
        transform(B_acts[b][0]), transform(B_acts[b][1])), 3)
        for b in FACTORS} for a in FACTORS}


def summarize(C, label):
    diag = np.mean([C[f][f] for f in FACTORS])
    off = np.mean([C[a][b] for a in FACTORS for b in FACTORS if a != b])
    print(f"   {label}: 대각={diag:.2f}  비대각={off:.2f}  격차={diag-off:+.2f}")
    return {"diag": round(float(diag), 3), "offdiag": round(float(off), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    cfg = {}
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
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
    L = getattr(be, "layer", "mock")
    print(f"\n=== 공용축 분해 분석 (layer={L}, alpha={args.alpha}) ===")

    A_acts, B_acts, vecs = {}, {}, {}
    for f in FACTORS:
        ap_ = be.acts([p for p, _ in bank[f]["A_train"]], f, +1)
        an_ = be.acts([n for _, n in bank[f]["A_train"]], f, -1)
        bp_ = be.acts([p for p, _ in bank[f]["B_train"]], f, +1)
        bn_ = be.acts([n for _, n in bank[f]["B_train"]], f, -1)
        A_acts[f], B_acts[f] = (ap_, an_), (bp_, bn_)
        vecs[f] = steering_vec(ap_, an_)

    # 대명사 대조축
    pp = be.acts([p for p, _ in PERSON_CONTROL], FACTORS[0], +1)
    pn = be.acts([n for _, n in PERSON_CONTROL], FACTORS[0], -1)
    v_person = steering_vec(pp, pn)
    v_shared = np.mean([vecs[f] for f in FACTORS], axis=0)
    v_shared /= (np.linalg.norm(v_shared) + 1e-8)

    out = {"layer": L, "alpha": args.alpha}

    print("\n[Q0] 요인벡터 기하학")
    cos_fp = {f: round(float(np.dot(vecs[f], v_person)), 3) for f in FACTORS}
    cos_ff = {f"{a}-{b}": round(float(np.dot(vecs[a], vecs[b])), 3)
              for i, a in enumerate(FACTORS) for b in FACTORS[i+1:]}
    print("   요인↔대명사축 cos:", cos_fp)
    print("   요인 간 cos:", cos_ff)
    out["cos_factor_person"], out["cos_factor_factor"] = cos_fp, cos_ff

    print("\n[Q1] 대명사축이 요인 판별을 얼마나 대신하는가")
    person_conf = {b: round(centroid_probe_acc(pp, pn, B_acts[b][0], B_acts[b][1]), 3)
                   for b in FACTORS}
    print("   대명사 probe → 각 요인 B군 정확도:", person_conf)
    out["person_probe_on_factors"] = person_conf

    print("\n[Q2] 투영 제거 후 잔여 요인구조")
    out["confusion_raw"] = summarize(confusion(A_acts, B_acts), "원본           ")
    out["confusion_minus_person"] = summarize(
        confusion(A_acts, B_acts, lambda X: project_out(X, v_person)), "−대명사축      ")
    out["confusion_minus_shared"] = summarize(
        confusion(A_acts, B_acts, lambda X: project_out(X, v_shared)), "−공유축(요인평균)")

    print("\n[Q3] 직교화 steering 선택성")
    base = {f: np.array([be.margin(it, f) for it in bank[f]["test"]]) for f in FACTORS}
    sd = {f: float(base[f].std() + 1e-6) for f in FACTORS}
    for tag, vv in [("원본벡터", vecs),
                    ("직교화벡터", {f: (lambda w: w / (np.linalg.norm(w) + 1e-8))(
                        project_out(vecs[f][None, :], v_shared)[0]) for f in FACTORS})]:
        diag, cross = [], []
        for a in FACTORS:
            for b in FACTORS:
                st = np.array([be.margin_steered(it, b, vv[a], args.alpha)
                               for it in bank[b]["test"]])
                e = abs(float((base[b] - st).mean())) / sd[b]
                (diag if a == b else cross).append(e)
        spec = float(np.mean(diag) / max(max(cross), 1e-6))
        print(f"   {tag}: 대각={np.mean(diag):.2f} 비대각max={max(cross):.2f} spec={spec:.2f}")
        out[f"steering_{tag}"] = {"diag_mean": round(float(np.mean(diag)), 3),
                                  "cross_max": round(float(max(cross)), 3),
                                  "specificity": round(spec, 3)}

    with open("decompose_report.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n상세: decompose_report.json")


if __name__ == "__main__":
    main()
