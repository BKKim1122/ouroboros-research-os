"""V17 파일럿 스윕 (동결 전 전용): layer_frac × alpha 격자 탐색.

사용:
  python sweep.py --seed 0                          # config.yaml의 모델로
  python sweep.py --seed 0 --layers 0.3,0.45,0.6,0.75 --alphas 4,8,16
  python sweep.py --seed 0 --mock                   # 파이프라인 검증

출력: 층별 probe 전이/혼동 + (층,alpha)별 대각/비대각 효과 표 + 권장 설정.
전 레이어 활성값을 한 번의 forward로 추출해 스윕 비용을 줄인다.
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_bank import build_bank
from run_seed import (FACTORS, MockBackend, HFBackend,
                      steering_vec, centroid_probe_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="0.3,0.45,0.6,0.75")
    ap.add_argument("--alphas", default="4,8,16")
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
        cfg.get("model", "Qwen/Qwen2.5-1.5B"), 0.5)
    n_layers = getattr(getattr(be, "model", None), "config", None)
    n_layers = n_layers.num_hidden_layers if n_layers else 10
    layer_fracs = [float(x) for x in args.layers.split(",")]
    layers = sorted({max(1, int(n_layers * lf)) for lf in layer_fracs})
    alphas = [float(x) for x in args.alphas.split(",")]

    # 전 레이어 활성값 일괄 추출
    A_acts, B_acts = {}, {}
    for f in FACTORS:
        ap_ = be.acts_multi([p for p, _ in bank[f]["A_train"]], layers)
        an_ = be.acts_multi([n for _, n in bank[f]["A_train"]], layers)
        bp_ = be.acts_multi([p for p, _ in bank[f]["B_train"]], layers)
        bn_ = be.acts_multi([n for _, n in bank[f]["B_train"]], layers)
        A_acts[f], B_acts[f] = (ap_, an_), (bp_, bn_)

    report = {"seed": args.seed, "n_layers": n_layers, "grid": []}
    print(f"\n모델 레이어 수: {n_layers} | 스윕 층: {layers} | alpha: {alphas}\n")
    for L in layers:
        vecs = {f: steering_vec(A_acts[f][0][L], A_acts[f][1][L]) for f in FACTORS}
        transfer = {f: centroid_probe_acc(A_acts[f][0][L], A_acts[f][1][L],
                                          B_acts[f][0][L], B_acts[f][1][L])
                    for f in FACTORS}
        offdiag = [centroid_probe_acc(A_acts[a][0][L], A_acts[a][1][L],
                                      B_acts[b][0][L], B_acts[b][1][L])
                   for a in FACTORS for b in FACTORS if a != b]
        rn = be.resid_norm([bank["identity"]["test"][0][0]], L) if not mock else 1.0
        print(f"— layer {L} (frac≈{L/n_layers:.2f})  잔차노름≈{rn:.1f}")
        print(f"   probe 전이: " + " ".join(f"{f[:4]}={transfer[f]:.2f}" for f in FACTORS)
              + f" | 혼동(비대각 평균)={np.mean(offdiag):.2f}")
        if not mock:
            be.layer = L
        base = {f: np.array([be.margin(it, f) for it in bank[f]["test"]]) for f in FACTORS}
        sd = {f: float(base[f].std() + 1e-6) for f in FACTORS}
        for alpha in alphas:
            diag, cross = [], []
            for a in FACTORS:
                for b in FACTORS:
                    st = np.array([be.margin_steered(it, b, vecs[a], alpha)
                                   for it in bank[b]["test"]])
                    e = abs(float((base[b] - st).mean())) / sd[b]
                    (diag if a == b else cross).append(e)
            neu0 = be.neutral_accuracy()
            neu1 = np.mean([be.neutral_accuracy(vecs[f], alpha) for f in FACTORS])
            spec = np.mean(diag) / max(max(cross), 1e-6)
            row = dict(layer=L, alpha=alpha, diag_mean=round(float(np.mean(diag)), 2),
                       cross_max=round(float(max(cross)), 2),
                       specificity=round(float(spec), 2),
                       neutral_damage=round(float(neu0 - neu1), 3),
                       probe_transfer=round(float(np.mean(list(transfer.values()))), 3),
                       probe_confusion_offdiag=round(float(np.mean(offdiag)), 3))
            report["grid"].append(row)
            print(f"   α={alpha:<5} 대각={row['diag_mean']:<6} 비대각max={row['cross_max']:<6} "
                  f"spec={row['specificity']:<6} 중립손상={row['neutral_damage']}")
    best = max(report["grid"], key=lambda r: (r["specificity"], -r["neutral_damage"]))
    print(f"\n권장: layer={best['layer']} (layer_frac≈{best['layer']/n_layers:.2f}), "
          f"alpha={best['alpha']}  → config.yaml에 반영 후 파일럿 재실행")
    with open("sweep_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("상세: sweep_report.json")


if __name__ == "__main__":
    main()
