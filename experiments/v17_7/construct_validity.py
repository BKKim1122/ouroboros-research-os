"""V17.7-b — nuisance 방향 construct validity 진단.

v17_7의 H_beyond("4요인 전이는 6 nuisance로 환원 안 됨")가 실질인지 확인한다.
남은 구멍: 제거한 nuisance 방향이 애초에 '진짜 그 nuisance를 잡는 방향'이었나?
방향이 부실하면 제거해도 4요인에 영향 없는 게 당연하므로 H_beyond가 약해진다.

두 가지를 잰다 (진단이라 kill criterion 없음 — 서술):
  1. self_acc  : 각 nuisance 축을 train/test로 나눠, train 방향으로 held-out 극성쌍을
                 판별하는 정확도. 높으면(우연 0.5 대비) 방향이 유효.
  2. cos_to_factor : 각 nuisance 방향과 4요인 방향의 최대 |cos|. 낮으면 4요인과 직교
                 → 제거해도 4요인 전이가 안 변한 이유가 '무력'이 아니라 '무관'임을 뒷받침.

분석자 사전 기대(봉인): nuisance 방향은 유효(self_acc 높음)하고 4요인과 대체로
직교(cos 낮음)할 것. 이러면 H_beyond가 '방향 유효한데 4요인과 무관'으로 확정된다.
반대로 self_acc가 우연 수준이면 v17_7을 '방향 부실'로 재해석해야 한다. 예단이며,
수치를 그대로 보고한다.

사용:
  python construct_validity.py            # config대로
  python construct_validity.py --mock     # 파이프라인 검증
"""
from __future__ import annotations
import argparse, os, sys, json, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V17_DIR = os.path.join(os.path.dirname(HERE), "v17")
sys.path.insert(0, V17_DIR); sys.path.insert(0, HERE)
from run_seed import FACTORS, MockBackend, HFBackend, steering_vec, centroid_probe_acc  # noqa: E402
from prompts_bank import build_bank                                                     # noqa: E402
from nuisance_prompts import AXES, build_nuisance                                       # noqa: E402


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

    self_acc = {ax: [] for ax in AXES}
    shuffle = {ax: [] for ax in AXES}
    cos_fac = {ax: [] for ax in AXES}

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
        acts = lambda t, f="identity", pole=0: be.acts(t, f, pole)

        fdirs = {f: steering_vec(acts([p for p, _ in bank[f]["A_train"]], f, +1),
                                 acts([n for _, n in bank[f]["A_train"]], f, -1))
                 for f in FACTORS}

        rng = np.random.default_rng(seed)
        for ax in AXES:
            pairs = nb[ax]; h = len(pairs) // 2
            tr, te = pairs[:h], pairs[h:]
            trp = acts([p for p, _ in tr]); trn = acts([n for _, n in tr])
            tep = acts([p for p, _ in te]); ten = acts([n for _, n in te])
            v = steering_vec(trp, trn)
            self_acc[ax].append(centroid_probe_acc(trp, trn, tep, ten))
            cos_fac[ax].append(max(abs(float(v @ fdirs[f])) for f in FACTORS))
            mixed = np.vstack([trp, trn]); idx = rng.permutation(len(mixed)); hh = len(mixed) // 2
            shuffle[ax].append(centroid_probe_acc(mixed[idx[:hh]], mixed[idx[hh:]], tep, ten))

    def m(xs):
        return round(st.mean(xs), 3)

    rows = [(ax, m(self_acc[ax]), m(shuffle[ax]), m(cos_fac[ax])) for ax in AXES]
    report = {"mock": mock, "seeds": args.seeds,
              "by_axis": {ax: {"self_acc": m(self_acc[ax]), "shuffle": m(shuffle[ax]),
                               "cos_to_factor_max": m(cos_fac[ax])} for ax in AXES},
              "overall": {"self_acc_mean": m([x for ax in AXES for x in self_acc[ax]]),
                          "cos_to_factor_mean": m([x for ax in AXES for x in cos_fac[ax]])}}

    print(f"\n=== nuisance 방향 construct validity (mock={mock}) ===")
    print(f"{'축':<14} self_acc  shuffle  |cos|_to_factor")
    for ax, sa, sh, cf in rows:
        print(f"{ax:<14} {sa:>7.3f}  {sh:>6.3f}  {cf:>10.3f}")
    ov = report["overall"]
    print(f"\n  전체 self_acc={ov['self_acc_mean']} (우연 0.5) | "
          f"4요인과 |cos| 평균={ov['cos_to_factor_mean']}")
    if ov["self_acc_mean"] >= 0.7 and ov["cos_to_factor_mean"] <= 0.4:
        note = ("→ nuisance 방향 유효 + 4요인과 대체로 직교. v17_7 H_beyond가 "
                "'방향 멀쩡한데 4요인과 무관'으로 확정된다.")
    elif ov["self_acc_mean"] < 0.6:
        note = ("→ nuisance 방향의 self-판별력이 우연 수준. v17_7 제거가 무의미했을 수 "
                "있어 H_beyond를 '방향 부실'로 재해석해야 한다.")
    else:
        note = "→ 중간. 수치를 그대로 서술."
    print(note)
    report["diagnostic_note"] = note

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    outp = os.path.join(HERE, "results", "construct_validity_report.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("상세:", outp)


if __name__ == "__main__":
    main()
