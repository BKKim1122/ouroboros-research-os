"""V20 시나리오 C 감사 — para 붕괴가 'authoring 오류'인지 '진짜 프롬프트 의존'인지 자동 판별.

C가 참(내가 para를 잘못 써서 요인 의미가 망가짐)이라면 데이터에 남을 지문:
  지문1 [구조] para에서 4요인 쌍분리 구조 자체가 붕괴 (통제쌍 id|priv 등도 안 갈림)
  지문3 [요인별] 특정 요인의 self/other 판별(loo)이 orig 대비 뚝 떨어짐 → 그 요인이 의심
C가 거짓(요인 의미는 보존, 병합만 프롬프트 의존)이라면:
  통제쌍 분리는 orig≈para, 요인별 self-판별도 보존, ben|conc만 차이
지문2 [어휘] 원본 ben↔conc 어휘/구문 겹침 > para 겹침 → "원본 병합의 표면 겹침 기여" 정량화

사용: python audit_para.py --model Qwen/Qwen2.5-1.5B [--seeds 0 1 2] [--null-R 200]
mock: python audit_para.py --mock   (구조 보존으로 나와야 정상 — truth가 요인 그대로)
"""
from __future__ import annotations
import argparse, os, sys, json, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V18 = os.path.join(HERE, "..", "v18"); V19 = os.path.join(HERE, "..", "v19")
sys.path.insert(0, V18); sys.path.insert(0, V19); sys.path.insert(0, HERE)
from discover import pair_separability, loo_ncc_acc  # v18 코어
from discover_multi import HFBackend, MockBackend     # v19 백엔드
from prompts_v20 import get_bank_builder, FACTORS4

PAIRS = [("identity", "beneficiary"), ("identity", "privilege"), ("identity", "concern"),
         ("beneficiary", "privilege"), ("beneficiary", "concern"), ("privilege", "concern")]
CONTROL_PAIRS = [p for p in PAIRS if p != ("beneficiary", "concern")]


# ------------------------------------------------- 어휘 겹침 (지문 2, 모델 불필요)
_word = re.compile(r"[A-Za-z가-힣']+")
def _tokens(s):
    return set(w.lower() for w in _word.findall(s))

def lexical_overlap(bank, fa, fb):
    """요인 fa/fb의 self극 문장들 간 평균 자카드 유사도(쌍별 최대 매칭 아닌 전체 평균)."""
    A = [p for p, _ in bank[fa]["A_train"] + bank[fa]["B_train"]]
    B = [p for p, _ in bank[fb]["A_train"] + bank[fb]["B_train"]]
    sims = []
    for a in A:
        ta = _tokens(a)
        for b in B:
            tb = _tokens(b)
            u = ta | tb
            if u:
                sims.append(len(ta & tb) / len(u))
    return float(np.mean(sims))


# ------------------------------------------------- 표상 수집
def collect(be, bankfn, seed):
    bank = bankfn(seed)
    diffs, flab, poles = [], [], {}
    for fi, f in enumerate(FACTORS4):
        pairs = bank[f]["A_train"] + bank[f]["B_train"]
        S = be.acts([p for p, _ in pairs], f, +1)
        O = be.acts([n for _, n in pairs], f, -1)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-8)
        poles[f] = (S, O)
        D = S - O
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
        diffs.append(D); flab += [fi] * len(D)
    return np.vstack(diffs), np.array(flab), poles


def factor_self_loo(poles, f):
    """요인 f 안에서 self극 vs other극 LOO 판별 — 그 요인 문장군의 건강도."""
    S, O = poles[f]
    X = np.vstack([S, O]); y = np.array([1] * len(S) + [0] * len(O))
    return float(loo_ncc_acc(X, y))


def make_mock(seed, bankfn):
    bank = bankfn(seed); truth = {}
    for f in FACTORS4:
        for p, n in bank[f]["A_train"] + bank[f]["B_train"]:
            truth[p] = (f, +1); truth[n] = (f, -1)
    return MockBackend(seed, truth=truth)




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
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--null-R", type=int, default=200)
    ap.add_argument("--pct", type=float, default=0.95)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    banks = ["orig", "para", "ko"]
    # ---- 지문 2: 어휘 겹침 (모델 불필요, seed 0 뱅크로)
    print("=" * 66)
    print("[지문2 — ben↔conc 어휘 겹침(자카드), 통제=id↔priv]")
    lex = {}
    for bk in banks:
        bank = get_bank_builder(bk)(0)
        bc = lexical_overlap(bank, "beneficiary", "concern")
        ip = lexical_overlap(bank, "identity", "privilege")
        lex[bk] = {"ben_conc": round(bc, 4), "id_priv_control": round(ip, 4),
                   "excess": round(bc - ip, 4)}
        print(f"  {bk:5s}: ben↔conc {bc:.3f} | 통제 id↔priv {ip:.3f} | 초과 {bc-ip:+.3f}")
    if lex["orig"]["excess"] > lex["para"]["excess"] + 0.02:
        print("  → 원본의 ben/conc가 para보다 표면적으로 더 닮음: '원본 병합에 표면 겹침 기여' 지지")
    else:
        print("  → 원본/para 겹침 비슷: 병합 차이를 어휘 겹침만으로 설명 어려움")

    # ---- 모델 로드
    if args.mock:
        be = None
        print("\n⚠️  MOCK — 구조 보존으로 나와야 정상(truth가 요인 그대로)")
    else:
        be = HFBackend(args.model, 0.5, device=(None if args.device == "auto" else args.device))
        n = be.model.config.num_hidden_layers
        be.layer = max(1, int(n * 0.5))
        print(f"\n[모델] {args.model} layer={be.layer}")

    # ---- 지문 1+3: 뱅크별 구조
    report = {"model": ("mock" if args.mock else args.model), "lexical": lex, "banks": {}}
    for bk in banks:
        bankfn = get_bank_builder(bk)
        rows = []
        for s in args.seeds:
            be_s = make_mock(s, bankfn) if args.mock else be
            P, flab, poles = collect(be_s, bankfn, s)
            row = {"seed": s}
            # 지문3: 요인별 self/other 판별
            row["factor_self_loo"] = {f: round(factor_self_loo(poles, f), 4) for f in FACTORS4}
            # 지문1: 6쌍 분리 (null-상대)
            seps = {}
            for fa, fb in PAIRS:
                obs, gate, ok = pair_separability(P, flab, FACTORS4.index(fa),
                                                  FACTORS4.index(fb), args.null_R, s, args.pct)
                seps[f"{fa[:4]}|{fb[:4]}"] = {"obs": obs, "gate": gate, "pass": ok}
            row["pair_sep"] = seps
            rows.append(row)
        # 집계
        ctrl_pass = float(np.mean([[row["pair_sep"][f"{a[:4]}|{b[:4]}"]["pass"]
                                    for a, b in CONTROL_PAIRS] for row in rows]))
        bc_pass = float(np.mean([row["pair_sep"]["bene|conc"]["pass"] for row in rows]))
        floo = {f: round(float(np.mean([r["factor_self_loo"][f] for r in rows])), 4)
                for f in FACTORS4}
        report["banks"][bk] = {"control_pair_pass": round(ctrl_pass, 4),
                               "bene_conc_pass": round(bc_pass, 4),
                               "factor_self_loo_mean": floo, "per_seed": rows}
        print(f"\n[{bk}] 통제쌍(5쌍) 분리 통과율 {ctrl_pass:.2f} | ben|conc 통과율 {bc_pass:.2f}")
        print(f"     요인별 self판별: " + "  ".join(f"{f[:4]}={floo[f]}" for f in FACTORS4))

    # ---- 자동 판정
    o, p = report["banks"]["orig"], report["banks"]["para"]
    structure_preserved = p["control_pair_pass"] >= o["control_pair_pass"] - 0.2
    # 의심 요인: para에서 orig 대비 self판별이 0.15+ 떨어진 요인
    suspects = [f for f in FACTORS4
                if o["factor_self_loo_mean"][f] - p["factor_self_loo_mean"][f] > 0.15]
    print("\n" + "=" * 66)
    if structure_preserved and not suspects:
        verdict = ("C 기각 — para에서 요인 구조 보존(통제쌍 분리 유지, 요인 건강도 유지). "
                   "PROMPT_DEPENDENT는 authoring 오류가 아니라 실제 프롬프트 의존으로 판단.")
    elif suspects:
        verdict = (f"C 부분 지지 — 의심 요인 {suspects}: para에서 self판별 급락. "
                   "해당 요인 문장 재작성 후 V20 재실행 필요.")
    else:
        verdict = "C 지지 — para 구조 전반 붕괴(통제쌍도 안 갈림). para 뱅크 전면 재작성 필요."
    print(f"[감사 판정] {verdict}")
    report["verdict"] = verdict
    report["suspect_factors"] = suspects

    # 의심 요인 문장 출력(눈으로 볼 후보만)
    if suspects and not args.mock:
        print("\n[눈으로 확인할 문장 — 의심 요인의 para self극]")
        bank = get_bank_builder("para")(0)
        for f in suspects:
            print(f"  ({f})")
            for ptxt, _ in (bank[f]["A_train"] + bank[f]["B_train"])[:6]:
                print(f"   · {ptxt}")

    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    p = save_report(outdir, "audit_para_report", report)
    print(f"\n상세: {p} (+ _latest.json)")


if __name__ == "__main__":
    main()
