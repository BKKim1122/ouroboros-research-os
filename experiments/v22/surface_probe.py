"""V22: 표면 환원 통제 — 나/남 축이 '대명사 토큰 통계'인가 '통합된 지시 표상'인가.

배경: V17~V20에서 유일하게 전 조건을 견딘 것이 self/other 구분(self_acc)이다. 그러나
최소쌍의 두 문장은 대명사/이름만 다르므로, 평균 풀링에 그 토큰 임베딩 차이가 그대로
실렸을 수 있다(SURFACE). 그러면 '지시적 핵'은 대명사 분류기로 격하된다. V22는 이를 가른다.

측정 3종:
  1) diff-마스킹 풀링(핵심): 쌍 내 두 문장이 '다른' 문자 구간(=대명사/이름부)을 찾아
     그 토큰들을 풀링에서 제외하고 self_acc 재측정. 남는 토큰은 문자열 동일 →
     거기서 판별되면 지시 정보가 attention으로 문장 전체에 통합된 것(INDEXICAL).
  2) layer-0 대비(내장 sanity): 임베딩층에선 같은 단어=같은 벡터(RoPE라 위치정보 없음)
     → 마스킹 후 layer0 self_acc는 우연이어야 정상(넘으면 마스킹 결함 플래그).
     비마스킹 layer0은 표면 기준선(높게 나올 것).
  3) pro-drop 전이(ko 보너스): 대명사 토큰이 0개인 한국어 화자함축/전문함축 쌍에,
     대명사 뱅크(BANK_KO)로 학습한 방향이 전이되는가. 전이되면 표면 환원 최강 기각.

봉인 판정(결과 전):
  INDEXICAL_INTEGRATED : 마스킹 중간층 self_acc > 라벨셔플 null 95pct 가
                         orig·para 두 뱅크에서 seed consistency>=0.75 (ko·prodrop은 보강)
  SURFACE_ONLY         : 마스킹 중간층 붕괴(null 이내) AND prodrop 전이 실패
  INDEXICAL_CONTEXTUAL : 마스킹 통과하나 prodrop 실패 — 통합은 있으나 대명사 계기 필요(중간)
  MASK_DEFECT          : 마스킹 layer0 sanity 실패 → 도구 결함, 판정 보류

mock: 두 합성 시나리오(SURFACE: 대명사 토큰에만 신호 / INTEGRATED: 전 토큰에 신호)로
파이프라인 판별력 검증. 확증은 mock 금지.

사용:
  python surface_probe.py --mock                       # 파이프라인 판별력 검증
  python surface_probe.py --models qwen25_1p5b --null-R 200
"""
from __future__ import annotations
import argparse, os, sys, json, difflib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V18 = os.path.join(HERE, "..", "v18"); V20 = os.path.join(HERE, "..", "v20")
sys.path.insert(0, V18); sys.path.insert(0, V20); sys.path.insert(0, HERE)
from discover import loo_ncc_acc                    # v18 코어
from prompts_v20 import get_bank_builder, FACTORS4  # v20 뱅크(orig/para/ko)
from prompts_v22 import PRODROP_KO


# ------------------------------------------------------------- diff 마스크
def diff_char_spans(a: str, b: str):
    """a에서 b와 '다른' 문자 구간들 [(s,e)...] — 대명사/이름부."""
    sm = difflib.SequenceMatcher(None, a, b)
    spans, pos = [], 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal" and i2 > i1:
            spans.append((i1, i2))
    return spans


class TokenBackend:
    """토큰별 hidden + 문자오프셋 마스킹 풀링. (mean-pool 대신 선택 풀링)"""
    def __init__(self, name, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            self.tok = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32, local_files_only=True).eval()
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.n_layers = self.model.config.num_hidden_layers
        self.d = self.model.config.hidden_size

    @torch.no_grad()
    def pooled(self, text, layer, mask_spans=None):
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        ids = {k: v.to(self.device) for k, v in enc.items()}
        hs = self.model(**ids, output_hidden_states=True).hidden_states[layer][0]
        keep = []
        for ti, (s, e) in enumerate(offs):
            if e <= s:            # special token
                continue
            masked = mask_spans and any(not (e <= ms or s >= me) for ms, me in mask_spans)
            if not masked:
                keep.append(ti)
        if not keep:              # 전부 마스킹된 극단 — 전체 평균 fallback + 플래그
            keep = [ti for ti, (s, e) in enumerate(offs) if e > s]
        v = hs[keep].mean(0).float().cpu().numpy()
        return v, len(keep)


class MockTokenBackend:
    """합성 검증: scenario='surface'면 대명사(=diff) 토큰에만 self신호,
    'integrated'면 모든 토큰에 신호. 토크나이즈는 문자 단위 흉내."""
    def __init__(self, scenario, seed=0, d=64):
        self.scenario, self.d, self.n_layers = scenario, d, 12
        rng = np.random.default_rng(seed)
        self.sig = rng.standard_normal(d); self.sig /= np.linalg.norm(self.sig)

    def pooled(self, text, layer, mask_spans=None):
        rng = np.random.default_rng(abs(hash(text)) % 10**8)
        n = max(4, len(text) // 3)
        toks = []
        step = max(1, len(text) // n)
        pos = 0
        # self극 판별용: 텍스트에 SELFMARK 포함 여부(호출측이 넣음)
        is_self = "§S" in text
        body = text.replace("§S", "").replace("§O", "")
        for i in range(n):
            s, e = pos, min(len(body), pos + step); pos = e
            in_diff = mask_spans and any(not (e <= ms or s >= me) for ms, me in mask_spans)
            base = rng.standard_normal(self.d) * 0.6
            pol = (1.0 if is_self else -1.0) * self.sig
            if layer == 0:
                # 임베딩층: diff 토큰만 극성 반영(단어 자체가 다르므로), 나머진 동일단어=동일벡터
                v = base + (pol * 3.0 if in_diff else 0.0)
            else:
                if self.scenario == "surface":
                    v = base + (pol * 3.0 if in_diff else 0.0)   # 통합 안 됨
                else:
                    v = base + pol * 1.5                          # 전 토큰에 통합
            masked = mask_spans and in_diff
            if not masked:
                toks.append(v)
        if not toks:
            toks = [rng.standard_normal(self.d)]
        return np.mean(toks, 0), len(toks)


# ------------------------------------------------------------- 측정
def bank_pairs(bankname, seed):
    bank = get_bank_builder(bankname)(seed)
    pairs = []
    for f in FACTORS4:
        pairs += bank[f]["A_train"] + bank[f]["B_train"]
    return pairs


def masked_self_acc(be, pairs, layer, masked, mock=False):
    X, y, kept = [], [], []
    for p, n in pairs:
        spans_p = diff_char_spans(p, n); spans_n = diff_char_spans(n, p)
        tp, tn = (("§S" + p), ("§O" + n)) if mock else (p, n)
        vp, kp = be.pooled(tp, layer, spans_p if masked else None)
        vn, kn = be.pooled(tn, layer, spans_n if masked else None)
        X += [vp, vn]; y += [1, 0]; kept += [kp, kn]
    X = np.stack(X); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return float(loo_ncc_acc(X, np.array(y))), float(np.mean(kept))


def acc_null_gate(acc_fn, R, seed, pct, X=None, y=None):
    """라벨셔플 null: X,y 재사용 형태가 아니라 acc 재계산이 비싸므로,
    여기선 X,y를 받아 셔플 LOO만 돌린다."""
    rng = np.random.default_rng(seed + 777)
    null = [loo_ncc_acc(X, y[rng.permutation(len(y))]) for _ in range(R)]
    return float(np.quantile(null, pct))


def masked_self_acc_with_null(be, pairs, layer, masked, R, seed, pct, mock=False):
    X, y = [], []
    for p, n in pairs:
        spans_p = diff_char_spans(p, n); spans_n = diff_char_spans(n, p)
        tp, tn = (("§S" + p), ("§O" + n)) if mock else (p, n)
        vp, _ = be.pooled(tp, layer, spans_p if masked else None)
        vn, _ = be.pooled(tn, layer, spans_n if masked else None)
        X += [vp, vn]; y += [1, 0]
    X = np.stack(X); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    y = np.array(y)
    obs = float(loo_ncc_acc(X, y))
    gate = acc_null_gate(None, R, seed, pct, X, y)
    return round(obs, 4), round(gate, 4), bool(obs > gate)


def prodrop_transfer(be, layer, seed, R, pct, mock=False):
    """BANK_KO(대명사 명시)로 self/other 방향 학습 → PRODROP(대명사 0)에 적용."""
    ko_pairs = bank_pairs("ko", seed)
    S = np.stack([be.pooled(("§S" + p) if mock else p, layer)[0] for p, _ in ko_pairs])
    O = np.stack([be.pooled(("§O" + n) if mock else n, layer)[0] for _, n in ko_pairs])
    w = S.mean(0) - O.mean(0); w = w / (np.linalg.norm(w) + 1e-8)
    Xt, yt = [], []
    for ps, po in PRODROP_KO:
        Xt += [be.pooled(("§S" + ps) if mock else ps, layer)[0],
               be.pooled(("§O" + po) if mock else po, layer)[0]]
        yt += [1, 0]
    Xt = np.stack(Xt); yt = np.array(yt)
    proj = Xt @ w
    thr = float(np.median(proj))
    obs = float(np.mean((proj > thr).astype(int) == yt))
    rng = np.random.default_rng(seed + 555)
    null = [float(np.mean((proj > thr).astype(int) == yt[rng.permutation(len(yt))]))
            for _ in range(R)]
    gate = float(np.quantile(null, pct))
    return round(obs, 4), round(gate, 4), bool(obs > gate)


# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--null-R", type=int, default=None)
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    args = ap.parse_args()
    import yaml, time
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    if args.mock and args.mode == "confirm":
        print("❌ 확증에서 mock 금지."); sys.exit(2)
    R = args.null_R or spec["null_R"]
    pct = spec["null_pct"]
    seeds = spec["pilot_seeds"] if args.mode == "pilot" else spec["confirmatory_seeds"]
    banks = spec["banks"]; cons = spec["stats"]["consistency_min"]
    models = spec["models"]
    if args.models:
        models = [m for m in models if m["tag"] in args.models]

    def analyze(be, mid_layer, mock):
        rows = []
        for s in seeds:
            row = {"seed": s, "banks": {}}
            for bk in banks:
                pairs = bank_pairs(bk, s)
                um_mid = masked_self_acc_with_null(be, pairs, mid_layer, False, R, s, pct, mock)
                mk_mid = masked_self_acc_with_null(be, pairs, mid_layer, True, R, s, pct, mock)
                um_l0 = masked_self_acc_with_null(be, pairs, 0, False, min(R, 100), s, pct, mock)
                mk_l0 = masked_self_acc_with_null(be, pairs, 0, True, min(R, 100), s, pct, mock)
                row["banks"][bk] = {
                    "unmasked_mid": um_mid[0], "masked_mid": mk_mid[0],
                    "masked_mid_gate": mk_mid[1], "masked_mid_pass": mk_mid[2],
                    "unmasked_L0": um_l0[0], "masked_L0": mk_l0[0],
                    "masked_L0_gate": mk_l0[1], "mask_sanity_ok": (not mk_l0[2]),
                }
            pd = prodrop_transfer(be, mid_layer, s, R, pct, mock)
            row["prodrop"] = {"acc": pd[0], "gate": pd[1], "pass": pd[2]}
            rows.append(row)
        return rows

    grid = []
    for m in models:
        tag = m["tag"]
        print(f"\n▶ {tag} …", flush=True)
        t0 = time.time()
        try:
            if args.mock:
                results = {}
                for scen in ["surface", "integrated"]:
                    be = MockTokenBackend(scen)
                    mid = be.n_layers // 2
                    results[scen] = analyze(be, mid, True)
                grid.append({"tag": tag, "mock_scenarios": results}); continue
            be = TokenBackend(m["name"], device=(None if args.device == "auto" else args.device))
            mid = max(1, int(be.n_layers * spec["layer_frac"]))
            rows = analyze(be, mid, False)
        except Exception as e:
            grid.append({"tag": tag, "status": "load_failed", "error": str(e)[:150]})
            print(f"   ❌ {str(e)[:120]}"); continue

        # 모델 집계
        def frac(get):
            return round(float(np.mean([get(r) for r in rows])), 4)
        agg = {"tag": tag, "status": "ok", "mid_layer": mid, "per_seed": rows}
        for bk in banks:
            agg[f"{bk}_masked_pass_frac"] = frac(lambda r: r["banks"][bk]["masked_mid_pass"])
            agg[f"{bk}_mask_sanity_frac"] = frac(lambda r: r["banks"][bk]["mask_sanity_ok"])
            agg[f"{bk}_masked_mid_mean"] = frac(lambda r: r["banks"][bk]["masked_mid"])
            agg[f"{bk}_unmasked_L0_mean"] = frac(lambda r: r["banks"][bk]["unmasked_L0"])
        agg["prodrop_pass_frac"] = frac(lambda r: r["prodrop"]["pass"])
        agg["secs"] = round(time.time() - t0, 1)
        grid.append(agg)
        line = " | ".join(f"{bk}: masked {agg[f'{bk}_masked_mid_mean']}"
                          f"(pass {agg[f'{bk}_masked_pass_frac']})" for bk in banks)
        print(f"   {line} | prodrop pass {agg['prodrop_pass_frac']}  [{agg['secs']}s]")

    # 판정 (mock은 시나리오 판별력만 출력)
    if args.mock:
        print("\n=== MOCK 판별력 검증 ===")
        for scen in ["surface", "integrated"]:
            rows = grid[0]["mock_scenarios"][scen]
            mp = np.mean([r["banks"]["orig"]["masked_mid_pass"] for r in rows])
            sane = np.mean([r["banks"]["orig"]["mask_sanity_ok"] for r in rows])
            pdp = np.mean([r["prodrop"]["pass"] for r in rows])
            print(f"  [{scen:10s}] masked_mid pass {mp:.2f} (기대: surface=0, integrated=1) "
                  f"| L0 sanity {sane:.2f} | prodrop {pdp:.2f}")
        return

    ok = [g for g in grid if g.get("status") == "ok"]
    sanity_all = all(g[f"{bk}_mask_sanity_frac"] >= cons for g in ok for bk in banks)
    def model_pass(g):
        return (g["orig_masked_pass_frac"] >= cons and g["para_masked_pass_frac"] >= cons)
    idx_frac = round(float(np.mean([model_pass(g) for g in ok])), 4) if ok else None
    pd_frac = round(float(np.mean([g["prodrop_pass_frac"] >= cons for g in ok])), 4) if ok else None

    if not sanity_all:
        verdict = "MASK_DEFECT — layer0 sanity 실패(마스킹 결함), 판정 보류"
    elif idx_frac is not None and idx_frac >= spec["replication_min"]:
        verdict = ("INDEXICAL_INTEGRATED — 대명사 토큰 제거 후에도 중간층 판별 유지"
                   + (" + prodrop 전이 성립(강화)" if pd_frac and pd_frac >= 0.5 else
                      " (prodrop 미성립 → INDEXICAL_CONTEXTUAL 성격)"))
    elif idx_frac is not None and idx_frac < (1 - spec["replication_min"]) and (pd_frac or 0) < 0.5:
        verdict = "SURFACE_ONLY — 마스킹 시 판별 붕괴 + prodrop 실패: 나/남 축은 대명사 토큰 통계"
    else:
        verdict = f"혼합/판정불가 — 모델별 상이(masked 재현 {idx_frac}, prodrop {pd_frac})"

    print("\n" + "=" * 66)
    print(f"[V22 판정] {verdict}")
    print(f"  masked(orig·para) 모델 재현율 {idx_frac} / prodrop 모델 재현율 {pd_frac} "
          f"/ L0 sanity {'OK' if sanity_all else 'FAIL'}")
    print("=" * 66)

    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    payload = {"mode": args.mode, "mock": False, "seeds": seeds, "null_R": R,
               "grid": grid, "indexical_model_frac": idx_frac,
               "prodrop_model_frac": pd_frac, "mask_sanity_all": sanity_all,
               "verdict": verdict}
    p = os.path.join(outdir, f"surface_probe_report_{ts}.json")
    json.dump(payload, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(payload, open(os.path.join(outdir, "surface_probe_report_latest.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"상세: {p} (+ _latest.json)")
    if args.mode == "pilot":
        print("※ 파일럿은 승격 없음.")


if __name__ == "__main__":
    main()
