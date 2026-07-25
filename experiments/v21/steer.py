"""V21: 나/남 축 개입(steering) — 읽히는 축을 밀면 출력이 선택적으로 바뀌나 (E3→E4).

V17 steering 실패는 '요인 사이 선택성'이었고 V18/V19가 그 원인(요인 병합, H-A)을 밝혔다.
V21은 대상을 바꾼다: 가장 튼튼한 **나/남 축**(self_acc 6모델×3언어 통과)을 밀어, 다음 토큰이
'나(me/I/나)' 쪽으로 쏠리나 본다. 무작위/직교 방향은 안 쏠려야(선택성 = E4 핵심).

판정(결과 전 봉인):
  HANDLE     : +w는 self-margin↑, -w는 ↓, 그 효과가 무작위방향 null 95pct 초과(부호 일치)
  NO-HANDLE  : 효과가 null 안 → 읽히지만 인과 손잡이 아님(읽기/쓰기 해리, H-B)
  NON-SELECT : 무작위 방향도 비슷하게 움직임 → 이 축 특유 효과 아님
  + 중립과제 손상 ~0 확인(개입이 모델을 부수지 않음).

mock(ToyLM): 손잡이를 심어두므로 HANDLE이 나와야 정상(판정 파이프라인 검증).
--check-hook: 실모델에서 hook이 실제로 logit을 바꾸는지만 확인(첫 스모크 자가검증).
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
V17 = os.path.join(HERE, "..", "v17")



def save_report(stem, payload):
    outdir = os.path.join(HERE, "results"); os.makedirs(outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outdir, f"{stem}_{ts}.json")
    json.dump(payload, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(payload, open(os.path.join(outdir, f"{stem}_latest.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    return path

# ----------------------------------------------------- 층 탐색(구조 무관) + hook
def find_layers(model):
    for path in ["model.layers", "gpt_neox.layers", "transformer.h", "model.decoder.layers"]:
        obj = model; ok = True
        for a in path.split("."):
            if hasattr(obj, a): obj = getattr(obj, a)
            else: ok = False; break
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    best = None
    for mod in model.modules():
        if isinstance(mod, nn.ModuleList) and (best is None or len(mod) > len(best)):
            best = mod
    return best


def _add_hook(layer_module, vec):
    """layer 출력(잔차)에 vec를 더하는 forward hook. 출력이 tuple/tensor 둘 다 처리.
    vec를 활성치와 같은 device/dtype로 맞춰 GPU/CPU 혼용 에러 방지."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = vec.to(device=h.device, dtype=h.dtype)
        if isinstance(out, tuple):
            return (h + v,) + tuple(out[1:])
        return h + v
    return layer_module.register_forward_hook(hook)


# ----------------------------------------------------- 실모델 backend
class SteerHF:
    def __init__(self, name, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # 캐시 우선(오프라인) — HF 네트워크 왕복/rate-limit 멈춤 방지. 캐시에 없으면 네트워크.
        try:
            self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32, local_files_only=True).eval()
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.layers = find_layers(self.model)
        self.n_layers = len(self.layers)
        self.d = self.model.config.hidden_size

    @torch.no_grad()
    def hidden(self, texts, layer):
        outs = []
        for t in texts:
            ids = self.tok(t, return_tensors="pt").input_ids.to(self.device)
            hs = self.model(ids, output_hidden_states=True).hidden_states[layer][0]
            outs.append(hs.mean(0).float().cpu().numpy())   # 문장 평균 풀링
        return np.stack(outs)

    @torch.no_grad()
    def margin(self, ctx, opt_self, opt_other, steer_vec, layer):
        h = _add_hook(self.layers[layer], steer_vec) if steer_vec is not None else None
        m = self._logp(ctx, opt_self) - self._logp(ctx, opt_other)
        if h: h.remove()
        return float(m)

    def _logp(self, ctx, opt):
        cids = self.tok(ctx, return_tensors="pt").input_ids.to(self.device)
        oids = self.tok(opt, add_special_tokens=False).input_ids
        full = torch.cat([cids, torch.tensor([oids]).to(self.device)], 1)
        lg = self.model(full).logits[0]
        lp = 0.0
        for i, t in enumerate(oids):
            pos = cids.shape[1] + i - 1
            lp += float(Fn.log_softmax(lg[pos], -1)[t])
        return lp


# ----------------------------------------------------- mock backend (ToyLM)
class SteerToy:
    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.d, self.V, self.n_layers = 48, 16, 6
        self.emb = nn.Embedding(self.V, self.d)
        self.layers = nn.ModuleList([_ToyLayer(self.d) for _ in range(self.n_layers)])
        self.head = nn.Linear(self.d, self.V, bias=False)
        self._w = torch.randn(self.d); self._w = self._w / self._w.norm()
        with torch.no_grad():   # self=토큰0, other=토큰1을 w축으로 읽게 심음(손잡이 존재)
            self.head.weight[0] = self._w * 2.5
            self.head.weight[1] = -self._w * 2.5
        self.device = "cpu"

    def planted_w(self): return self._w.numpy()

    def _fwd(self, ids, steer_vec, layer):
        x = self.emb(ids); h = None
        if steer_vec is not None:
            h = self.layers[layer].register_forward_hook(
                lambda m, i, o: (o[0] + torch.tensor(steer_vec, dtype=torch.float32),))
        for L in self.layers: x = L(x)[0]
        if h: h.remove()
        return self.head(x)

    @torch.no_grad()
    def hidden(self, texts, layer):   # mock: 텍스트→임의 id 시퀀스
        rng = np.random.default_rng(hash(tuple(texts)) % 9973)
        return np.stack([rng.standard_normal(self.d) for _ in texts])

    @torch.no_grad()
    def margin(self, ctx, opt_self, opt_other, steer_vec, layer):
        ids = torch.tensor([[2, 3, 4]])
        lg = self._fwd(ids, steer_vec, layer)[0, -1]
        return float(lg[0] - lg[1])


class _ToyLayer(nn.Module):
    def __init__(s, d): super().__init__(); s.lin = nn.Linear(d, d)
    def forward(s, x): return (x + torch.tanh(s.lin(x)),)


# ----------------------------------------------------- self/other 축 w + 프로브
def build_self_axis(be, layer, seed):
    """나/남 축 = 요인 통합 (self극 평균 − other극 평균). 단위 정규화."""
    sys.path.insert(0, V17)
    from prompts_bank import build_bank
    from run_seed import FACTORS
    bank = build_bank(seed); S, O = [], []
    for f in FACTORS:
        pairs = bank[f]["A_train"] + bank[f]["B_train"]
        S.append(be.hidden([p for p, _ in pairs], layer))
        O.append(be.hidden([n for _, n in pairs], layer))
    w = np.vstack(S).mean(0) - np.vstack(O).mean(0)
    return w / (np.linalg.norm(w) + 1e-8)


def probes():
    """self/other 2AFC 문맥 (v17 BANK test 재사용) + 중립(손상 확인)."""
    sys.path.insert(0, V17)
    from prompts_bank import BANK, NEUTRAL
    P = []
    for f in BANK:
        for ctx, a, b in BANK[f]["test"]:
            P.append((ctx, a, b))
    return P, [(c, a, b) for c, a, b in NEUTRAL]


# ----------------------------------------------------- 측정·판정
def steer_effect(be, w, layer, alpha, probes_list):
    vec = torch.tensor((alpha * w), dtype=torch.float32)
    base = np.mean([be.margin(c, a, b, None, layer) for c, a, b in probes_list])
    plus = np.mean([be.margin(c, a, b, vec, layer) for c, a, b in probes_list])
    minus = np.mean([be.margin(c, a, b, -vec, layer) for c, a, b in probes_list])
    return base, plus - base, minus - base   # baseline, +효과, -효과


def run(be, seed, layer, alpha, R, seal, w):
    P, NEU = probes()
    base, eff_plus, eff_minus = steer_effect(be, w, layer, alpha, P)
    D = 0.5 * (eff_plus - eff_minus)   # 반대칭 방향효과: 방향무관 오염(+w,-w 공통) 상쇄
    iso = 0.5 * (eff_plus + eff_minus) # 등방 오염 성분(보고용)
    # null: 무작위 방향 R개, 각 r에 대해 ±r 쌍으로 D_r 계산(같은 상쇄 적용)
    rng = np.random.default_rng(seed + 999)
    null, iso_null = [], []
    print(f"    [seed {seed}] null ±쌍 {R}개 계산 중 …", flush=True)
    for _i in range(R):
        if _i and _i % max(1, R // 5) == 0:
            print(f"      null {_i}/{R}", flush=True)
        r = rng.standard_normal(be.d); r = r / np.linalg.norm(r)
        vec = torch.tensor((alpha * r), dtype=torch.float32)
        ep = np.mean([be.margin(c, a, b, vec, layer) for c, a, b in P]) - base
        em = np.mean([be.margin(c, a, b, -vec, layer) for c, a, b in P]) - base
        null.append(0.5 * (ep - em)); iso_null.append(0.5 * (ep + em))
    print(f"      null 완료", flush=True)
    null = np.array(null); iso_null = np.array(iso_null)
    gate_hi = float(np.quantile(null, seal["null_pct"]))
    gate_lo = float(np.quantile(null, 1 - seal["null_pct"]))
    # 중립 손상: +w에서 중립과제 margin 변화 절대값
    vec = torch.tensor((alpha * w), dtype=torch.float32)
    neu_base = np.mean([be.margin(c, a, b, None, layer) for c, a, b in NEU])
    neu_plus = np.mean([be.margin(c, a, b, vec, layer) for c, a, b in NEU])
    damage = abs(neu_plus - neu_base)

    # 판정은 반대칭 D 기준(사전등록 가설: +w→self↑ 즉 D>0). D-null 자체가 쏠리면 오염.
    contaminated = abs(float(null.mean())) > 0.5 * float(null.std() + 1e-9)
    if contaminated:
        verdict = "CONTAMINATED(판정무효)"
    elif D > gate_hi:
        verdict = "HANDLE"
    elif D < gate_lo:
        verdict = "HANDLE-REVERSED(탐색적: 방향 반대 인과)"
    else:
        verdict = "NO-HANDLE"
    return {"seed": seed, "layer": layer, "alpha": alpha,
            "baseline_margin": round(base, 4),
            "D_directional": round(float(D), 4), "iso_bias": round(float(iso), 4),
            "iso_null_mean": round(float(iso_null.mean()), 4),
            "eff_plus": round(float(eff_plus), 4), "eff_minus": round(float(eff_minus), 4),
            "null_gate_hi": round(gate_hi, 4), "null_gate_lo": round(gate_lo, 4),
            "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
            "neutral_damage": round(float(damage), 4), "verdict": verdict}


def tag_or(args):
    return "mock" if args.mock else args.model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--model", default=None); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--device", default="auto"); ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--null-R", type=int, default=None)
    ap.add_argument("--check-hook", action="store_true", help="hook이 logit을 바꾸는지만 확인")
    ap.add_argument("--sweep", action="store_true",
                    help="α 스윕(seed 0, 4점): 깨끗한 측정 구간 탐색 → 본 파일럿 α 선택용")
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    args = ap.parse_args()
    import yaml
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    seal = spec["steer"]
    R = args.null_R or seal["null_R"]
    alpha = args.alpha if args.alpha is not None else seal["alpha_rel"]
    if args.mock and args.mode == "confirm":
        print("❌ 확증에서 mock 금지."); sys.exit(2)

    be = SteerToy(0) if args.mock else SteerHF(
        args.model, device=(None if args.device == "auto" else args.device))
    layer = args.layer if args.layer is not None else max(1, int(be.n_layers * seal["layer_frac"]))
    # 나/남 축 w + α 규모. mock은 심어둔 손잡이(planted_w)+고정 α, 실모델은 self축+잔차norm×alpha_rel.
    if args.mock:
        w0 = be.planted_w(); a = 5.0; scale = 10.0
    else:
        w0 = build_self_axis(be, layer, 0)
        scale = float(np.linalg.norm(be.hidden(["I made the decision."], layer)[0]))
        a = alpha * scale

    if args.check_hook:
        P, _ = probes()
        base = np.mean([be.margin(c, a2, b, None, layer) for c, a2, b in P[:4]])
        vec = torch.tensor((a * w0), dtype=torch.float32)
        st = np.mean([be.margin(c, a2, b, vec, layer) for c, a2, b in P[:4]])
        print(f"[check-hook] layer={layer} d={be.d} n_layers={be.n_layers}")
        print(f"  baseline margin {base:.3f} → +w {st:.3f}  (차이 {st-base:+.3f})")
        print("  → hook 작동:", "정상(logit 바뀜)" if abs(st - base) > 1e-3 else "❌ 안 바뀜(층/구조 확인 필요)")
        return

    if args.sweep:
        # α 스윕: 목적은 '깨끗한 측정 구간'(null≈0) 탐색. HANDLE 찾기가 아니라 sanity 기준.
        # 1차 실행 교훈: α=0.5×문장norm(~1150)은 비방향 오염. 훨씬 낮은 값들을 훑는다.
        rels = [0.01, 0.03, 0.1, 0.3]
        Rs = min(R, 60)
        print(f"=== α 스윕 (seed 0, R={Rs}) — sanity(null≈0) 우선, 방향성은 참고 ===")
        print(f"  scale(문장norm)={scale:.1f} → α 후보: " +
              ", ".join(f"{r}×={r*scale:.1f}" for r in rels))
        best = None; sweep_rows = []
        for rel in rels:
            row = run(be, 0, layer, rel * scale, Rs, seal, w0)
            row["alpha_rel"] = rel; sweep_rows.append(row)
            ok = "오염" if "CONTAMINATED" in row["verdict"] else "깨끗"
            print(f"  α_rel={rel:<5} α={rel*scale:<8.1f} D {row['D_directional']:+7.3f}"
                  f" | D-null [{row['null_gate_lo']:+.3f},{row['null_gate_hi']:+.3f}]"
                  f" | 등방오염 {row['iso_bias']:+.3f} [{ok}] | {row['verdict']}")
            if ok == "깨끗" and best is None:
                best = rel
        rec = (f'--alpha {best}  (상대값. 절대 α={best*scale:.1f})') if best else '전부 오염 — 측정 설계 재검'
        print(f"\n권장: {rec}")
        print("→ 본 파일럿: bash run_v21_pilot.sh --model ... --alpha <위 상대값> --null-R 100")
        print("   ⚠ --alpha 는 상대값(×문장norm). 절대값을 넣으면 안 됨.")
        p = save_report("steer_sweep", {"mode": "sweep", "model": tag_or(args), "layer": layer,
                                         "scale": scale, "rows": sweep_rows, "recommend": rec})
        print(f"스윕 리포트: {p}")
        return

    seeds = spec["pilot_seeds"] if args.mode == "pilot" else spec["confirmatory_seeds"]
    rows = [run(be, s, layer, a, R, seal,
                w0 if args.mock else build_self_axis(be, layer, s)) for s in seeds]
    handle_frac = float(np.mean([r["verdict"] == "HANDLE" for r in rows]))
    dmg = float(np.mean([r["neutral_damage"] for r in rows]))
    cons = spec["stats"]["consistency_min"]
    verdict = ("HANDLE (E4 지지)" if handle_frac >= cons
               else "NON-SELECTIVE" if any(r["verdict"] == "NON-SELECTIVE" for r in rows)
               else "NO-HANDLE (읽기/쓰기 해리)")

    tag = "mock" if args.mock else args.model
    print(f"\n=== V21 나/남 축 개입 [{args.mode}] {tag} layer={layer} α_rel={alpha} R={R} ===")
    for r in rows:
        print(f"  s{r['seed']}: {r['verdict']}  +효과 {r['eff_plus']} / -효과 {r['eff_minus']} "
              f"| null gate [{r['null_gate_lo']},{r['null_gate_hi']}] | 중립손상 {r['neutral_damage']}")
    print(f"[봉인 판정] {verdict}  (HANDLE seed비율 {handle_frac:.2f} vs {cons}, 평균 중립손상 {dmg:.3f})")
    p = save_report("steer_report", {"mock": args.mock, "mode": args.mode, "model": tag,
               "layer": layer, "alpha_rel": alpha, "null_R": R, "per_seed": rows,
               "handle_frac": handle_frac, "verdict_sealed": verdict})
    print(f"상세: {p} (+ steer_report_latest.json)")
    if args.mode == "pilot":
        print("※ 파일럿은 승격 없음. NO-HANDLE도 정직한 결과(H-B).")


if __name__ == "__main__":
    main()
