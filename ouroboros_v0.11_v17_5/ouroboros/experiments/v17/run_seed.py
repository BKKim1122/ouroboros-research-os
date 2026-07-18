"""V17: 사전학습 LLM에서 자기관련 요인 구조의 자연발생(probing) + 제어가능성(steering).

사용:
  python run_seed.py 0                # config.yaml 설정대로 (mock 또는 실모델)
  python run_seed.py 0 --mock        # 파이프라인 검증용 가짜 백엔드

출력: results/seed_{s}.json — ouroboros.audit 계약 형식
  factors / effects(개입행렬, 효과크기 d 단위) / controls
  + emergence 블록(cross-template probe 정확도, E6 판정 보조지표)

절차:
  1) 템플릿군 A의 극성쌍으로 요인별 steering 벡터(평균차) 추출
  2) 템플릿군 B로 probe 전이 검증 (M0 표면 cue 반박)
  3) 각 요인 벡터로 개입하며 4개 요인의 2AFC 로짓마진 변화 측정 → 4x4 행렬
  4) 대조: random direction / shuffled label / 중립과제 손상
"""
from __future__ import annotations
import argparse, json, os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts_bank import build_bank, NEUTRAL, PERSON_CONTROL

FACTORS = ["identity", "beneficiary", "privilege", "concern"]


# ---------------------------------------------------------------- backends
class MockBackend:
    """구조가 심어진 가짜 모델. 파이프라인·계약 검증 전용 (과학적 의미 없음)."""
    def __init__(self, seed, dim=64, truth=None):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        self.dirs = {f: q[:, i] for i, f in enumerate(FACTORS)}
        self.dim, self.rng = dim, rng
        self.truth = truth or {}  # text -> (factor, pole) 실제 정답

    def acts(self, texts, factor, pole):
        out = []
        for t in texts:
            f, p = self.truth.get(t, (factor, pole))
            base = self.dirs[f] * (2.0 if p > 0 else -2.0)
            out.append(base + self.rng.normal(0, 0.6, self.dim))
        return np.stack(out)

    def acts_multi(self, texts, layers):
        return {L: self.acts(texts, FACTORS[0], +1) for L in layers}

    def margin(self, item, factor):
        return 2.0 + self.rng.normal(0, 0.3)

    def margin_steered(self, item, factor, vec, alpha):
        overlap = abs(float(np.dot(vec, self.dirs[factor])))
        return self.margin(item, factor) - alpha * (overlap ** 2) * 0.35

    def neutral_accuracy(self, vec=None, alpha=0.0):
        return 1.0 - (0.01 if vec is not None else 0.0)


class HFBackend:
    """실모델 백엔드. torch + transformers 필요 (사용자 환경에서 실행)."""
    def __init__(self, model_name, layer_frac=0.6, device=None, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        n = self.model.config.num_hidden_layers
        self.layer = max(1, int(n * layer_frac))
        self._hook, self._vec, self._alpha = None, None, 0.0

    def _layers(self):
        m = self.model
        for attr in ("model", "transformer"):
            if hasattr(m, attr):
                mm = getattr(m, attr)
                for lattr in ("layers", "h"):
                    if hasattr(mm, lattr):
                        return getattr(mm, lattr)
        raise RuntimeError("레이어 접근 실패: 모델 구조 확인 필요")

    def acts(self, texts, factor=None, pole=None):
        outs = []
        with self.torch.no_grad():
            for t in texts:
                ids = self.tok(t, return_tensors="pt").to(self.device)
                h = self.model(**ids, output_hidden_states=True).hidden_states[self.layer]
                outs.append(h[0, -1].float().cpu().numpy())
        return np.stack(outs)

    def acts_multi(self, texts, layers):
        out = {L: [] for L in layers}
        with self.torch.no_grad():
            for t in texts:
                ids = self.tok(t, return_tensors="pt").to(self.device)
                hs = self.model(**ids, output_hidden_states=True).hidden_states
                for L in layers:
                    out[L].append(hs[L][0, -1].float().cpu().numpy())
        return {L: np.stack(v) for L, v in out.items()}

    def resid_norm(self, texts, layer):
        a = self.acts_multi(texts, [layer])[layer]
        return float(np.linalg.norm(a, axis=1).mean())

    def _install(self, vec, alpha):
        v = self.torch.tensor(vec, dtype=self.model.dtype, device=self.device)
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + alpha * v,) + out[1:]
            return out + alpha * v
        self._hook = self._layers()[self.layer].register_forward_hook(hook)

    def _remove(self):
        if self._hook:
            self._hook.remove(); self._hook = None

    def _logprob(self, ctx, option):
        ids_ctx = self.tok(ctx, return_tensors="pt").input_ids
        ids_all = self.tok(ctx + option, return_tensors="pt").input_ids.to(self.device)
        with self.torch.no_grad():
            logits = self.model(ids_all).logits.float()
        lp = self.torch.log_softmax(logits[0, :-1], dim=-1)
        start = ids_ctx.shape[1] - 1
        tgt = ids_all[0, ids_ctx.shape[1]:]
        return float(lp[start:start + len(tgt)].gather(1, tgt.unsqueeze(1)).sum())

    def margin(self, item, factor=None):
        ctx, pos, neg = item
        return self._logprob(ctx, pos) - self._logprob(ctx, neg)

    def margin_steered(self, item, factor, vec, alpha):
        self._install(vec, -alpha)  # 자기극 반대 방향으로 밀기
        try:
            return self.margin(item)
        finally:
            self._remove()

    def neutral_accuracy(self, vec=None, alpha=0.0):
        if vec is not None:
            self._install(vec, -alpha)
        try:
            return float(np.mean([self.margin(it) > 0 for it in NEUTRAL]))
        finally:
            self._remove()


# ---------------------------------------------------------------- analysis
def steering_vec(acts_pos, acts_neg):
    v = acts_pos.mean(0) - acts_neg.mean(0)
    return v / (np.linalg.norm(v) + 1e-8)

def centroid_probe_acc(train_p, train_n, test_p, test_n):
    cp, cn = train_p.mean(0), train_n.mean(0)
    def pred(x):
        return np.linalg.norm(x - cp, axis=1) < np.linalg.norm(x - cn, axis=1)
    return float((pred(test_p).sum() + (~pred(test_n)).sum()) / (len(test_p) + len(test_n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    cfg = {}
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if os.path.exists(cfg_path):
        import yaml
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    mock = args.mock or cfg.get("mock", True)
    alpha = float(cfg.get("alpha", 8.0))
    rng = np.random.default_rng(args.seed)
    bank = build_bank(args.seed)

    truth = {}
    for f in FACTORS:
        for p, n in bank[f]["A_train"] + bank[f]["B_train"]:
            truth[p] = (f, +1); truth[n] = (f, -1)
    be = MockBackend(args.seed, truth=truth) if mock else HFBackend(
        cfg.get("model", "Qwen/Qwen2.5-1.5B"), float(cfg.get("layer_frac", 0.6)),
        revision=cfg.get("revision"))

    # 1) A군으로 steering 벡터, 2) B군으로 probe 전이 (자연발생 검증)
    vecs, emergence, A_acts, B_acts = {}, {}, {}, {}
    for f in FACTORS:
        A = bank[f]["A_train"]; B = bank[f]["B_train"]
        ap_, an_ = be.acts([p for p, _ in A], f, +1), be.acts([n for _, n in A], f, -1)
        bp_, bn_ = be.acts([p for p, _ in B], f, +1), be.acts([n for _, n in B], f, -1)
        A_acts[f], B_acts[f] = (ap_, an_), (bp_, bn_)
        vecs[f] = steering_vec(ap_, an_)
        emergence[f] = {
            "cross_template_probe_acc": centroid_probe_acc(ap_, an_, bp_, bn_),
            "chance": 0.5,
        }

    # 2.5) probe 혼동행렬 (M1 공용축 vs M3 분리축 판별의 핵심)
    probe_confusion = {a: {b: round(centroid_probe_acc(
        A_acts[a][0], A_acts[a][1], B_acts[b][0], B_acts[b][1]), 3)
        for b in FACTORS} for a in FACTORS}

    # 2.7) 확증 지표 (V17 사전 등록 endpoint)
    pp = be.acts([p for p, _ in PERSON_CONTROL], FACTORS[0], +1)
    pn = be.acts([n for _, n in PERSON_CONTROL], FACTORS[0], -1)
    v_shared = np.mean([vecs[f] for f in FACTORS], axis=0)
    v_shared /= (np.linalg.norm(v_shared) + 1e-8)
    def _proj_out(X):
        return X - np.outer(X @ v_shared, v_shared)
    conf_proj = {a: {b: centroid_probe_acc(
        _proj_out(A_acts[a][0]), _proj_out(A_acts[a][1]),
        _proj_out(B_acts[b][0]), _proj_out(B_acts[b][1]))
        for b in FACTORS} for a in FACTORS}
    _diag = float(np.mean([conf_proj[f][f] for f in FACTORS]))
    _off = float(np.mean([conf_proj[a][b] for a in FACTORS for b in FACTORS if a != b]))
    transfer_vals = [emergence[f]["cross_template_probe_acc"] for f in FACTORS]
    confirmatory_metrics = {
        "cross_template_probe_mean": round(float(np.mean(transfer_vals)), 3),
        "cross_template_probe_min_factor": round(float(min(transfer_vals)), 3),
        "privilege_person_probe": round(centroid_probe_acc(
            pp, pn, B_acts["privilege"][0], B_acts["privilege"][1]), 3),
        "projection_gap": round(_diag - _off, 3),
    }

    # 3) 개입행렬: 효과크기 d = |Δmargin 평균| / 기저마진 SD
    base = {f: np.array([be.margin(it, f) for it in bank[f]["test"]]) for f in FACTORS}
    sd = {f: float(base[f].std() + 1e-6) for f in FACTORS}
    effects = {}
    for a in FACTORS:
        effects[a] = {}
        for b in FACTORS:
            steered = np.array([be.margin_steered(it, b, vecs[a], alpha)
                                for it in bank[b]["test"]])
            effects[a][b] = round(abs(float((base[b] - steered).mean())) / sd[b], 4)

    # 4) 대조
    rvec = rng.normal(size=len(vecs[FACTORS[0]])); rvec /= np.linalg.norm(rvec)
    rand_eff = np.mean([abs(float((base[b] - np.array(
        [be.margin_steered(it, b, rvec, alpha) for it in bank[b]["test"]])).mean())) / sd[b]
        for b in FACTORS])
    f0 = FACTORS[0]; A0 = bank[f0]["A_train"]
    mixed = [p for p, _ in A0] + [n for _, n in A0]
    rng.shuffle(mixed); half = len(mixed) // 2
    svec = steering_vec(be.acts(mixed[:half], f0, +1), be.acts(mixed[half:], f0, -1))
    shuf_eff = abs(float((base[f0] - np.array(
        [be.margin_steered(it, f0, svec, alpha) for it in bank[f0]["test"]])).mean())) / sd[f0]
    acc0 = be.neutral_accuracy()
    acc1 = np.mean([be.neutral_accuracy(vecs[f], alpha) for f in FACTORS])

    result = {
        "seed": args.seed,
        "backend": "mock(파이프라인 검증 전용)" if mock else cfg.get("model"),
        "factors": FACTORS,
        "effects": effects,
        "controls": {
            "random_direction": round(float(rand_eff), 4),
            "shuffled_label": round(float(shuf_eff), 4),
            "neutral_task_damage": round(float(acc0 - acc1), 4),
        },
        "emergence": emergence,
        "probe_confusion": probe_confusion,
        "confirmatory_metrics": confirmatory_metrics,
        "config": {"alpha": alpha, "mock": mock,
                   "layer": getattr(be, "layer", None),
                   "revision": getattr(be, "revision", None)},
    }
    os.makedirs("results", exist_ok=True)
    out = f"results/seed_{args.seed}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(out)


if __name__ == "__main__":
    main()
