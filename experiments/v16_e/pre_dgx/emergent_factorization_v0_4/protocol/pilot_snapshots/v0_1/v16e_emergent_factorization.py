"""V16-E pilot: task-induced emergent factorization in a unified Transformer.

The primary models have one shared Transformer representation and no
factor-specific slots, gains, or policy routes. The behavior-only condition is
trained only on action and observer-neutral world objectives. Identity target,
beneficiary target, and common protective responsiveness are evaluated only
post hoc with held-out probes and linear-subspace interventions.

Scope: synthetic task-induced latent organization, not spontaneous selfhood,
phenomenology, or a human-awareness model.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass


@dataclass
class Config:
    n_bits: int = 3
    n_symbols: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 3
    batch_size: int = 192
    train_steps: int = 650
    lr: float = 2e-3
    weight_decay: float = 1e-4
    self_weight: float = 0.90
    beneficiary_weight: float = 1.35
    cost_one: float = 0.52
    cost_both: float = 0.95
    action_logit_scale: float = 5.0
    world_weight: float = 0.45
    neutral_weight: float = 0.20
    factor_aux_weight: float = 0.30
    train_n_probe: int = 5000
    test_n_probe: int = 4000
    eval_n: int = 5000
    device: str = "cpu"
    torch_threads: int = 4

    @property
    def seq_len(self) -> int:
        # CLS, entity A, entity B, identity-role, beneficiary-role, concern-context
        return 6


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bits_to_risk(x: torch.Tensor) -> torch.Tensor:
    p = (2 ** torch.arange(x.shape[-1], device=x.device)).float()
    return (x * p).sum(-1) / float(2 ** x.shape[-1] - 1)


def binary_bacc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y.astype(int), pred.astype(int)))


def bit_bacc(y: np.ndarray, logits: np.ndarray) -> float:
    return binary_bacc(y.reshape(-1), (logits.reshape(-1) > 0).astype(int))


def true_action(
    risk_a: torch.Tensor,
    risk_b: torch.Tensor,
    identity: torch.Tensor,
    beneficiary: torch.Tensor,
    concern: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    wa = concern.float() + cfg.self_weight * (identity == 0).float() + cfg.beneficiary_weight * (beneficiary == 0).float()
    wb = concern.float() + cfg.self_weight * (identity == 1).float() + cfg.beneficiary_weight * (beneficiary == 1).float()
    losses = torch.stack(
        [
            wa * risk_a + wb * risk_b,
            torch.full_like(risk_a, cfg.cost_one) + wb * risk_b,
            torch.full_like(risk_a, cfg.cost_one) + wa * risk_a,
            torch.full_like(risk_a, cfg.cost_both),
        ],
        dim=-1,
    )
    return losses.argmin(-1)


def make_batch(
    n: int,
    cfg: Config,
    gen: torch.Generator,
    *,
    base: Optional[Dict[str, torch.Tensor]] = None,
    flip_factor: Optional[str] = None,
    lexical_permute: bool = False,
) -> Dict[str, torch.Tensor]:
    """Generate paired two-entity episodes.

    Role cues carry a symbol that must be relationally matched to one of two
    entity tokens. Entity order and symbol assignment are independently
    randomized. No factor-specific route is created in the model.
    """
    B = cfg.n_bits
    if base is None:
        risk_bits = torch.randint(0, 2, (n, 2, B), generator=gen).float()
        neutral_bits = torch.randint(0, 2, (n, 2, B), generator=gen).float()
        identity = torch.randint(0, 2, (n,), generator=gen)
        beneficiary = torch.randint(0, 2, (n,), generator=gen)
        concern = torch.randint(0, 2, (n,), generator=gen)
        # Two distinct symbols per episode.
        sym0 = torch.randint(0, cfg.n_symbols, (n,), generator=gen)
        offset = torch.randint(1, cfg.n_symbols, (n,), generator=gen)
        sym1 = (sym0 + offset) % cfg.n_symbols
        symbols = torch.stack([sym0, sym1], dim=1)
        order = torch.randint(0, 2, (n,), generator=gen)
    else:
        risk_bits = base["risk_bits"].clone()
        neutral_bits = base["neutral_bits"].clone()
        identity = base["identity"].clone()
        beneficiary = base["beneficiary"].clone()
        concern = base["concern"].clone()
        symbols = base["symbols"].clone()
        order = base["order"].clone()

    if flip_factor == "identity":
        identity = 1 - identity
    elif flip_factor == "beneficiary":
        beneficiary = 1 - beneficiary
    elif flip_factor == "concern":
        concern = 1 - concern
    elif flip_factor is not None:
        raise ValueError(f"unknown flip_factor={flip_factor}")

    if lexical_permute:
        # Deterministic global symbol relabeling; semantics must be recovered by
        # matching role and entity tokens rather than absolute symbol identity.
        symbols = (symbols * 3 + 1) % cfg.n_symbols

    row = torch.arange(n)
    # entity_order_pos gives which canonical entity is at token position 1/2.
    entity_order = torch.stack([order, 1 - order], dim=1)
    gather3 = entity_order[:, :, None].expand(-1, -1, B)
    risk_ordered = risk_bits.gather(1, gather3)
    neutral_ordered = neutral_bits.gather(1, gather3)
    symbol_ordered = symbols.gather(1, entity_order)

    id_symbol = symbols[row, identity]
    ben_symbol = symbols[row, beneficiary]
    risk = bits_to_risk(risk_bits)
    action = true_action(risk[:, 0], risk[:, 1], identity, beneficiary, concern, cfg)
    neutral_xor = (neutral_bits[:, 0].long() ^ neutral_bits[:, 1].long()).float()

    return {
        "risk_bits": risk_bits,
        "neutral_bits": neutral_bits,
        "risk": risk,
        "identity": identity,
        "beneficiary": beneficiary,
        "concern": concern,
        "symbols": symbols,
        "order": order,
        "entity_order": entity_order,
        "risk_ordered": risk_ordered,
        "neutral_ordered": neutral_ordered,
        "symbol_ordered": symbol_ordered,
        "id_symbol": id_symbol,
        "ben_symbol": ben_symbol,
        "action": action,
        "neutral_xor": neutral_xor,
    }


class UnifiedTransformer(nn.Module):
    """Single shared Transformer; no factor-specific internal pathway."""

    def __init__(self, cfg: Config, condition: str):
        super().__init__()
        self.cfg = cfg
        self.condition = condition
        d, B = cfg.d_model, cfg.n_bits
        self.type_emb = nn.Embedding(6, d)
        self.symbol_emb = nn.Embedding(cfg.n_symbols, d)
        self.risk_proj = nn.Linear(B, d)
        self.neutral_proj = nn.Linear(B, d)
        self.concern_proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, d) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.ff_mult * d,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d)
        self.policy = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.world = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * B))
        self.neutral = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, B))
        # These heads are trained only in the multitask condition. They do not
        # create separate routes; all read the same shared CLS representation.
        self.factor_heads = nn.ModuleDict(
            {
                "identity": nn.Linear(d, 2),
                "beneficiary": nn.Linear(d, 2),
                "concern": nn.Linear(d, 2),
            }
        )

    def embed(self, b: Dict[str, torch.Tensor]) -> torch.Tensor:
        n = b["risk_bits"].shape[0]
        device = b["risk_bits"].device
        x = torch.zeros(n, self.cfg.seq_len, self.cfg.d_model, device=device)
        x[:, 0] = self.cls
        # Entity tokens, positions 1 and 2.
        for j in range(2):
            x[:, 1 + j] = (
                self.type_emb(torch.full((n,), 1 + j, device=device, dtype=torch.long))
                + self.symbol_emb(b["symbol_ordered"][:, j])
                + self.risk_proj(b["risk_ordered"][:, j])
                + self.neutral_proj(b["neutral_ordered"][:, j])
            )
        # Identity and beneficiary role tokens.
        x[:, 3] = self.type_emb(torch.full((n,), 3, device=device, dtype=torch.long)) + self.symbol_emb(b["id_symbol"])
        x[:, 4] = self.type_emb(torch.full((n,), 4, device=device, dtype=torch.long)) + self.symbol_emb(b["ben_symbol"])
        x[:, 5] = self.type_emb(torch.full((n,), 5, device=device, dtype=torch.long)) + self.concern_proj(b["concern"].float()[:, None])
        return x + self.pos[None]

    def encode(
        self,
        b: Dict[str, torch.Tensor],
        *,
        intervene_layer: Optional[int] = None,
        cls_override: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.embed(b)
        states = [h]
        if intervene_layer == 0 and cls_override is not None:
            h = h.clone(); h[:, 0] = cls_override
        for li, layer in enumerate(self.layers, start=1):
            h = layer(h)
            if intervene_layer == li and cls_override is not None:
                h = h.clone(); h[:, 0] = cls_override
            states.append(h)
        h = self.final_ln(h)
        if intervene_layer == self.cfg.n_layers + 1 and cls_override is not None:
            h = h.clone(); h[:, 0] = cls_override
        states.append(h)
        return h, states if return_layers else []

    def heads_from_cls(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {
            "policy": self.cfg.action_logit_scale * self.policy(z),
            "world": self.world(z).view(-1, 2, self.cfg.n_bits),
            "neutral": self.neutral(z),
        }
        out.update({k: head(z) for k, head in self.factor_heads.items()})
        return out

    def forward(self, b: Dict[str, torch.Tensor], return_layers: bool = False) -> Dict[str, torch.Tensor]:
        h, states = self.encode(b, return_layers=return_layers)
        out = self.heads_from_cls(h[:, 0])
        if return_layers:
            out["states"] = states
        return out

    def forward_from_layer_cls(self, b: Dict[str, torch.Tensor], layer_idx: int, new_cls: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Recompute deterministically and insert the modified CLS at layer_idx.
        h, _ = self.encode(b, intervene_layer=layer_idx, cls_override=new_cls, return_layers=False)
        return self.heads_from_cls(h[:, 0])


def train_one(cfg: Config, seed: int, condition: str, outdir: Path) -> UnifiedTransformer:
    seed_all(seed)
    torch.set_num_threads(cfg.torch_threads)
    model = UnifiedTransformer(cfg, condition).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(100000 + seed)
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for step in range(cfg.train_steps):
        b = {k: v.to(cfg.device) for k, v in make_batch(cfg.batch_size, cfg, gen).items()}
        o = model(b)
        loss_action = F.cross_entropy(o["policy"], b["action"])
        loss_world = F.binary_cross_entropy_with_logits(o["world"], b["risk_bits"])
        loss_neutral = F.binary_cross_entropy_with_logits(o["neutral"], b["neutral_xor"])
        loss = loss_action + cfg.world_weight * loss_world + cfg.neutral_weight * loss_neutral
        loss_factor = torch.tensor(0.0, device=cfg.device)
        if condition == "unified_multitask":
            loss_factor = (
                F.cross_entropy(o["identity"], b["identity"])
                + F.cross_entropy(o["beneficiary"], b["beneficiary"])
                + F.cross_entropy(o["concern"], b["concern"])
            ) / 3.0
            loss = loss + cfg.factor_aux_weight * loss_factor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if step % 25 == 0 or step == cfg.train_steps - 1:
            with torch.no_grad():
                history.append(
                    {
                        "step": step,
                        "loss": float(loss),
                        "loss_action": float(loss_action),
                        "loss_world": float(loss_world),
                        "loss_neutral": float(loss_neutral),
                        "loss_factor": float(loss_factor),
                        "train_action_acc": float((o["policy"].argmax(-1) == b["action"]).float().mean()),
                    }
                )
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(outdir / "train_history.csv", index=False)
    torch.save(model.state_dict(), outdir / "checkpoint.pt")
    metadata = {
        "seed": seed,
        "condition": condition,
        "started_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.time() - t0,
        "config": asdict(cfg),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model


def collect_layers(model: UnifiedTransformer, b: Dict[str, torch.Tensor]) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    with torch.no_grad():
        o = model(b, return_layers=True)
    layers = [s[:, 0].detach().cpu().numpy() for s in o["states"]]
    outputs = {k: v.detach().cpu().numpy() for k, v in o.items() if k != "states"}
    return layers, outputs


class Probe:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Probe":
        xs = self.scaler.fit_transform(x)
        self.clf.fit(xs, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(x))

    def direction_raw(self) -> np.ndarray:
        # Logistic weight in raw activation coordinates.
        return self.clf.coef_[0] / np.maximum(self.scaler.scale_, 1e-8)


def probe_metrics(
    train_layers: List[np.ndarray],
    test_layers: List[np.ndarray],
    train_b: Dict[str, torch.Tensor],
    test_b: Dict[str, torch.Tensor],
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], Probe]]:
    rows = []
    probes: Dict[Tuple[int, str], Probe] = {}
    labels = ["identity", "beneficiary", "concern"]
    for li, (xtr, xte) in enumerate(zip(train_layers, test_layers)):
        for factor in labels:
            ytr = train_b[factor].cpu().numpy()
            yte = test_b[factor].cpu().numpy()
            p = Probe().fit(xtr, ytr)
            pred = p.predict(xte)
            # Cross-context split: train a separate probe in one context and
            # test in the complementary context.
            if factor == "identity":
                ctx_tr = train_b["beneficiary"].cpu().numpy() == 0
                ctx_te = test_b["beneficiary"].cpu().numpy() == 1
            elif factor == "beneficiary":
                ctx_tr = train_b["identity"].cpu().numpy() == 0
                ctx_te = test_b["identity"].cpu().numpy() == 1
            else:
                ctx_tr = (train_b["identity"] == train_b["beneficiary"]).cpu().numpy()
                ctx_te = (test_b["identity"] != test_b["beneficiary"]).cpu().numpy()
            pc = Probe().fit(xtr[ctx_tr], ytr[ctx_tr])
            pred_c = pc.predict(xte[ctx_te])
            rows.append(
                {
                    "layer": li,
                    "factor": factor,
                    "probe_bacc": binary_bacc(yte, pred),
                    "cross_context_bacc": binary_bacc(yte[ctx_te], pred_c),
                }
            )
            probes[(li, factor)] = p
    return pd.DataFrame(rows), probes


def evaluate_base(model: UnifiedTransformer, cfg: Config, seed: int, lexical_permute: bool = False) -> Dict[str, float]:
    b = make_batch(cfg.eval_n, cfg, torch.Generator().manual_seed(300000 + seed), lexical_permute=lexical_permute)
    with torch.no_grad():
        o = model(b)
    r = {
        "action_acc": float((o["policy"].argmax(-1) == b["action"]).float().mean()),
        "world_bacc": bit_bacc(b["risk_bits"].numpy(), o["world"].numpy()),
        "neutral_bacc": bit_bacc(b["neutral_xor"].numpy(), o["neutral"].numpy()),
    }
    for factor in ["identity", "beneficiary", "concern"]:
        r[f"trained_head_{factor}_acc"] = float((o[factor].argmax(-1) == b[factor]).float().mean())
    return r


def linear_swap_interventions(
    model: UnifiedTransformer,
    cfg: Config,
    seed: int,
    layer_idx: int,
    probes: Dict[Tuple[int, str], Probe],
) -> pd.DataFrame:
    n = cfg.eval_n
    gen = torch.Generator().manual_seed(400000 + seed)
    base = make_batch(n, cfg, gen)
    base_layers, base_o = collect_layers(model, base)
    z = base_layers[layer_idx]
    rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(500000 + seed)
    for factor in ["identity", "beneficiary", "concern"]:
        cf = make_batch(n, cfg, torch.Generator().manual_seed(1), base=base, flip_factor=factor)
        cf_layers, cf_o = collect_layers(model, cf)
        zcf = cf_layers[layer_idx]
        probe = probes[(layer_idx, factor)]
        w = probe.direction_raw().astype(np.float64)
        w = w / (np.linalg.norm(w) + 1e-12)
        proj = z @ w
        proj_cf = zcf @ w
        zswap = z + (proj_cf - proj)[:, None] * w[None, :]
        # Norm-matched random direction control.
        rdir = rng.normal(size=w.shape)
        rdir -= w * float(rdir @ w)
        rdir = rdir / (np.linalg.norm(rdir) + 1e-12)
        zrand = z + (proj_cf - proj)[:, None] * rdir[None, :]
        with torch.no_grad():
            oswap = model.forward_from_layer_cls(base, layer_idx, torch.tensor(zswap, dtype=torch.float32))
            orand = model.forward_from_layer_cls(base, layer_idx, torch.tensor(zrand, dtype=torch.float32))
        pred0 = base_o["policy"].argmax(-1)
        predcf = cf_o["policy"].argmax(-1)
        predsw = oswap["policy"].numpy().argmax(-1)
        predr = orand["policy"].numpy().argmax(-1)
        conflict = pred0 != predcf
        if conflict.sum() == 0:
            follow = follow_rand = float("nan")
        else:
            follow = float((predsw[conflict] == predcf[conflict]).mean())
            follow_rand = float((predr[conflict] == predcf[conflict]).mean())
        # Probe flip and off-target stability at the intervened layer.
        target_probe_before = probe.predict(z)
        target_probe_after = probe.predict(zswap)
        desired = cf[factor].numpy()
        target_transfer = float((target_probe_after == desired).mean())
        off_stab = []
        for other in ["identity", "beneficiary", "concern"]:
            if other == factor:
                continue
            po = probes[(layer_idx, other)]
            off_stab.append(float((po.predict(zswap) == po.predict(z)).mean()))
        world0 = (base_o["world"] > 0).astype(int)
        worldsw = (oswap["world"].numpy() > 0).astype(int)
        rows.append(
            {
                "seed": seed,
                "condition": model.condition,
                "layer": layer_idx,
                "factor": factor,
                "conflict_n": int(conflict.sum()),
                "counterfactual_follow": follow,
                "random_follow": follow_rand,
                "target_probe_transfer": target_transfer,
                "offtarget_probe_stability": float(np.mean(off_stab)),
                "world_prediction_stability": float((world0 == worldsw).mean()),
                "policy_change_rate": float((predsw != pred0).mean()),
                "cf_policy_change_rate": float(conflict.mean()),
            }
        )
    return pd.DataFrame(rows)


def run_seed(root: Path, cfg: Config, seed: int, condition: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dir = root / "raw" / "pilot" / f"{condition}_seed{seed}"
    model = train_one(cfg, seed, condition, run_dir)
    base = evaluate_base(model, cfg, seed, lexical_permute=False)
    lex = evaluate_base(model, cfg, seed + 77, lexical_permute=True)
    base_row = {"seed": seed, "condition": condition, **base, **{f"lexical_{k}": v for k, v in lex.items()}}
    pd.DataFrame([base_row]).to_csv(run_dir / "base_metrics.csv", index=False)

    train_b = make_batch(cfg.train_n_probe, cfg, torch.Generator().manual_seed(600000 + seed))
    test_b = make_batch(cfg.test_n_probe, cfg, torch.Generator().manual_seed(700000 + seed))
    train_layers, _ = collect_layers(model, train_b)
    test_layers, _ = collect_layers(model, test_b)
    probes_df, probes = probe_metrics(train_layers, test_layers, train_b, test_b)
    probes_df.insert(0, "condition", condition)
    probes_df.insert(0, "seed", seed)
    probes_df.to_csv(run_dir / "probe_metrics.csv", index=False)

    # Pilot chooses the layer with the highest mean cross-context accuracy,
    # excluding embedding layer 0; confirmatory layer will be frozen later.
    layer_scores = probes_df[probes_df.layer > 0].groupby("layer")["cross_context_bacc"].mean()
    best_layer = int(layer_scores.idxmax())
    interventions = linear_swap_interventions(model, cfg, seed, best_layer, probes)
    interventions.to_csv(run_dir / "interventions.csv", index=False)
    (run_dir / "selected_layer.json").write_text(json.dumps({"best_layer": best_layer, "criterion": "mean cross_context_bacc"}, indent=2), encoding="utf-8")
    return pd.DataFrame([base_row]), probes_df, interventions


def write_registry(root: Path, rows: List[Dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(root / "run_registry.csv", index=False)


def parse_seeds(s: str) -> List[int]:
    if "-" in s and "," not in s:
        a, b = map(int, s.split("-")); return list(range(a, b + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", default="16120-16122")
    ap.add_argument("--conditions", default="unified_behavior_only,unified_multitask")
    ap.add_argument("--steps", type=int, default=650)
    ap.add_argument("--eval-n", type=int, default=5000)
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = Config(train_steps=args.steps, eval_n=args.eval_n)
    seeds = parse_seeds(args.seeds)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    all_base, all_probe, all_int = [], [], []
    registry = []
    source_path = Path(__file__).resolve()
    source_hash = sha256(source_path)
    for condition in conditions:
        if condition not in {"unified_behavior_only", "unified_multitask"}:
            raise ValueError(condition)
        for seed in seeds:
            started = datetime.now(timezone.utc).isoformat()
            status = "completed"
            error = ""
            t0 = time.time()
            try:
                b, p, i = run_seed(root, cfg, seed, condition)
                all_base.append(b); all_probe.append(p); all_int.append(i)
            except Exception as exc:
                status = "failed"; error = repr(exc)
                raise
            finally:
                registry.append(
                    {
                        "run_id": f"{condition}_seed{seed}",
                        "condition": condition,
                        "seed": seed,
                        "started_utc": started,
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_s": time.time() - t0,
                        "source_sha256": source_hash,
                        "status": status,
                        "error": error,
                    }
                )
                write_registry(root, registry)
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    base_df = pd.concat(all_base, ignore_index=True)
    probe_df = pd.concat(all_probe, ignore_index=True)
    int_df = pd.concat(all_int, ignore_index=True)
    base_df.to_csv(analysis / "pilot_base_metrics.csv", index=False)
    probe_df.to_csv(analysis / "pilot_probe_metrics.csv", index=False)
    int_df.to_csv(analysis / "pilot_interventions.csv", index=False)
    base_df.groupby("condition").mean(numeric_only=True).to_csv(analysis / "pilot_base_summary.csv")
    probe_df.groupby(["condition", "layer", "factor"]).mean(numeric_only=True).to_csv(analysis / "pilot_probe_summary.csv")
    int_df.groupby(["condition", "factor"]).mean(numeric_only=True).to_csv(analysis / "pilot_intervention_summary.csv")
    (root / "configs" / "pilot_v0_1_resolved.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "root": str(root), "runs": len(registry), "source_sha256": source_hash}, indent=2))


if __name__ == "__main__":
    main()
