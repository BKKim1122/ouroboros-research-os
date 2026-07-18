"""V16-E.6 surface/transition robustness pilot.

Key constraints:
- no canonical body embeddings
- no factor-specific latent slots, gains, routes, or factor-report losses
- episode-local random entity codes; relation tokens bind by code equality
- generic relation token type with randomized relation-token order
- randomized phase-query order
- held-out symbol-code distribution and held-out transition-family composition
"""
from __future__ import annotations
import argparse, json, math, random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    n_bits: int = 3
    code_dim: int = 16
    d: int = 72
    heads: int = 4
    layers: int = 2
    batch: int = 256
    steps: int = 950
    lr: float = 1.8e-3
    outcome_weight: float = 0.9
    self_weight: float = 1.1
    ben_weight: float = 1.65
    cost_one: float = 0.62
    cost_both: float = 1.08
    threads: int = 1
    eval_n: int = 1800
    # threshold, effect, and operation-sign combinations
    train_rules: tuple = (
        (0.35, 0.45, 0.40, 0.25, 1.0, 1.0),
        (0.50, 0.55, 0.50, 0.25, -1.0, 1.0),
        (0.65, 0.65, 0.60, 0.30, 1.0, -1.0),
    )
    heldout_rule: tuple = (0.425, 0.60, 0.55, 0.275, -1.0, -1.0)
    heldout_factor_combos: tuple = ((0, 0, 0), (1, 1, 1))
    train_symbol_ids: tuple = (0,1,2,3,4,5,6,7,8,9,10,11)
    ood_symbol_ids: tuple = (12,13,14,15)
    codebook_seed: int = 0

    @property
    def L(self):
        # 2 entity + 3 generic relation + target + rule + 4 phase queries
        return 11


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bits_value(x: torch.Tensor) -> torch.Tensor:
    p = (2 ** torch.arange(x.shape[-1])).float()
    return (x * p).sum(-1) / (2 ** x.shape[-1] - 1)


def canonical_to_slot(entity: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return (order == entity[:, None]).long().argmax(-1)


def canonical_action_to_slot(action: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    out = action.clone()
    a_slot = canonical_to_slot(torch.zeros_like(action), order)
    b_slot = canonical_to_slot(torch.ones_like(action), order)
    out = torch.where(action == 1, a_slot + 1, out)
    out = torch.where(action == 2, b_slot + 1, out)
    return out


def action_for_entity_slot(entity: torch.Tensor, needed: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    slot = canonical_to_slot(entity, order)
    return torch.where(needed, slot + 1, torch.zeros_like(slot))


def apply_action_slot(base_ordered: torch.Tensor, action: torch.Tensor, benefit: torch.Tensor) -> torch.Tensor:
    out = base_ordered.clone()
    out[:, 0] += benefit * ((action == 1) | (action == 3)).float()
    out[:, 1] += benefit * ((action == 2) | (action == 3)).float()
    return out.clamp(0, 1)


def loss_table(ra, rb, wa, wb, cfg):
    return torch.stack([
        wa * ra + wb * rb,
        torch.full_like(ra, cfg.cost_one) + wb * rb,
        torch.full_like(ra, cfg.cost_one) + wa * ra,
        torch.full_like(ra, cfg.cost_both),
    ], -1)


def sample_factors(n, gen, mode, cfg):
    if mode == "factor_ood":
        combos = torch.tensor(cfg.heldout_factor_combos, dtype=torch.long)
        idx = torch.randint(0, len(combos), (n,), generator=gen)
        c = combos[idx]
        return c[:, 0], c[:, 1], c[:, 2]
    if mode in ("train", "id", "encoding_ood", "rule_ood"):
        held = set(tuple(x) for x in cfg.heldout_factor_combos)
        allowed = torch.tensor([
            (i, b, c) for i in (0, 1) for b in (0, 1) for c in (0, 1)
            if (i, b, c) not in held
        ], dtype=torch.long)
        idx = torch.randint(0, len(allowed), (n,), generator=gen)
        vals = allowed[idx]
        return vals[:, 0], vals[:, 1], vals[:, 2]
    ident = torch.randint(0, 2, (n,), generator=gen)
    ben = torch.randint(0, 2, (n,), generator=gen)
    conc = torch.randint(0, 2, (n,), generator=gen)
    return ident, ben, conc


def hadamard(n: int) -> torch.Tensor:
    h=torch.ones(1,1)
    while h.shape[0] < n:
        h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
    return h

def codebook(cfg):
    h=hadamard(cfg.code_dim)
    g=torch.Generator().manual_seed(880000+int(cfg.codebook_seed))
    perm=torch.randperm(cfg.code_dim,generator=g)
    signs=torch.where(torch.randint(0,2,(cfg.code_dim,),generator=g)==0,-1.0,1.0)
    return h[:,perm]*signs[None]

def sample_entity_codes(n, gen, mode, cfg):
    ids=cfg.ood_symbol_ids if mode=="encoding_ood" else cfg.train_symbol_ids
    pool=torch.tensor(ids,dtype=torch.long)
    i0=torch.randint(0,len(pool),(n,),generator=gen)
    i1=torch.randint(0,len(pool)-1,(n,),generator=gen)
    i1=i1+(i1>=i0).long()
    sym=torch.stack([pool[i0],pool[i1]],1)
    cb=codebook(cfg)
    return cb[sym]


def sample_rules(n, gen, mode, cfg):
    if mode == "rule_ood":
        return torch.tensor(cfg.heldout_rule).float()[None].repeat(n, 1)
    rules = torch.tensor(cfg.train_rules).float()
    idx = torch.randint(0, len(rules), (n,), generator=gen)
    return rules[idx]


def make_batch(n, cfg, gen, mode="train", base=None, flip=None):
    B = cfg.n_bits
    if base is None:
        risk_bits = torch.randint(0, 2, (n, 2, B), generator=gen).float()
        energy_bits = torch.randint(0, 2, (n, 2, B), generator=gen).float()
        memory = torch.randint(0, 2, (n, 2), generator=gen).float()
        target = torch.randint(0, 2, (n,), generator=gen).float()
        ident, ben, conc = sample_factors(n, gen, mode, cfg)
        codes = sample_entity_codes(n, gen, mode, cfg)
        order = torch.randint(0, 2, (n,), generator=gen)
        rules = sample_rules(n, gen, mode, cfg)
        # random relation-token order and random phase-query order per episode
        rel_order = torch.argsort(torch.rand(n, 3, generator=gen), dim=1)
        phase_order = torch.argsort(torch.rand(n, 4, generator=gen), dim=1)
    else:
        risk_bits = base["risk_bits"].clone(); energy_bits = base["energy_bits"].clone(); memory = base["memory"].clone(); target = base["target"].clone()
        ident = base["identity"].clone(); ben = base["beneficiary"].clone(); conc = base["concern"].clone(); codes = base["codes"].clone(); order = base["order"].clone(); rules = base["rules"].clone()
        rel_order = base["rel_order"].clone(); phase_order = base["phase_order"].clone()
    if flip == "identity": ident = 1 - ident
    elif flip == "beneficiary": ben = 1 - ben
    elif flip == "concern": conc = 1 - conc

    row = torch.arange(n)
    risk = bits_value(risk_bits); energy = bits_value(energy_bits)
    entity_order = torch.stack([order, 1 - order], 1)
    risk_ord = risk.gather(1, entity_order); energy_ord = energy.gather(1, entity_order); mem_ord = memory.gather(1, entity_order)
    risk_bits_ord = risk_bits.gather(1, entity_order[:, :, None].expand(-1, -1, B)); energy_bits_ord = energy_bits.gather(1, entity_order[:, :, None].expand(-1, -1, B))
    codes_ord = codes.gather(1, entity_order[:, :, None].expand(-1, -1, cfg.code_dim))

    alloc_thresh, protect_thresh, alloc_benefit, protect_factor, alloc_sign, protect_sign = [rules[:, i] for i in range(6)]
    # continuity
    need0 = memory[row, ident] != target
    a0 = action_for_entity_slot(ident, need0, entity_order)
    mem_out = mem_ord.clone()
    mem_out[:, 0] = torch.where((a0 == 1) | (a0 == 3), target, mem_out[:, 0])
    mem_out[:, 1] = torch.where((a0 == 2) | (a0 == 3), target, mem_out[:, 1])
    # allocation transition family: side-of-threshold is encoded by alloc_sign
    e_ben = energy[row, ben]
    need1 = torch.where(alloc_sign > 0, e_ben < alloc_thresh, e_ben > alloc_thresh)
    a1 = action_for_entity_slot(ben, need1, entity_order)
    energy_out = apply_action_slot(energy_ord, a1, alloc_benefit)
    # protection transition family: high/low side encoded by protect_sign
    cond0 = torch.where(protect_sign > 0, risk_ord[:, 0] > protect_thresh, risk_ord[:, 0] < protect_thresh)
    cond1 = torch.where(protect_sign > 0, risk_ord[:, 1] > protect_thresh, risk_ord[:, 1] < protect_thresh)
    a2 = torch.zeros(n, dtype=torch.long)
    a2 = torch.where((conc == 1) & cond0 & ~cond1, torch.ones_like(a2), a2)
    a2 = torch.where((conc == 1) & cond1 & ~cond0, torch.full_like(a2, 2), a2)
    a2 = torch.where((conc == 1) & cond0 & cond1, torch.full_like(a2, 3), a2)
    harm_out = risk_ord.clone()
    harm_out[:, 0] *= torch.where((a2 == 1) | (a2 == 3), protect_factor, torch.ones_like(protect_factor))
    harm_out[:, 1] *= torch.where((a2 == 2) | (a2 == 3), protect_factor, torch.ones_like(protect_factor))
    # integrated V15-like policy
    wa = conc.float() + cfg.self_weight * (ident == 0).float() + cfg.ben_weight * (ben == 0).float()
    wb = conc.float() + cfg.self_weight * (ident == 1).float() + cfg.ben_weight * (ben == 1).float()
    a3c = loss_table(risk[:, 0], risk[:, 1], wa, wb, cfg).argmin(-1)
    a3 = canonical_action_to_slot(a3c, entity_order)
    actions_can = torch.stack([a0, a1, a2, a3], 1)
    outcomes_can = torch.stack([mem_out, energy_out, harm_out, harm_out], 1)
    # query-order targets
    actions_q = actions_can.gather(1, phase_order)
    outcomes_q = outcomes_can.gather(1, phase_order[:, :, None].expand(-1, -1, 2))
    # relation payloads in canonical role order: identity code, beneficiary code, concern signed code
    id_code = codes[row, ident]
    ben_code = codes[row, ben]
    conc_code = torch.zeros(n, cfg.code_dim)
    conc_code[:, 0] = torch.where(conc == 1, 1.0, -1.0)
    rel_payload_can = torch.stack([id_code, ben_code, conc_code], 1)
    rel_role_can = torch.arange(3)[None].repeat(n, 1)
    rel_payload = rel_payload_can.gather(1, rel_order[:, :, None].expand(-1, -1, cfg.code_dim))
    rel_roles = rel_role_can.gather(1, rel_order)
    return dict(
        risk_bits=risk_bits, energy_bits=energy_bits, memory=memory, target=target, identity=ident, beneficiary=ben, concern=conc,
        codes=codes, order=order, entity_order=entity_order, risk_ordered=risk_bits_ord, energy_ordered=energy_bits_ord,
        memory_ordered=mem_ord, codes_ordered=codes_ord, rules=rules, rel_order=rel_order, rel_payload=rel_payload,
        rel_roles=rel_roles, phase_order=phase_order, actions=actions_q, outcomes=outcomes_q,
        actions_canonical=actions_can, outcomes_canonical=outcomes_can,
    )


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; d = cfg.d; B = cfg.n_bits
        self.type_emb = nn.Embedding(6, d)
        self.code = nn.Linear(cfg.code_dim, d, bias=False)
        self.role = nn.Embedding(3, d)
        self.phase = nn.Embedding(4, d)
        self.risk = nn.Linear(B, d); self.energy = nn.Linear(B, d); self.memory = nn.Linear(1, d)
        self.scalar = nn.Linear(1, d); self.rule = nn.Linear(6, d)
        self.pos = nn.Parameter(torch.randn(cfg.L, d) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, cfg.heads, 3*d, dropout=0., activation="gelu", batch_first=True) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(d)
        self.policy = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.outcome = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def embed(self, b):
        n = b["identity"].shape[0]
        x = torch.zeros(n, self.cfg.L, self.cfg.d)
        # entity tokens
        for j in range(2):
            x[:, j] = self.type_emb(torch.full((n,), 0, dtype=torch.long)) + self.code(b["codes_ordered"][:, j]) + self.risk(b["risk_ordered"][:, j]) + self.energy(b["energy_ordered"][:, j]) + self.memory(b["memory_ordered"][:, j, None])
        # randomized generic relation tokens
        for j in range(3):
            x[:, 2+j] = self.type_emb(torch.full((n,), 1, dtype=torch.long)) + self.role(b["rel_roles"][:, j]) + self.code(b["rel_payload"][:, j])
        x[:, 5] = self.type_emb(torch.full((n,), 2, dtype=torch.long)) + self.scalar(b["target"][:, None])
        x[:, 6] = self.type_emb(torch.full((n,), 3, dtype=torch.long)) + self.rule(b["rules"])
        # randomized phase-query order
        for j in range(4):
            x[:, 7+j] = self.type_emb(torch.full((n,), 4, dtype=torch.long)) + self.phase(b["phase_order"][:, j])
        return x + self.pos[None]

    def forward(self, b, return_states=False):
        h = self.embed(b); states = [h]
        for blk in self.blocks:
            h = blk(h); states.append(h)
        h = self.ln(h); states.append(h)
        z = h[:, 7:11]
        return dict(policy=5*self.policy(z), outcome=self.outcome(z), states=states if return_states else None)


def unpermute_queries(x, phase_order):
    # x is [N,4,...] in query order. Return canonical phase order.
    inv = torch.argsort(phase_order, dim=1)
    if x.ndim == 2:
        return x.gather(1, inv)
    return x.gather(1, inv[:, :, None].expand(-1, -1, x.shape[-1]))


def train(cfg, seed, out):
    seed_all(seed); torch.set_num_threads(cfg.threads)
    m = Model(cfg); opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(1600000 + seed); hist = []
    for st in range(cfg.steps):
        b = make_batch(cfg.batch, cfg, g, "train")
        o = m(b)
        ce = F.cross_entropy(o["policy"].reshape(-1, 4), b["actions"].reshape(-1), reduction="none").reshape(-1, 4)
        # Weight semantic phases after mapping canonical weights into the episode-specific query order.
        phase_w_can = torch.tensor([1.45, 1.45, 0.45, 1.45])
        phase_w_q = phase_w_can[b["phase_order"]]
        la = (ce * phase_w_q).sum() / phase_w_q.sum(); lo = F.mse_loss(torch.sigmoid(o["outcome"]), b["outcomes"])
        loss = la + cfg.outcome_weight * lo
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 25 == 0 or st == cfg.steps - 1:
            hist.append(dict(step=st, loss=float(loss.detach()), action_acc=float((o["policy"].argmax(-1) == b["actions"]).float().mean()), outcome_mse=float(lo.detach())))
        if st % 25 == 0 or st == cfg.steps - 1:
            print(json.dumps({"seed": seed, "step": st, "loss": float(loss.detach()), "action_acc": float((o["policy"].argmax(-1) == b["actions"]).float().mean())}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), out/"checkpoint.pt")
    pd.DataFrame(hist).to_csv(out/"history.csv", index=False)
    (out/"metadata.json").write_text(json.dumps(dict(version="V16-E.6B", seed=seed, config=asdict(cfg), finished_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    return m


def evaluate_mode(m, cfg, seed, mode):
    b = make_batch(cfg.eval_n, cfg, torch.Generator().manual_seed(1700000 + seed + sum(map(ord, mode))), mode)
    with torch.inference_mode(): o = m(b)
    pred_can = unpermute_queries(o["policy"].argmax(-1), b["phase_order"])
    out_can = unpermute_queries(torch.sigmoid(o["outcome"]), b["phase_order"])
    rows = []
    for q, name in enumerate(["continuity", "allocation", "protection", "integrated"]):
        rows.append(dict(seed=seed, eval_mode=mode, phase=q, name=name,
                         action_acc=float((pred_can[:, q] == b["actions_canonical"][:, q]).float().mean()),
                         outcome_mae=float((out_can[:, q] - b["outcomes_canonical"][:, q]).abs().mean())))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--seed", type=int, default=16160); ap.add_argument("--steps", type=int, default=950)
    a = ap.parse_args(); root = Path(a.root); cfg = Config(steps=a.steps, codebook_seed=a.seed); out = root/"raw"/"V16-E.6B"/f"orthogonal_replication_seed{a.seed}"
    m = train(cfg, a.seed, out); rows = []
    for mode in ["id", "encoding_ood", "factor_ood", "rule_ood"]:
        rows.extend(evaluate_mode(m, cfg, a.seed, mode))
    df = pd.DataFrame(rows); df.to_csv(out/"base_metrics.csv", index=False); print(df.to_string(index=False))

if __name__ == "__main__":
    main()
