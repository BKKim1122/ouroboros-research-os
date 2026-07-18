"""V16-E.C1 frozen confirmatory experiment.

Two parameter-count-matched rule encodings are compared:
- compositional: shared generic comparator embeddings + domain/numeric tokens
- packed: the same six rule scalars packed into a repeated rule vector

No factor-specific latent slots, gains, routes, factor reports, or factor-label losses.
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
    d: int = 56
    heads: int = 4
    layers: int = 2
    batch: int = 160
    steps: int = 800
    lr: float = 1.7e-3
    outcome_weight: float = 1.0
    self_weight: float = 1.1
    ben_weight: float = 1.65
    cost_one: float = 0.62
    cost_both: float = 1.08
    threads: int = 1
    eval_n: int = 1200
    heldout_factor_combos: tuple = ((0, 0, 0), (1, 1, 1))
    train_op_combos: tuple = ((1, 1), (-1, 1), (1, -1))
    heldout_op_combo: tuple = (-1, -1)

    @property
    def L(self):
        # 2 entity + 3 relation + target + 4 rule tokens + 4 phase queries
        return 14


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def bits_value(x: torch.Tensor) -> torch.Tensor:
    p = (2 ** torch.arange(x.shape[-1], device=x.device)).float()
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
        vals = combos[torch.randint(0, len(combos), (n,), generator=gen)]
        return vals[:, 0], vals[:, 1], vals[:, 2]
    held = set(tuple(x) for x in cfg.heldout_factor_combos)
    allowed = torch.tensor([(i,b,c) for i in (0,1) for b in (0,1) for c in (0,1) if (i,b,c) not in held], dtype=torch.long)
    vals = allowed[torch.randint(0, len(allowed), (n,), generator=gen)]
    return vals[:, 0], vals[:, 1], vals[:, 2]


def _normalize(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8) * math.sqrt(v.shape[-1])


def sample_entity_codes(n, gen, mode, cfg):
    D = cfg.code_dim
    if mode == "encoding_ood":
        idx0 = torch.randint(0, D, (n,), generator=gen)
        idx1 = torch.randint(0, D - 1, (n,), generator=gen)
        idx1 = idx1 + (idx1 >= idx0).long()
        s0 = torch.where(torch.randint(0, 2, (n,), generator=gen) == 0, -1.0, 1.0)
        s1 = torch.where(torch.randint(0, 2, (n,), generator=gen) == 0, -1.0, 1.0)
        codes = torch.zeros(n, 2, D)
        codes[torch.arange(n), 0, idx0] = s0
        codes[torch.arange(n), 1, idx1] = s1
        return codes * math.sqrt(D)
    fam = torch.randint(0, 2, (n,), generator=gen)
    rad = torch.where(torch.randint(0, 2, (n, 2, D), generator=gen) == 0, -1.0, 1.0)
    gau = torch.randn(n, 2, D, generator=gen)
    codes = _normalize(torch.where(fam[:, None, None] == 0, rad, gau))
    sim = (codes[:, 0] * codes[:, 1]).sum(-1).abs() / D
    for _ in range(8):
        bad = sim > 0.75
        if not bad.any(): break
        repl = _normalize(torch.randn(int(bad.sum()), D, generator=gen))
        codes[bad, 1] = repl
        sim = (codes[:, 0] * codes[:, 1]).sum(-1).abs() / D
    return codes


def sample_rules(n, gen, mode, cfg):
    if mode == "rule_ood":
        signs = torch.tensor(cfg.heldout_op_combo, dtype=torch.float32)[None].repeat(n, 1)
    else:
        combos = torch.tensor(cfg.train_op_combos, dtype=torch.float32)
        signs = combos[torch.randint(0, len(combos), (n,), generator=gen)]
    # Numeric values are independently sampled from shared grids in every operator combination.
    # The held-out condition is therefore only a new operator composition, not a new scalar range.
    def pick(vals):
        v = torch.tensor(vals, dtype=torch.float32)
        return v[torch.randint(0, len(v), (n,), generator=gen)]
    alloc_thresh = pick((0.35, 0.50, 0.65))
    protect_thresh = pick((0.35, 0.50, 0.65))
    alloc_benefit = pick((0.40, 0.50, 0.60))
    protect_factor = pick((0.22, 0.34, 0.46))
    return torch.stack([alloc_thresh, protect_thresh, alloc_benefit, protect_factor, signs[:,0], signs[:,1]], 1)


def make_batch(n, cfg, gen, mode="train", base=None, flip=None, flip_rule=None):
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
        rel_order = torch.argsort(torch.rand(n, 3, generator=gen), dim=1)
        rule_order = torch.argsort(torch.rand(n, 4, generator=gen), dim=1)
        phase_order = torch.argsort(torch.rand(n, 4, generator=gen), dim=1)
    else:
        keys = ["risk_bits","energy_bits","memory","target","identity","beneficiary","concern","codes","order","rules","rel_order","rule_order","phase_order"]
        vals = {k: base[k].clone() for k in keys}
        risk_bits=vals["risk_bits"]; energy_bits=vals["energy_bits"]; memory=vals["memory"]; target=vals["target"]
        ident=vals["identity"]; ben=vals["beneficiary"]; conc=vals["concern"]; codes=vals["codes"]; order=vals["order"]; rules=vals["rules"]
        rel_order=vals["rel_order"]; rule_order=vals["rule_order"]; phase_order=vals["phase_order"]
    if flip == "identity": ident = 1-ident
    elif flip == "beneficiary": ben = 1-ben
    elif flip == "concern": conc = 1-conc
    if flip_rule == "allocation": rules[:,4] *= -1
    elif flip_rule == "protection": rules[:,5] *= -1

    row = torch.arange(n)
    risk = bits_value(risk_bits); energy = bits_value(energy_bits)
    entity_order = torch.stack([order, 1-order], 1)
    risk_ord = risk.gather(1, entity_order); energy_ord = energy.gather(1, entity_order); mem_ord = memory.gather(1, entity_order)
    risk_bits_ord = risk_bits.gather(1, entity_order[:,:,None].expand(-1,-1,B))
    energy_bits_ord = energy_bits.gather(1, entity_order[:,:,None].expand(-1,-1,B))
    codes_ord = codes.gather(1, entity_order[:,:,None].expand(-1,-1,cfg.code_dim))

    alloc_thresh, protect_thresh, alloc_benefit, protect_factor, alloc_sign, protect_sign = [rules[:,i] for i in range(6)]
    need0 = memory[row, ident] != target
    a0 = action_for_entity_slot(ident, need0, entity_order)
    mem_out = mem_ord.clone()
    mem_out[:,0] = torch.where((a0==1)|(a0==3), target, mem_out[:,0])
    mem_out[:,1] = torch.where((a0==2)|(a0==3), target, mem_out[:,1])

    e_ben = energy[row, ben]
    need1 = torch.where(alloc_sign > 0, e_ben < alloc_thresh, e_ben > alloc_thresh)
    a1 = action_for_entity_slot(ben, need1, entity_order)
    energy_out = apply_action_slot(energy_ord, a1, alloc_benefit)

    cond0 = torch.where(protect_sign > 0, risk_ord[:,0] > protect_thresh, risk_ord[:,0] < protect_thresh)
    cond1 = torch.where(protect_sign > 0, risk_ord[:,1] > protect_thresh, risk_ord[:,1] < protect_thresh)
    a2 = torch.zeros(n, dtype=torch.long)
    a2 = torch.where((conc==1)&cond0&~cond1, torch.ones_like(a2), a2)
    a2 = torch.where((conc==1)&cond1&~cond0, torch.full_like(a2,2), a2)
    a2 = torch.where((conc==1)&cond0&cond1, torch.full_like(a2,3), a2)
    harm_out = risk_ord.clone()
    harm_out[:,0] *= torch.where((a2==1)|(a2==3), protect_factor, torch.ones_like(protect_factor))
    harm_out[:,1] *= torch.where((a2==2)|(a2==3), protect_factor, torch.ones_like(protect_factor))

    wa = conc.float() + cfg.self_weight*(ident==0).float() + cfg.ben_weight*(ben==0).float()
    wb = conc.float() + cfg.self_weight*(ident==1).float() + cfg.ben_weight*(ben==1).float()
    a3c = loss_table(risk[:,0], risk[:,1], wa, wb, cfg).argmin(-1)
    a3 = canonical_action_to_slot(a3c, entity_order)
    actions_can = torch.stack([a0,a1,a2,a3],1)
    outcomes_can = torch.stack([mem_out,energy_out,harm_out,harm_out],1)
    actions_q = actions_can.gather(1, phase_order)
    outcomes_q = outcomes_can.gather(1, phase_order[:,:,None].expand(-1,-1,2))

    id_code = codes[row, ident]; ben_code = codes[row, ben]
    conc_code = torch.zeros(n, cfg.code_dim); conc_code[:,0] = torch.where(conc==1,1.0,-1.0)
    rel_payload_can = torch.stack([id_code,ben_code,conc_code],1)
    rel_role_can = torch.arange(3)[None].repeat(n,1)
    rel_payload = rel_payload_can.gather(1, rel_order[:,:,None].expand(-1,-1,cfg.code_dim))
    rel_roles = rel_role_can.gather(1, rel_order)

    # Canonical compositional rule tokens: alloc comparator, alloc numeric, protect comparator, protect numeric.
    rule_domain_can = torch.tensor([0,0,1,1])[None].repeat(n,1)
    rule_kind_can = torch.tensor([0,1,0,1])[None].repeat(n,1)  # 0 comparator, 1 numeric
    rule_op_can = torch.stack([(alloc_sign<0).long(), torch.zeros(n,dtype=torch.long), (protect_sign<0).long(), torch.zeros(n,dtype=torch.long)],1)
    rule_num_can = torch.zeros(n,4,2)
    rule_num_can[:,1] = torch.stack([alloc_thresh,alloc_benefit],1)
    rule_num_can[:,3] = torch.stack([protect_thresh,protect_factor],1)
    rule_domain = rule_domain_can.gather(1, rule_order)
    rule_kind = rule_kind_can.gather(1, rule_order)
    rule_op = rule_op_can.gather(1, rule_order)
    rule_num = rule_num_can.gather(1, rule_order[:,:,None].expand(-1,-1,2))

    return dict(risk_bits=risk_bits,energy_bits=energy_bits,memory=memory,target=target,identity=ident,beneficiary=ben,concern=conc,
                codes=codes,order=order,entity_order=entity_order,risk_ordered=risk_bits_ord,energy_ordered=energy_bits_ord,memory_ordered=mem_ord,codes_ordered=codes_ord,
                rules=rules,rel_order=rel_order,rel_payload=rel_payload,rel_roles=rel_roles,rule_order=rule_order,rule_domain=rule_domain,rule_kind=rule_kind,rule_op=rule_op,rule_num=rule_num,
                phase_order=phase_order,actions=actions_q,outcomes=outcomes_q,actions_canonical=actions_can,outcomes_canonical=outcomes_can)


class Model(nn.Module):
    def __init__(self, cfg, rule_mode="compositional"):
        super().__init__(); self.cfg=cfg; self.rule_mode=rule_mode; d=cfg.d; B=cfg.n_bits
        self.type_emb=nn.Embedding(6,d); self.code=nn.Linear(cfg.code_dim,d,bias=False); self.role=nn.Embedding(3,d); self.phase=nn.Embedding(4,d)
        self.risk=nn.Linear(B,d); self.energy=nn.Linear(B,d); self.memory=nn.Linear(1,d); self.scalar=nn.Linear(1,d)
        # Both modes instantiate all rule modules, keeping exact parameter count equal.
        self.rule_domain=nn.Embedding(2,d); self.rule_kind=nn.Embedding(2,d); self.rule_op=nn.Embedding(2,d); self.rule_num=nn.Linear(2,d)
        self.packed_rule=nn.Linear(6,d); self.packed_slot=nn.Embedding(4,d)
        self.pos=nn.Parameter(torch.randn(cfg.L,d)*0.02)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d,cfg.heads,3*d,dropout=0.,activation="gelu",batch_first=True) for _ in range(cfg.layers)])
        self.ln=nn.LayerNorm(d); self.policy=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,4)); self.outcome=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))

    def embed(self,b):
        n=b["identity"].shape[0]; device=self.pos.device
        x=torch.zeros(n,self.cfg.L,self.cfg.d,device=device)
        zlong=torch.zeros(n,dtype=torch.long,device=device)
        for j in range(2):
            x[:,j]=self.type_emb(zlong)+self.code(b["codes_ordered"][:,j])+self.risk(b["risk_ordered"][:,j])+self.energy(b["energy_ordered"][:,j])+self.memory(b["memory_ordered"][:,j,None])
        for j in range(3):
            x[:,2+j]=self.type_emb(torch.ones(n,dtype=torch.long,device=device))+self.role(b["rel_roles"][:,j])+self.code(b["rel_payload"][:,j])
        x[:,5]=self.type_emb(torch.full((n,),2,dtype=torch.long,device=device))+self.scalar(b["target"][:,None])
        if self.rule_mode == "compositional":
            for j in range(4):
                kind=b["rule_kind"][:,j]
                rule=self.rule_domain(b["rule_domain"][:,j])+self.rule_kind(kind)
                rule=rule+torch.where(kind[:,None]==0,self.rule_op(b["rule_op"][:,j]),self.rule_num(b["rule_num"][:,j]))
                x[:,6+j]=self.type_emb(torch.full((n,),3,dtype=torch.long,device=device))+rule
        else:
            for j in range(4):
                canon_slot=b["rule_order"][:,j]
                x[:,6+j]=self.type_emb(torch.full((n,),3,dtype=torch.long,device=device))+self.packed_rule(b["rules"])+self.packed_slot(canon_slot)
        for j in range(4):
            x[:,10+j]=self.type_emb(torch.full((n,),4,dtype=torch.long,device=device))+self.phase(b["phase_order"][:,j])
        return x+self.pos[None]

    def forward(self,b,return_states=False):
        h=self.embed(b); states=[h]
        for blk in self.blocks: h=blk(h); states.append(h)
        h=self.ln(h); states.append(h); z=h[:,10:14]
        return dict(policy=5*self.policy(z),outcome=self.outcome(z),states=states if return_states else None)


def unpermute_queries(x,phase_order):
    inv=torch.argsort(phase_order,dim=1)
    if x.ndim==2: return x.gather(1,inv)
    return x.gather(1,inv[:,:,None].expand(-1,-1,x.shape[-1]))


def train(cfg, seed, out, rule_mode, device, resume=False):
    torch.set_num_threads(cfg.threads); configure_determinism(); out.mkdir(parents=True,exist_ok=True)
    state_path=out/"training_state.pt"; hist=[]
    if resume and state_path.exists():
        stt=torch.load(state_path,map_location="cpu",weights_only=False)
        m=Model(cfg,rule_mode).to(device); m.load_state_dict(stt["model"])
        opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=1e-4); opt.load_state_dict(stt["optimizer"]); optimizer_to(opt,device)
        g=torch.Generator(); g.set_state(stt["generator_state"]); start_step=int(stt["next_step"]); hist=list(stt.get("history",[]))
        print(json.dumps({"version":"V16-E.C1","rule_mode":rule_mode,"seed":seed,"device":str(device),"resume_step":start_step}),flush=True)
    else:
        seed_all(seed); m=Model(cfg,rule_mode).to(device); opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=1e-4)
        g=torch.Generator().manual_seed(1600000+seed); start_step=0
    phase_w_can=torch.tensor([1.25,1.55,1.55,1.15],device=device)
    for st in range(start_step,cfg.steps):
        b_cpu=make_batch(cfg.batch,cfg,g,"train"); b=batch_to_device(b_cpu,device); o=m(b)
        ce=F.cross_entropy(o["policy"].reshape(-1,4),b["actions"].reshape(-1),reduction="none").reshape(-1,4)
        phase_w_q=phase_w_can[b["phase_order"]]
        la=(ce*phase_w_q).sum()/phase_w_q.sum(); lo=F.mse_loss(torch.sigmoid(o["outcome"]),b["outcomes"]); loss=la+cfg.outcome_weight*lo
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        acc=float((o["policy"].argmax(-1)==b["actions"]).float().mean().detach().cpu())
        if st%25==0 or st==cfg.steps-1:
            rec=dict(step=st,loss=float(loss.detach().cpu()),action_acc=acc,outcome_mse=float(lo.detach().cpu())); hist.append(rec)
            print(json.dumps({"version":"V16-E.C1","rule_mode":rule_mode,"seed":seed,"device":str(device),**rec}),flush=True)
        if st%50==49 or st==cfg.steps-1:
            cpu_state={k:v.detach().cpu() for k,v in m.state_dict().items()}
            opt_state=opt.state_dict()
            torch.save(dict(model=cpu_state,optimizer=opt_state,generator_state=g.get_state(),next_step=st+1,history=hist),state_path)
            pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
    cpu_state={k:v.detach().cpu() for k,v in m.state_dict().items()}
    torch.save(cpu_state,out/"checkpoint.pt"); pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
    meta=dict(version="V16-E.C1",seed=seed,rule_mode=rule_mode,parameter_count=sum(p.numel() for p in m.parameters()),config=asdict(cfg),device=str(device),torch_version=torch.__version__,finished_utc=datetime.now(timezone.utc).isoformat())
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)); return m

def evaluate_mode(m,cfg,seed,mode,device):
    b_cpu=make_batch(cfg.eval_n,cfg,torch.Generator().manual_seed(1700000+seed+sum(map(ord,mode))),mode)
    b=batch_to_device(b_cpu,device)
    with torch.inference_mode(): o=m(b)
    pred=unpermute_queries(o["policy"].argmax(-1),b["phase_order"]); out=unpermute_queries(torch.sigmoid(o["outcome"]),b["phase_order"])
    rows=[]
    for q,name in enumerate(["continuity","allocation","protection","integrated"]):
        rows.append(dict(seed=seed,rule_mode=m.rule_mode,eval_mode=mode,phase=q,name=name,action_acc=float((pred[:,q]==b["actions_canonical"][:,q]).float().mean().cpu()),outcome_mae=float((out[:,q]-b["outcomes_canonical"][:,q]).abs().mean().cpu())))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--rule-mode",choices=["compositional","packed"],required=True)
    ap.add_argument("--steps",type=int,default=1100); ap.add_argument("--threads",type=int,default=2); ap.add_argument("--device",default="auto")
    ap.add_argument("--train-only",action="store_true"); ap.add_argument("--eval-only",action="store_true"); ap.add_argument("--resume",action="store_true"); ap.add_argument("--eval-n",type=int,default=4000)
    a=ap.parse_args(); root=Path(a.root); out=root/"raw"/"V16-E.C1"/f"{a.rule_mode}_seed{a.seed}"; device=resolve_device(a.device); configure_determinism()
    if a.eval_only:
        meta=json.loads((out/"metadata.json").read_text()); cfg=Config(**meta["config"]); cfg.eval_n=a.eval_n; torch.set_num_threads(a.threads)
        m=Model(cfg,a.rule_mode).to(device); m.load_state_dict(torch.load(out/"checkpoint.pt",map_location=device,weights_only=True)); m.eval()
    else:
        cfg=Config(steps=a.steps,threads=a.threads,eval_n=a.eval_n); m=train(cfg,a.seed,out,a.rule_mode,device,resume=a.resume)
    if not a.train_only:
        rows=[]
        for mode in ["id","encoding_ood","factor_ood","rule_ood"]: rows.extend(evaluate_mode(m,cfg,a.seed,mode,device))
        df=pd.DataFrame(rows); df.to_csv(out/"base_metrics.csv",index=False); print(df.to_string(index=False))

if __name__=="__main__": main()
