"""V16-E.5 relational-binding pilot.
No canonical body embeddings, no factor heads/slots/gains.
Entity selection is defined over presented slots; role cues identify entities only by compositional symbol code.
"""
from __future__ import annotations
import argparse, json, random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

@dataclass
class Config:
    n_bits:int=3
    symbol_bits:int=4
    d:int=64
    heads:int=4
    layers:int=2
    batch:int=256
    steps:int=750
    lr:float=2e-3
    outcome_weight:float=.8
    self_weight:float=1.1
    ben_weight:float=1.65
    cost_one:float=.62
    cost_both:float=1.08
    threads:int=4
    eval_n:int=1800
    train_symbol_ids:tuple=(0,1,2,4,6,7,8,9,11,13,14,15)
    ood_symbol_ids:tuple=(3,5,10,12)
    heldout_factor_combos:tuple=((0,0,0),(1,1,1))
    train_rules:tuple=((.35,.45,.40,.25),(.50,.55,.50,.25),(.65,.65,.60,.30))
    ood_rule:tuple=(.425,.60,.55,.275)
    @property
    def L(self): return 11

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def bits_value(x):
    p=(2**torch.arange(x.shape[-1])).float()
    return (x*p).sum(-1)/(2**x.shape[-1]-1)

def int_to_bits(x,n_bits):
    shifts=torch.arange(n_bits)
    bits=((x[:,None]>>shifts)&1).float()
    return 2.0*bits-1.0

def loss_table(ra,rb,wa,wb,cfg):
    return torch.stack([
        wa*ra+wb*rb,
        torch.full_like(ra,cfg.cost_one)+wb*rb,
        torch.full_like(ra,cfg.cost_one)+wa*ra,
        torch.full_like(ra,cfg.cost_both)
    ],-1)

def canonical_to_slot(entity,order):
    # order[:,slot] gives canonical entity in that visible slot.
    return (order==entity[:,None]).long().argmax(-1)

def canonical_action_to_slot(action,order):
    # 0 none, 1 canonical A, 2 canonical B, 3 both -> visible slot action.
    out=action.clone()
    a_slot=canonical_to_slot(torch.zeros_like(action),order)
    b_slot=canonical_to_slot(torch.ones_like(action),order)
    out=torch.where(action==1,a_slot+1,out)
    out=torch.where(action==2,b_slot+1,out)
    return out

def action_for_entity_slot(entity,needed,order):
    slot=canonical_to_slot(entity,order)
    return torch.where(needed,slot+1,torch.zeros_like(slot))

def apply_action_slot(base_ordered,action,benefit):
    out=base_ordered.clone()
    out[:,0]+=benefit*((action==1)|(action==3)).float()
    out[:,1]+=benefit*((action==2)|(action==3)).float()
    return out.clamp(0,1)

def sample_factors(n,gen,mode,cfg):
    if mode=='factor_ood':
        combos=torch.tensor(cfg.heldout_factor_combos,dtype=torch.long)
        idx=torch.randint(0,len(combos),(n,),generator=gen)
        c=combos[idx]
        return c[:,0],c[:,1],c[:,2]
    ident=torch.randint(0,2,(n,),generator=gen)
    ben=torch.randint(0,2,(n,),generator=gen)
    conc=torch.randint(0,2,(n,),generator=gen)
    if mode in ('train','id','symbol_ood','rule_ood'):
        held=set(tuple(x) for x in cfg.heldout_factor_combos)
        bad=torch.tensor([tuple(map(int,x)) in held for x in torch.stack([ident,ben,conc],1)],dtype=torch.bool)
        while bad.any():
            k=int(bad.sum())
            ident[bad]=torch.randint(0,2,(k,),generator=gen)
            ben[bad]=torch.randint(0,2,(k,),generator=gen)
            conc[bad]=torch.randint(0,2,(k,),generator=gen)
            bad=torch.tensor([tuple(map(int,x)) in held for x in torch.stack([ident,ben,conc],1)],dtype=torch.bool)
    return ident,ben,conc

def sample_symbols(n,gen,mode,cfg):
    if mode=='curriculum_easy': ids=(0,1,2,4)
    else: ids=cfg.ood_symbol_ids if mode=='symbol_ood' else cfg.train_symbol_ids
    pool=torch.tensor(ids,dtype=torch.long)
    i0=torch.randint(0,len(pool),(n,),generator=gen)
    i1=torch.randint(0,len(pool)-1,(n,),generator=gen)
    i1=i1+(i1>=i0).long()
    return torch.stack([pool[i0],pool[i1]],1)

def sample_rules(n,gen,mode,cfg):
    if mode=='curriculum_easy':
        return torch.tensor(cfg.train_rules[1]).float()[None].repeat(n,1)
    if mode=='rule_ood':
        return torch.tensor(cfg.ood_rule).float()[None].repeat(n,1)
    rules=torch.tensor(cfg.train_rules).float()
    idx=torch.randint(0,len(rules),(n,),generator=gen)
    return rules[idx]

def make_batch(n,cfg,gen,mode='train',base=None,flip=None):
    B=cfg.n_bits
    if base is None:
        risk_bits=torch.randint(0,2,(n,2,B),generator=gen).float()
        energy_bits=torch.randint(0,2,(n,2,B),generator=gen).float()
        memory=torch.randint(0,2,(n,2),generator=gen).float()
        target=torch.randint(0,2,(n,),generator=gen).float()
        ident,ben,conc=sample_factors(n,gen,mode,cfg)
        symbols=sample_symbols(n,gen,mode,cfg)
        order=torch.randint(0,2,(n,),generator=gen)
        rules=sample_rules(n,gen,mode,cfg)
    else:
        risk_bits=base['risk_bits'].clone(); energy_bits=base['energy_bits'].clone(); memory=base['memory'].clone(); target=base['target'].clone()
        ident=base['identity'].clone(); ben=base['beneficiary'].clone(); conc=base['concern'].clone(); symbols=base['symbols'].clone(); order=base['order'].clone(); rules=base['rules'].clone()
    if flip=='identity': ident=1-ident
    elif flip=='beneficiary': ben=1-ben
    elif flip=='concern': conc=1-conc
    row=torch.arange(n)
    risk=bits_value(risk_bits); energy=bits_value(energy_bits)
    entity_order=torch.stack([order,1-order],1)
    risk_ord=risk.gather(1,entity_order); energy_ord=energy.gather(1,entity_order); mem_ord=memory.gather(1,entity_order); sym_ord=symbols.gather(1,entity_order)
    risk_bits_ord=risk_bits.gather(1,entity_order[:,:,None].expand(-1,-1,B)); energy_bits_ord=energy_bits.gather(1,entity_order[:,:,None].expand(-1,-1,B))
    alloc_thresh=rules[:,0]; protect_thresh=rules[:,1]; alloc_benefit=rules[:,2]; protect_factor=rules[:,3]
    # continuity in visible-slot action space.
    need0=(memory[row,ident]!=target)
    a0=action_for_entity_slot(ident,need0,entity_order)
    mem_out=mem_ord.clone()
    mem_out[:,0]=torch.where((a0==1)|(a0==3),target,mem_out[:,0]); mem_out[:,1]=torch.where((a0==2)|(a0==3),target,mem_out[:,1])
    # allocation.
    need1=energy[row,ben]<alloc_thresh
    a1=action_for_entity_slot(ben,need1,entity_order)
    energy_out=apply_action_slot(energy_ord,a1,alloc_benefit)
    # protection.
    hi0=risk_ord[:,0]>protect_thresh; hi1=risk_ord[:,1]>protect_thresh
    a2=torch.zeros(n,dtype=torch.long)
    a2=torch.where((conc==1)&hi0&~hi1,torch.ones_like(a2),a2)
    a2=torch.where((conc==1)&hi1&~hi0,torch.full_like(a2,2),a2)
    a2=torch.where((conc==1)&hi0&hi1,torch.full_like(a2,3),a2)
    harm_out=risk_ord.clone()
    harm_out[:,0]*=torch.where((a2==1)|(a2==3),protect_factor,torch.ones_like(protect_factor))
    harm_out[:,1]*=torch.where((a2==2)|(a2==3),protect_factor,torch.ones_like(protect_factor))
    # integrated policy computed canonically, then converted to visible-slot action.
    wa=conc.float()+cfg.self_weight*(ident==0).float()+cfg.ben_weight*(ben==0).float()
    wb=conc.float()+cfg.self_weight*(ident==1).float()+cfg.ben_weight*(ben==1).float()
    a3_c=loss_table(risk[:,0],risk[:,1],wa,wb,cfg).argmin(-1)
    a3=canonical_action_to_slot(a3_c,entity_order)
    actions=torch.stack([a0,a1,a2,a3],1)
    outcomes=torch.stack([mem_out,energy_out,harm_out,harm_out],1)
    return dict(
        risk_bits=risk_bits,energy_bits=energy_bits,memory=memory,target=target,identity=ident,beneficiary=ben,concern=conc,
        symbols=symbols,order=order,entity_order=entity_order,risk_ordered=risk_bits_ord,energy_ordered=energy_bits_ord,
        memory_ordered=mem_ord,symbol_ordered=sym_ord,symbol_bits_ordered=int_to_bits(sym_ord.reshape(-1),cfg.symbol_bits).reshape(n,2,cfg.symbol_bits),
        id_symbol=symbols[row,ident],ben_symbol=symbols[row,ben],id_symbol_bits=int_to_bits(symbols[row,ident],cfg.symbol_bits),ben_symbol_bits=int_to_bits(symbols[row,ben],cfg.symbol_bits),
        rules=rules,actions=actions,outcomes=outcomes
    )

class Model(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; d=cfg.d; B=cfg.n_bits
        self.type_emb=nn.Embedding(8,d); self.symbol=nn.Linear(cfg.symbol_bits,d,bias=False); self.risk=nn.Linear(B,d); self.energy=nn.Linear(B,d); self.memory=nn.Linear(1,d); self.scalar=nn.Linear(1,d); self.rule=nn.Linear(4,d); self.phase=nn.Embedding(4,d); self.pos=nn.Parameter(torch.randn(cfg.L,d)*.02)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d,cfg.heads,3*d,dropout=0.,activation='gelu',batch_first=True) for _ in range(cfg.layers)])
        self.ln=nn.LayerNorm(d); self.policy=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,4)); self.outcome=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))
    def embed(self,b):
        n=b['identity'].shape[0]; x=torch.zeros(n,self.cfg.L,self.cfg.d)
        for j in range(2):
            x[:,j]=self.type_emb(torch.full((n,),j,dtype=torch.long))+self.symbol(b['symbol_bits_ordered'][:,j])+self.risk(b['risk_ordered'][:,j])+self.energy(b['energy_ordered'][:,j])+self.memory(b['memory_ordered'][:,j,None])
        x[:,2]=self.type_emb(torch.full((n,),2,dtype=torch.long))+self.symbol(b['id_symbol_bits'])
        x[:,3]=self.type_emb(torch.full((n,),3,dtype=torch.long))+self.symbol(b['ben_symbol_bits'])
        x[:,4]=self.type_emb(torch.full((n,),4,dtype=torch.long))+self.scalar(b['concern'].float()[:,None])
        x[:,5]=self.type_emb(torch.full((n,),5,dtype=torch.long))+self.scalar(b['target'][:,None])
        x[:,6]=self.type_emb(torch.full((n,),7,dtype=torch.long))+self.rule(b['rules'])
        for q in range(4): x[:,7+q]=self.type_emb(torch.full((n,),6,dtype=torch.long))+self.phase(torch.full((n,),q,dtype=torch.long))
        return x+self.pos[None]
    def forward(self,b,return_states=False):
        h=self.embed(b); states=[h]
        for blk in self.blocks: h=blk(h); states.append(h)
        h=self.ln(h); states.append(h); z=h[:,7:11]
        return dict(policy=5*self.policy(z),outcome=self.outcome(z),states=states if return_states else None)

def train(cfg,seed,out):
    seed_all(seed); torch.set_num_threads(cfg.threads); m=Model(cfg); opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=1e-4); g=torch.Generator().manual_seed(1600000+seed); hist=[]
    for st in range(cfg.steps):
        mode='curriculum_easy' if st<175 else 'train'
        b=make_batch(cfg.batch,cfg,g,mode); o=m(b)
        ce=F.cross_entropy(o['policy'].reshape(-1,4),b['actions'].reshape(-1),reduction='none').reshape(-1,4)
        phase_w=torch.tensor([1.35,1.35,.65,1.35])
        la=(ce*phase_w[None]).mean(); lo=F.mse_loss(torch.sigmoid(o['outcome']),b['outcomes']); loss=la+cfg.outcome_weight*lo
        opt.zero_grad(); loss.backward(); opt.step()
        if st%25==0 or st==cfg.steps-1: hist.append(dict(step=st,loss=float(loss.detach()),action_acc=float((o['policy'].argmax(-1)==b['actions']).float().mean()),outcome_mse=float(lo.detach())))
    out.mkdir(parents=True,exist_ok=True); torch.save(m.state_dict(),out/'checkpoint.pt'); pd.DataFrame(hist).to_csv(out/'history.csv',index=False); (out/'metadata.json').write_text(json.dumps(dict(version='V16-E.5',seed=seed,config=asdict(cfg),finished_utc=datetime.now(timezone.utc).isoformat()),indent=2)); return m

def evaluate_mode(m,cfg,seed,out,mode):
    b=make_batch(cfg.eval_n,cfg,torch.Generator().manual_seed(1700000+seed+sum(map(ord,mode))),mode)
    with torch.inference_mode(): o=m(b)
    pred=o['policy'].argmax(-1); rows=[]
    for q,name in enumerate(['continuity','allocation','protection','integrated']):
        rows.append(dict(seed=seed,eval_mode=mode,phase=q,name=name,action_acc=float((pred[:,q]==b['actions'][:,q]).float().mean()),outcome_mae=float((torch.sigmoid(o['outcome'][:,q])-b['outcomes'][:,q]).abs().mean())))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--seed',type=int,default=16150); ap.add_argument('--steps',type=int,default=750); a=ap.parse_args(); root=Path(a.root); cfg=Config(steps=a.steps); out=root/'raw/V16-E.5'/f'relational_binding_seed{a.seed}'
    m=train(cfg,a.seed,out); rows=[]
    for mode in ['id','symbol_ood','factor_ood','rule_ood']: rows.extend(evaluate_mode(m,cfg,a.seed,out,mode))
    pd.DataFrame(rows).to_csv(out/'base_metrics.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
