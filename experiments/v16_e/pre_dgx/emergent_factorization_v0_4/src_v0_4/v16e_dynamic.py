"""V16-E v0.4 dynamic identifiability pilot.
Unified Transformer, shared policy/outcome heads, no factor heads/slots/gains.
"""
from __future__ import annotations
import argparse, json, random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F

try: torch.use_deterministic_algorithms(True)
except Exception: pass

@dataclass
class Config:
    n_bits:int=3; n_symbols:int=8; d:int=64; heads:int=4; layers:int=2
    batch:int=256; steps:int=650; lr:float=2e-3; outcome_weight:float=.8
    self_weight:float=1.1; ben_weight:float=1.65; cost_one:float=.62; cost_both:float=1.08
    threads:int=4; eval_n:int=1600
    @property
    def L(self): return 10

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def bits_value(x):
    p=(2**torch.arange(x.shape[-1])).float();return (x*p).sum(-1)/(2**x.shape[-1]-1)
def loss_table(ra,rb,wa,wb,cfg):
    return torch.stack([wa*ra+wb*rb,torch.full_like(ra,cfg.cost_one)+wb*rb,torch.full_like(ra,cfg.cost_one)+wa*ra,torch.full_like(ra,cfg.cost_both)],-1)

def action_for_entity(entity, needed):
    # action 0 none, 1 A, 2 B, 3 both
    return torch.where(needed,torch.where(entity==0,torch.ones_like(entity),torch.full_like(entity,2)),torch.zeros_like(entity))

def apply_action(base, action, benefit=.5):
    # base [n,2] in [0,1]; action adds benefit to selected entities, clipped.
    out=base.clone();out[:,0]+=benefit*((action==1)|(action==3)).float();out[:,1]+=benefit*((action==2)|(action==3)).float();return out.clamp(0,1)

def make_batch(n,cfg,gen,base=None,flip=None):
    B=cfg.n_bits
    if base is None:
        risk_bits=torch.randint(0,2,(n,2,B),generator=gen).float();energy_bits=torch.randint(0,2,(n,2,B),generator=gen).float();memory=torch.randint(0,2,(n,2),generator=gen).float();target=torch.randint(0,2,(n,),generator=gen).float()
        ident=torch.randint(0,2,(n,),generator=gen);ben=torch.randint(0,2,(n,),generator=gen);conc=torch.randint(0,2,(n,),generator=gen)
        sym0=torch.randint(0,cfg.n_symbols,(n,),generator=gen);off=torch.randint(1,cfg.n_symbols,(n,),generator=gen);sym1=(sym0+off)%cfg.n_symbols;symbols=torch.stack([sym0,sym1],1);order=torch.randint(0,2,(n,),generator=gen)
    else:
        risk_bits=base['risk_bits'].clone();energy_bits=base['energy_bits'].clone();memory=base['memory'].clone();target=base['target'].clone();ident=base['identity'].clone();ben=base['beneficiary'].clone();conc=base['concern'].clone();symbols=base['symbols'].clone();order=base['order'].clone()
    if flip=='identity':ident=1-ident
    elif flip=='beneficiary':ben=1-ben
    elif flip=='concern':conc=1-conc
    row=torch.arange(n);risk=bits_value(risk_bits);energy=bits_value(energy_bits)
    # t0 continuity: preserve identity-linked memory only when it mismatches required target.
    need0=(memory[row,ident]!=target);a0=action_for_entity(ident,need0)
    mem_out=memory.clone();sel0=((a0==1)|(a0==3));sel1=((a0==2)|(a0==3));mem_out[:,0]=torch.where(sel0,target,mem_out[:,0]);mem_out[:,1]=torch.where(sel1,target,mem_out[:,1])
    # t1 allocation: replenish beneficiary if terminal energy is below .5.
    need1=energy[row,ben]<.5;a1=action_for_entity(ben,need1);energy_out=apply_action(energy,a1,.5)
    # t2 common protection: concern=1 protects every body above .5 risk.
    hi0=risk[:,0]>.5;hi1=risk[:,1]>.5;both=hi0&hi1;only0=hi0&~hi1;only1=hi1&~hi0
    a2=torch.zeros(n,dtype=torch.long);a2=torch.where((conc==1)&only0,torch.ones_like(a2),a2);a2=torch.where((conc==1)&only1,torch.full_like(a2,2),a2);a2=torch.where((conc==1)&both,torch.full_like(a2,3),a2)
    harm_out=risk.clone();harm_out[:,0]*=torch.where((a2==1)|(a2==3),torch.tensor(.25),torch.tensor(1.));harm_out[:,1]*=torch.where((a2==2)|(a2==3),torch.tensor(.25),torch.tensor(1.))
    # t3 integrated policy.
    wa=conc.float()+cfg.self_weight*(ident==0).float()+cfg.ben_weight*(ben==0).float();wb=conc.float()+cfg.self_weight*(ident==1).float()+cfg.ben_weight*(ben==1).float();a3=loss_table(risk[:,0],risk[:,1],wa,wb,cfg).argmin(-1)
    integrated_out=harm_out.clone() # auxiliary target remains factual post-protection harm
    actions=torch.stack([a0,a1,a2,a3],1)
    outcomes=torch.stack([mem_out,energy_out,harm_out,integrated_out],1)
    entity_order=torch.stack([order,1-order],1);risk_ord=risk_bits.gather(1,entity_order[:,:,None].expand(-1,-1,B));energy_ord=energy_bits.gather(1,entity_order[:,:,None].expand(-1,-1,B));mem_ord=memory.gather(1,entity_order);sym_ord=symbols.gather(1,entity_order)
    return dict(risk_bits=risk_bits,energy_bits=energy_bits,memory=memory,target=target,identity=ident,beneficiary=ben,concern=conc,symbols=symbols,order=order,entity_order=entity_order,risk_ordered=risk_ord,energy_ordered=energy_ord,memory_ordered=mem_ord,symbol_ordered=sym_ord,id_symbol=symbols[row,ident],ben_symbol=symbols[row,ben],actions=actions,outcomes=outcomes)

class Model(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;d=cfg.d;B=cfg.n_bits
        self.type_emb=nn.Embedding(8,d);self.sym=nn.Embedding(cfg.n_symbols,d);self.body=nn.Embedding(2,d);self.risk=nn.Linear(B,d);self.energy=nn.Linear(B,d);self.memory=nn.Linear(1,d);self.scalar=nn.Linear(1,d);self.phase=nn.Embedding(4,d);self.pos=nn.Parameter(torch.randn(cfg.L,d)*.02)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d,cfg.heads,3*d,dropout=0.,activation='gelu',batch_first=True) for _ in range(cfg.layers)]);self.ln=nn.LayerNorm(d)
        self.policy=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,4));self.outcome=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))
    def embed(self,b):
        n=b['identity'].shape[0];x=torch.zeros(n,self.cfg.L,self.cfg.d)
        for j in range(2):
            x[:,j]=self.type_emb(torch.full((n,),j,dtype=torch.long))+self.sym(b['symbol_ordered'][:,j])+self.body(b['entity_order'][:,j])+self.risk(b['risk_ordered'][:,j])+self.energy(b['energy_ordered'][:,j])+self.memory(b['memory_ordered'][:,j,None])
        x[:,2]=self.type_emb(torch.full((n,),2,dtype=torch.long))+self.sym(b['id_symbol'])+self.body(b['identity'])
        x[:,3]=self.type_emb(torch.full((n,),3,dtype=torch.long))+self.sym(b['ben_symbol'])+self.body(b['beneficiary'])
        x[:,4]=self.type_emb(torch.full((n,),4,dtype=torch.long))+self.scalar(b['concern'].float()[:,None])
        x[:,5]=self.type_emb(torch.full((n,),5,dtype=torch.long))+self.scalar(b['target'][:,None])
        for q in range(4):x[:,6+q]=self.type_emb(torch.full((n,),6,dtype=torch.long))+self.phase(torch.full((n,),q,dtype=torch.long))
        return x+self.pos[None]
    def forward(self,b,return_states=False):
        h=self.embed(b);states=[h]
        for blk in self.blocks:h=blk(h);states.append(h)
        h=self.ln(h);states.append(h);z=h[:,6:10];return dict(policy=5*self.policy(z),outcome=self.outcome(z),states=states if return_states else None)

def train(cfg,seed,out):
    seed_all(seed);torch.set_num_threads(cfg.threads);m=Model(cfg);opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=1e-4);g=torch.Generator().manual_seed(1200000+seed);hist=[]
    for st in range(cfg.steps):
        b=make_batch(cfg.batch,cfg,g);o=m(b);la=F.cross_entropy(o['policy'].reshape(-1,4),b['actions'].reshape(-1));lo=F.mse_loss(torch.sigmoid(o['outcome']),b['outcomes']);loss=la+cfg.outcome_weight*lo;opt.zero_grad();loss.backward();opt.step()
        if st%25==0 or st==cfg.steps-1:hist.append(dict(step=st,loss=float(loss.detach()),action_acc=float((o['policy'].argmax(-1)==b['actions']).float().mean()),outcome_mse=float(lo.detach())))
    out.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),out/'checkpoint.pt');pd.DataFrame(hist).to_csv(out/'history.csv',index=False);(out/'metadata.json').write_text(json.dumps(dict(seed=seed,config=asdict(cfg),finished_utc=datetime.now(timezone.utc).isoformat()),indent=2));return m

def evaluate(m,cfg,seed,out):
    b=make_batch(cfg.eval_n,cfg,torch.Generator().manual_seed(1300000+seed));
    with torch.inference_mode():o=m(b)
    pred=o['policy'].argmax(-1);rows=[]
    for q,name in enumerate(['continuity','allocation','protection','integrated']):rows.append(dict(seed=seed,phase=q,name=name,action_acc=float((pred[:,q]==b['actions'][:,q]).float().mean()),outcome_mae=float((torch.sigmoid(o['outcome'][:,q])-b['outcomes'][:,q]).abs().mean())))
    pd.DataFrame(rows).to_csv(out/'base_metrics.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seed',type=int,default=16140);ap.add_argument('--steps',type=int,default=650);a=ap.parse_args();root=Path(a.root);cfg=Config(steps=a.steps);out=root/'raw/pilot_v0_4'/f'dynamic_behavior_only_seed{a.seed}';m=train(cfg,a.seed,out);evaluate(m,cfg,a.seed,out);print(pd.read_csv(out/'base_metrics.csv').to_string(index=False))
if __name__=='__main__':main()
