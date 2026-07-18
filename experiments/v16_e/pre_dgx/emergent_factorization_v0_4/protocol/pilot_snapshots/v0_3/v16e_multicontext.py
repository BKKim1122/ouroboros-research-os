"""V16-E pilot v0.3: multi-context behavior-only identifiability test.

One shared Transformer receives the same latent role variables but is trained on
four behavior contexts. No factor labels, factor heads, factor slots, or factor
gains are used. The purpose is to test whether independent downstream causal
consequences are sufficient for factor-specific latent subspaces to emerge.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

try: torch.use_deterministic_algorithms(True)
except Exception: pass

@dataclass
class Config:
    n_bits:int=3; n_symbols:int=8; d:int=64; heads:int=4; layers:int=2; batch:int=256; steps:int=650; lr:float=2e-3
    self_weight:float=1.1; ben_weight:float=1.65; cost_one:float=.62; cost_both:float=1.08
    world_weight:float=.8; eval_n:int=2000; cal_n:int=2500; threads:int=4
    @property
    def L(self): return 7

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def bits_to_risk(x):
    p=(2**torch.arange(x.shape[-1])).float();return (x*p).sum(-1)/(2**x.shape[-1]-1)
def losses(ra,rb,wa,wb,cfg):
    return torch.stack([wa*ra+wb*rb,torch.full_like(ra,cfg.cost_one)+wb*rb,torch.full_like(ra,cfg.cost_one)+wa*ra,torch.full_like(ra,cfg.cost_both)],-1)
def make_batch(n,cfg,gen,query=None,base=None,flip=None):
    B=cfg.n_bits
    if base is None:
        risk_bits=torch.randint(0,2,(n,2,B),generator=gen).float();ident=torch.randint(0,2,(n,),generator=gen);ben=torch.randint(0,2,(n,),generator=gen);conc=torch.randint(0,2,(n,),generator=gen)
        sym0=torch.randint(0,cfg.n_symbols,(n,),generator=gen);off=torch.randint(1,cfg.n_symbols,(n,),generator=gen);sym1=(sym0+off)%cfg.n_symbols;symbols=torch.stack([sym0,sym1],1);order=torch.randint(0,2,(n,),generator=gen)
    else:
        risk_bits=base['risk_bits'].clone();ident=base['identity'].clone();ben=base['beneficiary'].clone();conc=base['concern'].clone();symbols=base['symbols'].clone();order=base['order'].clone()
    if flip=='identity':ident=1-ident
    elif flip=='beneficiary':ben=1-ben
    elif flip=='concern':conc=1-conc
    if query is None:q=torch.randint(0,4,(n,),generator=gen)
    elif isinstance(query,int):q=torch.full((n,),query,dtype=torch.long)
    else:q=query.clone()
    entity_order=torch.stack([order,1-order],1);risk_ordered=risk_bits.gather(1,entity_order[:,:,None].expand(-1,-1,B));symbol_ordered=symbols.gather(1,entity_order);row=torch.arange(n)
    risk=bits_to_risk(risk_bits);ra,rb=risk[:,0],risk[:,1]
    # q0: identity-conditioned motor choice. q1: beneficiary allocation.
    a_id=torch.where(ident==0,torch.ones_like(ident),torch.full_like(ident,2))
    a_ben=torch.where(ben==0,torch.ones_like(ben),torch.full_like(ben,2))
    # q2: symmetric common protection only.
    wc=conc.float();a_common=losses(ra,rb,wc,wc,cfg).argmin(-1)
    # q3: integrated V15-like policy.
    wa=conc.float()+cfg.self_weight*(ident==0).float()+cfg.ben_weight*(ben==0).float();wb=conc.float()+cfg.self_weight*(ident==1).float()+cfg.ben_weight*(ben==1).float();a_int=losses(ra,rb,wa,wb,cfg).argmin(-1)
    actions=torch.stack([a_id,a_ben,a_common,a_int],1);action=actions[row,q]
    return dict(risk_bits=risk_bits,risk=risk,identity=ident,beneficiary=ben,concern=conc,symbols=symbols,order=order,entity_order=entity_order,risk_ordered=risk_ordered,symbol_ordered=symbol_ordered,id_symbol=symbols[row,ident],ben_symbol=symbols[row,ben],query=q,all_actions=actions,action=action)

class Model(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;d=cfg.d;B=cfg.n_bits
        self.type_emb=nn.Embedding(7,d);self.sym=nn.Embedding(cfg.n_symbols,d);self.body=nn.Embedding(2,d);self.risk=nn.Linear(B,d);self.conc=nn.Linear(1,d);self.query=nn.Embedding(4,d);self.pos=nn.Parameter(torch.randn(cfg.L,d)*.02);self.cls=nn.Parameter(torch.randn(1,d)*.02)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d,cfg.heads,3*d,dropout=0.,activation='gelu',batch_first=True) for _ in range(cfg.layers)]);self.ln=nn.LayerNorm(d);self.policy=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,4));self.world=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,2*B))
    def embed(self,b):
        n=b['risk_bits'].shape[0];x=torch.zeros(n,self.cfg.L,self.cfg.d);x[:,0]=self.cls
        for j in range(2):x[:,1+j]=self.type_emb(torch.full((n,),1+j,dtype=torch.long))+self.sym(b['symbol_ordered'][:,j])+self.body(b['entity_order'][:,j])+self.risk(b['risk_ordered'][:,j])
        x[:,3]=self.type_emb(torch.full((n,),3,dtype=torch.long))+self.sym(b['id_symbol'])+self.body(b['identity'])
        x[:,4]=self.type_emb(torch.full((n,),4,dtype=torch.long))+self.sym(b['ben_symbol'])+self.body(b['beneficiary'])
        x[:,5]=self.type_emb(torch.full((n,),5,dtype=torch.long))+self.conc(b['concern'].float()[:,None])
        x[:,6]=self.type_emb(torch.full((n,),6,dtype=torch.long))+self.query(b['query'])
        return x+self.pos[None]
    def encode(self,b,return_layers=False,override_layer=None,override_cls=None):
        h=self.embed(b);states=[h]
        if override_layer==0:h=h.clone();h[:,0]=override_cls
        for i,blk in enumerate(self.blocks,1):
            h=blk(h)
            if override_layer==i:h=h.clone();h[:,0]=override_cls
            states.append(h)
        h=self.ln(h);states.append(h);return h,states if return_layers else []
    def heads(self,z):return dict(policy=5.*self.policy(z),world=self.world(z).view(-1,2,self.cfg.n_bits))
    def forward(self,b,return_layers=False):
        h,s=self.encode(b,return_layers);o=self.heads(h[:,0]);o['states']=s if return_layers else None;return o
    def from_layer(self,b,li,z):h,_=self.encode(b,False,li,z);return self.heads(h[:,0])

def train(cfg,seed,out):
    seed_all(seed);torch.set_num_threads(cfg.threads);m=Model(cfg);opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=1e-4);gen=torch.Generator().manual_seed(100000+seed);hist=[]
    for st in range(cfg.steps):
        b=make_batch(cfg.batch,cfg,gen);o=m(b);la=F.cross_entropy(o['policy'],b['action']);lw=F.binary_cross_entropy_with_logits(o['world'],b['risk_bits']);loss=la+cfg.world_weight*lw;opt.zero_grad();loss.backward();opt.step()
        if st%25==0 or st==cfg.steps-1:hist.append(dict(step=st,loss=float(loss),action_acc=float((o['policy'].argmax(-1)==b['action']).float().mean()),world_loss=float(lw)))
    out.mkdir(parents=True,exist_ok=True);pd.DataFrame(hist).to_csv(out/'history.csv',index=False);torch.save(m.state_dict(),out/'checkpoint.pt');(out/'metadata.json').write_text(json.dumps(dict(seed=seed,config=asdict(cfg),finished_utc=datetime.now(timezone.utc).isoformat()),indent=2));return m

def collect(m,b):
    with torch.no_grad():o=m(b,True)
    return [s[:,0].numpy() for s in o['states']],{k:v.numpy() for k,v in o.items() if k!='states' and v is not None}
def bacc(y,p):return float(balanced_accuracy_score(y,p))
def fit_probe(x,y):sc=StandardScaler().fit(x);cl=LogisticRegression(max_iter=2000,solver='liblinear').fit(sc.transform(x),y);return sc,cl
def build_basis(m,cfg,seed,factor,query,layer,k=4):
    b=make_batch(cfg.cal_n,cfg,torch.Generator().manual_seed(300000+seed+query),query=query);cf=make_batch(cfg.cal_n,cfg,torch.Generator().manual_seed(1),query=query,base=b,flip=factor);z=collect(m,b)[0][layer];zc=collect(m,cf)[0][layer];sign=np.where(b[factor].numpy()==0,1.,-1.)[:,None];D=(zc-z)*sign;D-=D.mean(0);_,_,vt=np.linalg.svd(D,full_matrices=False);return vt[:k].T

def evaluate(m,cfg,seed,out,layer=2,k=4):
    rows=[];prows=[];factors=['identity','beneficiary','concern'];home={'identity':0,'beneficiary':1,'concern':2}
    # Probe on each query and cross-query generalization to integrated q3.
    for q in range(4):
        b=make_batch(cfg.eval_n,cfg,torch.Generator().manual_seed(400000+seed+q),query=q);ls,o=collect(m,b);pred=o['policy'].argmax(-1);rows.append(dict(seed=seed,query=q,action_acc=float((pred==b['action'].numpy()).mean()),world_bacc=bacc(b['risk_bits'].numpy().reshape(-1),(o['world'].reshape(-1)>0).astype(int))))
    trainsets={};
    for f in factors:
        q=home[f];b=make_batch(cfg.cal_n,cfg,torch.Generator().manual_seed(500000+seed+q),query=q);ls,_=collect(m,b);trainsets[f]=(ls[layer],b[f].numpy(),fit_probe(ls[layer],b[f].numpy()))
        for tq in range(4):
            tb=make_batch(cfg.eval_n,cfg,torch.Generator().manual_seed(510000+seed+10*q+tq),query=tq);tls,_=collect(m,tb);sc,cl=trainsets[f][2];prows.append(dict(seed=seed,factor=f,train_query=q,test_query=tq,bacc=bacc(tb[f].numpy(),cl.predict(sc.transform(tls[layer])))))
    bases={f:build_basis(m,cfg,seed,f,home[f],layer,k) for f in factors}
    irows=[];rng=np.random.default_rng(600000+seed)
    for target in factors:
      for q in range(4):
        b=make_batch(cfg.eval_n,cfg,torch.Generator().manual_seed(610000+seed+q),query=q);cf=make_batch(cfg.eval_n,cfg,torch.Generator().manual_seed(1),query=q,base=b,flip=target);ls,o=collect(m,b);lsc,oc=collect(m,cf);z=ls[layer];zc=lsc[layer];pred=o['policy'].argmax(-1);predc=oc['policy'].argmax(-1);conf=pred!=predc;delta=zc-z
        for source in factors+['random']:
            if source=='random':U=np.linalg.qr(rng.normal(size=(cfg.d,k)))[0][:,:k]
            else:U=bases[source]
            zs=z+(delta@U)@U.T
            with torch.no_grad():os=m.from_layer(b,layer,torch.tensor(zs,dtype=torch.float32))
            ps=os['policy'].numpy().argmax(-1);irows.append(dict(seed=seed,target_factor=target,query=q,source_subspace=source,k=k,conflict_n=int(conf.sum()),follow=float((ps[conf]==predc[conf]).mean()) if conf.any() else np.nan,policy_change=float((ps!=pred).mean()),world_stability=float(((os['world'].numpy()>0)==(o['world']>0)).mean())))
    pd.DataFrame(rows).to_csv(out/'query_metrics.csv',index=False);pd.DataFrame(prows).to_csv(out/'probe_cross_query.csv',index=False);pd.DataFrame(irows).to_csv(out/'intervention_matrix.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seed',type=int,default=16130);ap.add_argument('--steps',type=int,default=650);a=ap.parse_args();root=Path(a.root);cfg=Config(steps=a.steps);out=root/'raw/pilot_v0_3'/f'behavior_only_seed{a.seed}';m=train(cfg,a.seed,out);evaluate(m,cfg,a.seed,out);print('done',out)
if __name__=='__main__':main()
