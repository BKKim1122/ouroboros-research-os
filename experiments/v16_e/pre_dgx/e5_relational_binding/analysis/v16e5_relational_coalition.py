from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

def load(path):
    spec=importlib.util.spec_from_file_location('v16e5',str(path)); m=importlib.util.module_from_spec(spec); sys.modules['v16e5']=m; spec.loader.exec_module(m); return m

def cont(model,h,layer=1):
    for i in range(layer+1,model.cfg.layers+1): h=model.blocks[i-1](h)
    h=model.ln(h); z=h[:,7:11]
    return 5*model.policy(z)

def norm_noise(base,donor,idx,seed):
    out=base.clone(); delta=donor[:,idx]-base[:,idx]; g=torch.Generator().manual_seed(seed); noise=torch.randn(delta.shape,generator=g)
    noise=noise/(noise.norm(dim=-1,keepdim=True)+1e-8)*delta.norm(dim=-1,keepdim=True); out[:,idx]=base[:,idx]+noise; return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[16150,16151,16152]); ap.add_argument('--n',type=int,default=800); a=ap.parse_args(); root=Path(a.root); mod=load(root/'src_v16e_5/v16e5_relational_binding.py')
    factors=['identity','beneficiary','concern']; role={'identity':2,'beneficiary':3,'concern':4}; phase={'identity':0,'beneficiary':1,'concern':2}; rows=[]
    for seed in a.seeds:
        run=root/'raw/V16-E.5'/f'relational_binding_seed{seed}'; meta=json.loads((run/'metadata.json').read_text()); cfg=mod.Config(**meta['config']); torch.set_num_threads(cfg.threads); model=mod.Model(cfg); model.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu')); model.eval()
        for fi,f in enumerate(factors):
            b=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(2200000+seed+fi),'id'); cf=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1),'id',base=b,flip=f)
            with torch.inference_mode(): ob=model(b,True); oc=model(cf,True)
            p0=ob['policy'].argmax(-1); pc=oc['policy'].argmax(-1); h=ob['states'][1]; hc=oc['states'][1]
            r=role[f]
            for q in [phase[f],3]:
                sets={
                    'role_only':[r], 'entities_only':[0,1], 'phase_only':[7+q],
                    'role_entities':[r,0,1], 'role_phase':[r,7+q], 'entities_phase':[0,1,7+q],
                    'full_relation_coalition':[r,0,1,7+q]
                }
                conf=p0[:,q]!=pc[:,q]
                for name,idx in sets.items():
                    hp=h.clone(); hp[:,idx]=hc[:,idx]
                    with torch.inference_mode(): pp=cont(model,hp,1).argmax(-1)[:,q]
                    rows.append(dict(seed=seed,factor=f,phase=q,patch=name,conflict_n=int(conf.sum()),follow=float((pp[conf]==pc[conf,q]).float().mean()) if conf.any() else np.nan,policy_change=float((pp!=p0[:,q]).float().mean())))
                idx=[r,0,1,7+q]
                perm=torch.randperm(a.n,generator=torch.Generator().manual_seed(2300000+seed+fi+q)); hp=h.clone(); hp[:,idx]=hc[perm][:,idx]
                with torch.inference_mode(): pp=cont(model,hp,1).argmax(-1)[:,q]
                rows.append(dict(seed=seed,factor=f,phase=q,patch='shuffled_full',conflict_n=int(conf.sum()),follow=float((pp[conf]==pc[conf,q]).float().mean()) if conf.any() else np.nan,policy_change=float((pp!=p0[:,q]).float().mean())))
                hp=norm_noise(h,hc,idx,2400000+seed+fi+q)
                with torch.inference_mode(): pp=cont(model,hp,1).argmax(-1)[:,q]
                rows.append(dict(seed=seed,factor=f,phase=q,patch='random_norm_full',conflict_n=int(conf.sum()),follow=float((pp[conf]==pc[conf,q]).float().mean()) if conf.any() else np.nan,policy_change=float((pp!=p0[:,q]).float().mean())))
    df=pd.DataFrame(rows); od=root/'analysis/V16-E.5'; df.to_csv(od/'relational_coalition_all.csv',index=False)
    sm=df.groupby(['factor','phase','patch'],as_index=False).agg(mean_follow=('follow','mean'),min_follow=('follow','min'),max_follow=('follow','max'),mean_policy_change=('policy_change','mean'),total_conflicts=('conflict_n','sum'))
    sm.to_csv(od/'relational_coalition_summary.csv',index=False); print(sm.to_string(index=False))
if __name__=='__main__': main()
