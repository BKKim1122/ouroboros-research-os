from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

def load(path):
    spec=importlib.util.spec_from_file_location('v16e5',str(path)); m=importlib.util.module_from_spec(spec); sys.modules['v16e5']=m; spec.loader.exec_module(m); return m

def cont(model,h,layer):
    for i in range(layer+1,model.cfg.layers+1): h=model.blocks[i-1](h)
    h=model.ln(h); z=h[:,7:11]
    return dict(policy=5*model.policy(z),outcome=model.outcome(z))

def random_norm_patch(base,donor,idx,gen):
    out=base.clone(); delta=donor[:,idx]-base[:,idx]
    noise=torch.randn(delta.shape,generator=gen)
    nrm=delta.norm(dim=-1,keepdim=True); noise=noise/(noise.norm(dim=-1,keepdim=True)+1e-8)*nrm
    out[:,idx]=base[:,idx]+noise
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[16150,16151,16152]); ap.add_argument('--n',type=int,default=600); a=ap.parse_args(); root=Path(a.root); mod=load(root/'src_v16e_5/v16e5_relational_binding.py')
    factors=['identity','beneficiary','concern']; role={'identity':2,'beneficiary':3,'concern':4}; phase={'identity':0,'beneficiary':1,'concern':2}; rows=[]
    for seed in a.seeds:
        run=root/'raw/V16-E.5'/f'relational_binding_seed{seed}'; meta=json.loads((run/'metadata.json').read_text()); cfg=mod.Config(**meta['config']); torch.set_num_threads(cfg.threads); model=mod.Model(cfg); model.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu')); model.eval()
        for fi,f in enumerate(factors):
            b=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1800000+seed+fi),'id'); cf=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1),'id',base=b,flip=f)
            with torch.inference_mode(): ob=model(b,True); oc=model(cf,True)
            p0=ob['policy'].argmax(-1); pc=oc['policy'].argmax(-1); out0=torch.sigmoid(ob['outcome'])
            own=role[f]; cross=role[factors[(fi+1)%3]]
            for layer in range(cfg.layers+1):
                h=ob['states'][layer]; hc=oc['states'][layer]; qlist=[phase[f],3]
                for q in qlist:
                    sets={
                        'own_role':[own],
                        'phase_only':[7+q],
                        'own_plus_phase':[own,7+q],
                        'cross_plus_phase':[cross,7+q],
                        'entities_plus_phase':[0,1,7+q],
                    }
                    for name,idx in sets.items():
                        hp=h.clone(); hp[:,idx]=hc[:,idx]
                        with torch.inference_mode(): oo=cont(model,hp,layer)
                        pp=oo['policy'].argmax(-1); out=torch.sigmoid(oo['outcome'])
                        conf=(p0[:,q]!=pc[:,q])
                        rows.append(dict(seed=seed,factor=f,layer=layer,phase=q,patch=name,conflict_n=int(conf.sum()),follow=float((pp[conf,q]==pc[conf,q]).float().mean()) if conf.any() else np.nan,policy_change=float((pp[:,q]!=p0[:,q]).float().mean()),outcome_change_mae=float((out[:,q]-out0[:,q]).abs().mean())))
                    # Shuffled own+phase donor.
                    idx=[own,7+q]; perm=torch.randperm(a.n,generator=torch.Generator().manual_seed(1900000+seed+layer+q+fi))
                    hp=h.clone(); hp[:,idx]=hc[perm][:,idx]
                    with torch.inference_mode(): oo=cont(model,hp,layer)
                    pp=oo['policy'].argmax(-1); out=torch.sigmoid(oo['outcome']); conf=(p0[:,q]!=pc[:,q])
                    rows.append(dict(seed=seed,factor=f,layer=layer,phase=q,patch='shuffled_own_plus_phase',conflict_n=int(conf.sum()),follow=float((pp[conf,q]==pc[conf,q]).float().mean()) if conf.any() else np.nan,policy_change=float((pp[:,q]!=p0[:,q]).float().mean()),outcome_change_mae=float((out[:,q]-out0[:,q]).abs().mean())))
                    # Norm-matched random perturbation.
                    gen=torch.Generator().manual_seed(2000000+seed+layer+q+fi)
                    hp=random_norm_patch(h,hc,idx,gen)
                    with torch.inference_mode(): oo=cont(model,hp,layer)
                    pp=oo['policy'].argmax(-1); out=torch.sigmoid(oo['outcome']); conf=(p0[:,q]!=pc[:,q])
                    rows.append(dict(seed=seed,factor=f,layer=layer,phase=q,patch='random_norm_own_plus_phase',conflict_n=int(conf.sum()),follow=float((pp[conf,q]==pc[conf,q]).float().mean()) if conf.any() else np.nan,policy_change=float((pp[:,q]!=p0[:,q]).float().mean()),outcome_change_mae=float((out[:,q]-out0[:,q]).abs().mean())))
    df=pd.DataFrame(rows); od=root/'analysis/V16-E.5'; od.mkdir(parents=True,exist_ok=True); df.to_csv(od/'path_patch_all.csv',index=False)
    sm=df.groupby(['factor','layer','phase','patch'],as_index=False).agg(mean_follow=('follow','mean'),min_follow=('follow','min'),max_follow=('follow','max'),mean_policy_change=('policy_change','mean'),mean_outcome_change=('outcome_change_mae','mean'),total_conflicts=('conflict_n','sum'))
    sm.to_csv(od/'path_patch_summary.csv',index=False); print(sm.to_string(index=False))
if __name__=='__main__': main()
