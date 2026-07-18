from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

def load(path):
    spec=importlib.util.spec_from_file_location('v16e5',str(path)); m=importlib.util.module_from_spec(spec); sys.modules['v16e5']=m; spec.loader.exec_module(m); return m

def collect(model,mod,cfg,seed,factor,mode,n):
    phase={'identity':0,'beneficiary':1,'concern':2}[factor]; role={'identity':2,'beneficiary':3,'concern':4}[factor]
    b=mod.make_batch(n,cfg,torch.Generator().manual_seed(2100000+seed+len(factor)+sum(map(ord,mode))),mode)
    cf=mod.make_batch(n,cfg,torch.Generator().manual_seed(1),mode,base=b,flip=factor)
    with torch.inference_mode(): ob=model(b,True); oc=model(cf,True)
    same=(b['actions'][:,phase]==cf['actions'][:,phase])
    # Label relation target in visible-slot coordinates, not canonical A/B.
    def labels(batch):
        if factor=='identity': return mod.canonical_to_slot(batch['identity'],batch['entity_order'])
        if factor=='beneficiary': return mod.canonical_to_slot(batch['beneficiary'],batch['entity_order'])
        return batch['concern']
    y0=labels(b); y1=labels(cf); rows=[]
    for layer in range(cfg.layers+1):
        for token_name,token in [('role',role),('designated_phase',7+phase),('integrated_phase',10)]:
            X=torch.cat([ob['states'][layer][same,token],oc['states'][layer][same,token]],0).cpu().numpy(); y=torch.cat([y0[same],y1[same]],0).cpu().numpy()
            rows.append((layer,token_name,X,y,int(same.sum())))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[16150,16151,16152]); ap.add_argument('--n',type=int,default=3000); a=ap.parse_args(); root=Path(a.root); mod=load(root/'src_v16e_5/v16e5_relational_binding.py'); factors=['identity','beneficiary','concern']; out=[]
    for seed in a.seeds:
        run=root/'raw/V16-E.5'/f'relational_binding_seed{seed}'; meta=json.loads((run/'metadata.json').read_text()); cfg=mod.Config(**meta['config']); torch.set_num_threads(cfg.threads); model=mod.Model(cfg); model.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu')); model.eval()
        for f in factors:
            train=collect(model,mod,cfg,seed,f,'id',a.n); test_id=collect(model,mod,cfg,seed+100,f,'id',a.n); test_ood=collect(model,mod,cfg,seed+200,f,'symbol_ood',a.n)
            for (layer,tok,X,y,npairs),(_,_,Xi,yi,npairsi),(_,_,Xo,yo,npairso) in zip(train,test_id,test_ood):
                clf=LogisticRegression(max_iter=1000,class_weight='balanced').fit(X,y)
                out.append(dict(seed=seed,factor=f,layer=layer,token=tok,train_pairs=npairs,test_mode='id',test_pairs=npairsi,bacc=balanced_accuracy_score(yi,clf.predict(Xi))))
                out.append(dict(seed=seed,factor=f,layer=layer,token=tok,train_pairs=npairs,test_mode='symbol_ood',test_pairs=npairso,bacc=balanced_accuracy_score(yo,clf.predict(Xo))))
    df=pd.DataFrame(out); od=root/'analysis/V16-E.5'; od.mkdir(parents=True,exist_ok=True); df.to_csv(od/'action_matched_probe_all.csv',index=False)
    sm=df.groupby(['factor','layer','token','test_mode'],as_index=False).agg(mean_bacc=('bacc','mean'),min_bacc=('bacc','min'),max_bacc=('bacc','max'),mean_train_pairs=('train_pairs','mean'),mean_test_pairs=('test_pairs','mean'))
    sm.to_csv(od/'action_matched_probe_summary.csv',index=False); print(sm.to_string(index=False))
if __name__=='__main__': main()
