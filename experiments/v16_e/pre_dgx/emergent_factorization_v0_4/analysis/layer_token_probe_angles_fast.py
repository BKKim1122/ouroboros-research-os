from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

def load_module(path):
    spec=importlib.util.spec_from_file_location('v16m',str(path));m=importlib.util.module_from_spec(spec);sys.modules['v16m']=m;spec.loader.exec_module(m);return m

def collect(model,b):
    with torch.inference_mode():o=model(b,True)
    return [s.cpu().numpy() for s in o['states']]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[16130,16131,16132]);ap.add_argument('--n',type=int,default=400);ap.add_argument('--k',type=int,default=4);a=ap.parse_args()
    root=Path(a.root);mod=load_module(root/'src_v0_3/v16e_multicontext.py');factors=['identity','beneficiary','concern'];home={'identity':0,'beneficiary':1,'concern':2};tn=['cls','entity0','entity1','identity_role','beneficiary_role','concern_context','query'];pr=[];ar=[]
    for seed in a.seeds:
        run=root/'raw/pilot_v0_3'/f'behavior_only_seed{seed}';meta=json.loads((run/'metadata.json').read_text());cfg=mod.Config(**meta['config']);torch.set_num_threads(cfg.threads);model=mod.Model(cfg);model.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu'));model.eval();ns=cfg.layers+2
        cache={}
        for q in range(4):
            b=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1000000+seed+q),query=q);cache[q]=(b,collect(model,b))
        for f in factors:
            q=home[f];b,st=cache[q]
            for li in range(ns):
                for ti,name in enumerate(tn):
                    sc=StandardScaler().fit(st[li][:,ti]);cl=RidgeClassifier(alpha=1.0).fit(sc.transform(st[li][:,ti]),b[f].numpy())
                    for tq in sorted(set([q,3])):
                        tb,tst=cache[tq];p=cl.predict(sc.transform(tst[li][:,ti]));pr.append(dict(seed=seed,factor=f,train_query=q,test_query=tq,layer=li,token=name,bacc=float(balanced_accuracy_score(tb[f].numpy(),p))))
        pair={}
        for f in factors:
            q=home[f];b=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1100000+seed+q),query=q);cf=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1),query=q,base=b,flip=f);pair[f]=(b,collect(model,b),collect(model,cf))
        for li in range(ns):
            for ti,name in enumerate(tn):
                U={}
                for f in factors:
                    b,s,scf=pair[f];sg=np.where(b[f].numpy()==0,1.,-1.)[:,None];d=(scf[li][:,ti]-s[li][:,ti])*sg;d-=d.mean(0);_,_,vt=np.linalg.svd(d,full_matrices=False);U[f]=vt[:a.k].T
                for i,f1 in enumerate(factors):
                    for f2 in factors[i+1:]:
                        sv=np.clip(np.linalg.svd(U[f1].T@U[f2],compute_uv=False),0,1);ang=np.degrees(np.arccos(sv));ar.append(dict(seed=seed,layer=li,token=name,factor_a=f1,factor_b=f2,projection_overlap=float(np.sum(sv*sv)/a.k),min_angle_deg=float(ang.min()),mean_angle_deg=float(ang.mean())))
    od=root/'analysis/p2a_v0_3';od.mkdir(parents=True,exist_ok=True);pdf=pd.DataFrame(pr);adf=pd.DataFrame(ar);pdf.to_csv(od/'layer_token_probe.csv',index=False);adf.to_csv(od/'factor_subspace_angles.csv',index=False)
    ps=pdf[pdf.test_query==3].groupby(['factor','layer','token'],as_index=False).bacc.mean().sort_values(['factor','bacc'],ascending=[True,False]);ps.to_csv(od/'integrated_query_probe_summary.csv',index=False)
    sm=adf.groupby(['layer','token','factor_a','factor_b'],as_index=False).mean(numeric_only=True);sm.to_csv(od/'factor_subspace_angle_summary.csv',index=False)
    print(ps.groupby('factor').head(8).to_string(index=False));print('\nLOWEST OVERLAP LOCATIONS');print(sm.sort_values('projection_overlap').head(20).to_string(index=False))
if __name__=='__main__':main()
