from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import pandas as pd, torch, numpy as np

def loadmod(path):
 spec=importlib.util.spec_from_file_location('m',str(path)); m=importlib.util.module_from_spec(spec); sys.modules['m']=m; spec.loader.exec_module(m); return m

def cont(model,h,layer):
 for i in range(layer+1,model.cfg.layers+1): h=model.blocks[i-1](h)
 h=model.ln(h); return 5*model.policy(h[:,7:11])

def pos(b,fi,q):
 r=2+(b['rel_roles']==fi).long().argmax(1); qj=(b['phase_order']==q).long().argmax(1); return r,7+qj,qj

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--n',type=int,default=400); a=ap.parse_args(); root=Path(a.root); mod=loadmod(root/'src/v16e6b_orthogonal_replication.py'); fac=['identity','beneficiary','concern']; ph=[0,1,2]; rows=[]
 for seed in range(16160,16165):
  run=root/'raw/V16-E.6B'/f'orthogonal_replication_seed{seed}'; meta=json.loads((run/'metadata.json').read_text()); cfg=mod.Config(**meta['config']); torch.set_num_threads(1); model=mod.Model(cfg); model.load_state_dict(torch.load(run/'checkpoint.pt',weights_only=True,map_location='cpu')); model.eval()
  for fi,f in enumerate(fac):
   b=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(3100000+seed+fi),'id'); cf=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1),'id',base=b,flip=f)
   with torch.inference_mode(): ob=model(b,True); oc=model(cf,True)
   for q in [ph[fi],3]:
    r,qp,qj=pos(b,fi,q); _,_,qjc=pos(cf,fi,q); row=torch.arange(a.n); p0=ob['policy'].argmax(-1)[row,qj]; pc=oc['policy'].argmax(-1)[row,qjc]; conf=p0!=pc
    for layer in [0,1,2]:
     h=ob['states'][layer].clone(); hc=oc['states'][layer]; h[row,r]=hc[row,r]; h[:,0:2]=hc[:,0:2]; h[row,qp]=hc[row,qp]
     with torch.inference_mode(): pp=cont(model,h,layer).argmax(-1)[row,qj]
     rows.append(dict(seed=seed,factor=f,phase=q,layer=layer,conflict_n=int(conf.sum()),follow=float((pp[conf]==pc[conf]).float().mean()) if conf.any() else np.nan))
 df=pd.DataFrame(rows); out=root/'analysis/V16-E.6'; df.to_csv(out/'layer_diagnostic_all.csv',index=False); sm=df.groupby(['factor','phase','layer'],as_index=False).agg(mean_follow=('follow','mean'),min_follow=('follow','min'),max_follow=('follow','max')); sm.to_csv(out/'layer_diagnostic_summary.csv',index=False); print(df[(df.factor=='concern') & (df.phase==3)].to_string(index=False)); print('\nSUMMARY\n',sm.to_string(index=False))
if __name__=='__main__': main()
