from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

def load(path):
    spec=importlib.util.spec_from_file_location('v16m',str(path));m=importlib.util.module_from_spec(spec);sys.modules['v16m']=m;spec.loader.exec_module(m);return m

def continue_from(model,h,layer_idx):
    # h is state after layer_idx (0=embedding). Run remaining blocks.
    for i in range(layer_idx+1, model.cfg.layers+1): h=model.blocks[i-1](h)
    h=model.ln(h);return model.heads(h[:,0])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seed',type=int,default=16130);ap.add_argument('--layer',type=int,default=1);ap.add_argument('--eval-n',type=int,default=1200);a=ap.parse_args();root=Path(a.root);mod=load(root/'src_v0_3/v16e_multicontext.py');meta=json.loads((root/'raw/pilot_v0_3'/f'behavior_only_seed{a.seed}'/'metadata.json').read_text());cfg=mod.Config(**meta['config']);torch.set_num_threads(cfg.threads);model=mod.Model(cfg);model.load_state_dict(torch.load(root/'raw/pilot_v0_3'/f'behavior_only_seed{a.seed}'/'checkpoint.pt',map_location='cpu'));model.eval()
    factors=['identity','beneficiary','concern'];token={'identity':3,'beneficiary':4,'concern':5};rows=[]
    for target in factors:
      for q in range(4):
        b=mod.make_batch(a.eval_n,cfg,torch.Generator().manual_seed(710000+a.seed+q),query=q);cf=mod.make_batch(a.eval_n,cfg,torch.Generator().manual_seed(1),query=q,base=b,flip=target)
        with torch.no_grad():ob=model(b,True);oc=model(cf,True)
        h=ob['states'][a.layer].clone();hc=oc['states'][a.layer];p0=ob['policy'].argmax(-1).numpy();pc=oc['policy'].argmax(-1).numpy();conf=p0!=pc;w0=(ob['world'].numpy()>0).astype(int)
        for source in factors+['cls','random_token']:
            hp=h.clone()
            if source in token:hp[:,token[source]]=hc[:,token[source]]
            elif source=='cls':hp[:,0]=hc[:,0]
            else:
                # Fixed unrelated entity token control.
                hp[:,1]=hc[:,1]
            with torch.no_grad():o=continue_from(model,hp,a.layer)
            pp=o['policy'].argmax(-1).numpy();w=(o['world'].numpy()>0).astype(int)
            rows.append(dict(seed=a.seed,layer=a.layer,target_factor=target,query=q,source_patch=source,conflict_n=int(conf.sum()),follow=float((pp[conf]==pc[conf]).mean()) if conf.any() else np.nan,policy_change=float((pp!=p0).mean()),world_stability=float((w==w0).mean())))
    df=pd.DataFrame(rows);df.to_csv(root/'raw/pilot_v0_3'/f'behavior_only_seed{a.seed}'/f'token_patch_layer{a.layer}.csv',index=False);print(df[df.conflict_n>0].groupby(['target_factor','source_patch'])['follow'].mean().round(4).to_string())
if __name__=='__main__':main()
