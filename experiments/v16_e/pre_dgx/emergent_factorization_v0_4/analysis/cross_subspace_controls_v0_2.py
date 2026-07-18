from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def load_module(path: Path):
    spec=importlib.util.spec_from_file_location('v16e_mod2',str(path)); mod=importlib.util.module_from_spec(spec);sys.modules['v16e_mod2']=mod;spec.loader.exec_module(mod);return mod

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--condition',required=True);ap.add_argument('--layer',type=int,default=2);ap.add_argument('--k',type=int,default=4);ap.add_argument('--cal-n',type=int,default=1000);ap.add_argument('--eval-n',type=int,default=600);a=ap.parse_args()
    root=Path(a.root);mod=load_module(root/'src'/'v16e_emergent_factorization.py')
    meta=json.loads((root/'raw'/'pilot'/f'{a.condition}_seed{a.seed}'/'metadata.json').read_text());cfg=mod.Config(**meta['config']);torch.set_num_threads(cfg.torch_threads)
    model=mod.UnifiedTransformer(cfg,a.condition);model.load_state_dict(torch.load(root/'raw'/'pilot'/f'{a.condition}_seed{a.seed}'/'checkpoint.pt',map_location='cpu'));model.eval()
    factors=['identity','beneficiary','concern']
    cal=mod.make_batch(a.cal_n,cfg,torch.Generator().manual_seed(910000+a.seed));ev=mod.make_batch(a.eval_n,cfg,torch.Generator().manual_seed(920000+a.seed))
    zcal=mod.collect_layers(model,cal)[0][a.layer];zev,ev_o=mod.collect_layers(model,ev);zev=zev[a.layer]
    bases={}
    evcfs={}; evcf_layers={}; evcf_out={}
    for f in factors:
        calcf=mod.make_batch(a.cal_n,cfg,torch.Generator().manual_seed(1),base=cal,flip_factor=f)
        zcf=mod.collect_layers(model,calcf)[0][a.layer]
        sign=np.where(cal[f].numpy()==0,1.0,-1.0)[:,None];D=(zcf-zcal)*sign;D-=D.mean(0,keepdims=True)
        _,_,vt=np.linalg.svd(D,full_matrices=False);bases[f]=vt[:a.k].T
        ecf=mod.make_batch(a.eval_n,cfg,torch.Generator().manual_seed(2),base=ev,flip_factor=f);evcfs[f]=ecf
        ls,o=mod.collect_layers(model,ecf);evcf_layers[f]=ls[a.layer];evcf_out[f]=o
    rows=[]
    pred0=ev_o['policy'].argmax(-1);world0=(ev_o['world']>0).astype(int)
    rng=np.random.default_rng(930000+a.seed)
    for target in factors:
        predcf=evcf_out[target]['policy'].argmax(-1);conf=pred0!=predcf;delta=evcf_layers[target]-zev
        for source in factors+['random']:
            if source=='random':
                q,_=np.linalg.qr(rng.normal(size=(zev.shape[1],a.k)));U=q[:,:a.k]
            else: U=bases[source]
            zsw=zev+(delta@U)@U.T
            with torch.no_grad():o=model.forward_from_layer_cls(ev,a.layer,torch.tensor(zsw,dtype=torch.float32))
            ps=o['policy'].numpy().argmax(-1);world=(o['world'].numpy()>0).astype(int)
            rows.append(dict(seed=a.seed,condition=a.condition,target_factor=target,source_subspace=source,k=a.k,conflict_n=int(conf.sum()),counterfactual_follow=float((ps[conf]==predcf[conf]).mean()),policy_change_rate=float((ps!=pred0).mean()),world_prediction_stability=float((world==world0).mean())))
    out=pd.DataFrame(rows);out.to_csv(root/'raw'/'pilot'/f'{a.condition}_seed{a.seed}'/'cross_subspace_controls.csv',index=False);print(out.to_string(index=False))
if __name__=='__main__':main()
