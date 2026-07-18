from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

def load(path):
    spec=importlib.util.spec_from_file_location('v16d',str(path));m=importlib.util.module_from_spec(spec);sys.modules['v16d']=m;spec.loader.exec_module(m);return m

def continue_from(model,h,layer):
    for i in range(layer+1,model.cfg.layers+1):h=model.blocks[i-1](h)
    h=model.ln(h);z=h[:,6:10];return dict(policy=5*model.policy(z),outcome=model.outcome(z))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seed',type=int,default=16140);ap.add_argument('--eval-n',type=int,default=500);a=ap.parse_args();root=Path(a.root);mod=load(root/'src_v0_4/v16e_dynamic.py');run=root/'raw/pilot_v0_4'/f'dynamic_behavior_only_seed{a.seed}';meta=json.loads((run/'metadata.json').read_text());cfg=mod.Config(**meta['config']);torch.set_num_threads(cfg.threads);model=mod.Model(cfg);model.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu'));model.eval()
    factors=['identity','beneficiary','concern'];phases=['continuity','allocation','protection','integrated'];patches={'identity_role':[2],'beneficiary_role':[3],'concern_context':[4],'entities':[0,1],'phase0':[6],'phase1':[7],'phase2':[8],'phase3':[9],'all_phases':[6,7,8,9]};rows=[]
    for target in factors:
        b=mod.make_batch(a.eval_n,cfg,torch.Generator().manual_seed(1400000+a.seed+len(target)));cf=mod.make_batch(a.eval_n,cfg,torch.Generator().manual_seed(1),base=b,flip=target)
        with torch.inference_mode():ob=model(b,True);oc=model(cf,True)
        p0=ob['policy'].argmax(-1).cpu().numpy();pc=oc['policy'].argmax(-1).cpu().numpy();out0=torch.sigmoid(ob['outcome']).cpu().numpy()
        for layer in range(cfg.layers+1):
            h=ob['states'][layer].clone();hc=oc['states'][layer]
            for pname,idx in patches.items():
                hp=h.clone();hp[:,idx]=hc[:,idx]
                with torch.inference_mode():oo=continue_from(model,hp,layer)
                pp=oo['policy'].argmax(-1).cpu().numpy();out=torch.sigmoid(oo['outcome']).cpu().numpy()
                for q,qname in enumerate(phases):
                    conf=p0[:,q]!=pc[:,q]
                    rows.append(dict(seed=a.seed,target_factor=target,layer=layer,source_patch=pname,phase=q,phase_name=qname,conflict_n=int(conf.sum()),follow=float((pp[conf,q]==pc[conf,q]).mean()) if conf.any() else np.nan,policy_change=float((pp[:,q]!=p0[:,q]).mean()),outcome_change_mae=float(np.abs(out[:,q]-out0[:,q]).mean())))
    df=pd.DataFrame(rows);df.to_csv(run/'dynamic_path_patch.csv',index=False)
    s=df[df.conflict_n>0].groupby(['target_factor','layer','source_patch','phase_name'],as_index=False).agg(follow=('follow','mean'),policy_change=('policy_change','mean'),outcome_change_mae=('outcome_change_mae','mean'),conflict_n=('conflict_n','sum'))
    s.to_csv(run/'dynamic_path_patch_summary.csv',index=False)
    print(s.sort_values(['target_factor','layer','phase_name','follow'],ascending=[True,True,True,False]).to_string(index=False))
if __name__=='__main__':main()
