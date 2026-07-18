from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import pandas as pd, torch

def load(path):
    spec=importlib.util.spec_from_file_location('v16d',str(path));m=importlib.util.module_from_spec(spec);sys.modules['v16d']=m;spec.loader.exec_module(m);return m

def cont(model,h,layer=1):
    for i in range(layer+1,model.cfg.layers+1):h=model.blocks[i-1](h)
    h=model.ln(h);z=h[:,6:10];return 5*model.policy(z)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[16140,16141,16142]);ap.add_argument('--n',type=int,default=250);a=ap.parse_args();root=Path(a.root);mod=load(root/'src_v0_4/v16e_dynamic.py')
    factors=['identity','beneficiary','concern'];role={'identity':2,'beneficiary':3,'concern':4};phase_specific={'identity':0,'beneficiary':1,'concern':2};rows=[]
    for seed in a.seeds:
        run=root/'raw/pilot_v0_4'/f'dynamic_behavior_only_seed{seed}';meta=json.loads((run/'metadata.json').read_text());cfg=mod.Config(**meta['config']);torch.set_num_threads(cfg.threads);model=mod.Model(cfg);model.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu'));model.eval()
        for fi,f in enumerate(factors):
            b=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1500000+seed+fi));cf=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1),base=b,flip=f)
            with torch.inference_mode():ob=model(b,True);oc=model(cf,True)
            p0=ob['policy'].argmax(-1);pc=oc['policy'].argmax(-1);h=ob['states'][1];hc=oc['states'][1]
            unrelated=role[factors[(fi+1)%3]]
            for q in [phase_specific[f],3]:
                conf=(p0[:,q]!=pc[:,q]).cpu().numpy()
                sets={'role_only':[role[f]],'phase_only':[6+q],'role_plus_phase':[role[f],6+q],'unrelated_plus_phase':[unrelated,6+q],'all_roles_plus_phase':[2,3,4,6+q]}
                for name,idx in sets.items():
                    hp=h.clone();hp[:,idx]=hc[:,idx]
                    with torch.inference_mode():pp=cont(model,hp,1).argmax(-1)[:,q].cpu().numpy()
                    rows.append(dict(seed=seed,target_factor=f,phase=q,patch=name,conflict_n=int(conf.sum()),follow=float((pp[conf]==pc[:,q].cpu().numpy()[conf]).mean()) if conf.any() else float('nan')))
    df=pd.DataFrame(rows);od=root/'analysis/p2b_v0_4_3seed';od.mkdir(parents=True,exist_ok=True);df.to_csv(od/'dynamic_coalition_layer1.csv',index=False);sm=df.groupby(['target_factor','phase','patch'],as_index=False).agg(mean_follow=('follow','mean'),min_follow=('follow','min'),max_follow=('follow','max'),total_conflicts=('conflict_n','sum'));sm.to_csv(od/'dynamic_coalition_layer1_summary.csv',index=False);print(sm.to_string(index=False))
if __name__=='__main__':main()
