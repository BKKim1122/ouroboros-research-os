from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def loadmod(path):
    spec=importlib.util.spec_from_file_location('v16e6b',str(path)); m=importlib.util.module_from_spec(spec); sys.modules['v16e6b']=m; spec.loader.exec_module(m); return m


def continue_from(model,h,layer=1):
    for i in range(layer+1,model.cfg.layers+1):
        h=model.blocks[i-1](h)
    h=model.ln(h); z=h[:,7:11]
    return dict(policy=5*model.policy(z),outcome=model.outcome(z))


def dynamic_positions(batch,factor_idx,phase_idx):
    rel_j=(batch['rel_roles']==factor_idx).long().argmax(1)
    phase_j=(batch['phase_order']==phase_idx).long().argmax(1)
    return 2+rel_j, 7+phase_j, phase_j


def copy_dynamic(h,hc,rel_pos,phase_pos,parts):
    hp=h.clone(); row=torch.arange(h.shape[0])
    if 'role' in parts: hp[row,rel_pos]=hc[row,rel_pos]
    if 'entities' in parts: hp[:,0:2]=hc[:,0:2]
    if 'phase' in parts: hp[row,phase_pos]=hc[row,phase_pos]
    return hp


def copy_shuffled(h,hc,b,bc,rel_pos,phase_pos,factor_idx,phase_idx,perm):
    hp=h.clone(); row=torch.arange(h.shape[0]); donor=perm
    donor_rel=(bc['rel_roles'][donor]==factor_idx).long().argmax(1)+2
    donor_phase=(bc['phase_order'][donor]==phase_idx).long().argmax(1)+7
    hp[row,rel_pos]=hc[donor,donor_rel]
    hp[:,0:2]=hc[donor,0:2]
    hp[row,phase_pos]=hc[donor,donor_phase]
    return hp


def random_norm(h,hc,rel_pos,phase_pos,seed):
    hp=h.clone(); row=torch.arange(h.shape[0]); g=torch.Generator().manual_seed(seed)
    # dynamic role and phase positions
    for pos in [rel_pos,phase_pos]:
        delta=hc[row,pos]-h[row,pos]; noise=torch.randn(delta.shape,generator=g); noise=noise/(noise.norm(dim=-1,keepdim=True)+1e-8)*delta.norm(dim=-1,keepdim=True); hp[row,pos]=h[row,pos]+noise
    delta=hc[:,0:2]-h[:,0:2]; noise=torch.randn(delta.shape,generator=g); noise=noise/(noise.norm(dim=-1,keepdim=True)+1e-8)*delta.norm(dim=-1,keepdim=True); hp[:,0:2]=h[:,0:2]+noise
    return hp


def select_query(x,qj):
    return x[torch.arange(x.shape[0]),qj]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[16160,16161,16162,16163,16164]); ap.add_argument('--n',type=int,default=700)
    a=ap.parse_args(); root=Path(a.root); mod=loadmod(root/'src/v16e6b_orthogonal_replication.py'); factors=['identity','beneficiary','concern']; phase_of={'identity':0,'beneficiary':1,'concern':2}; rows=[]
    for seed in a.seeds:
        run=root/'raw/V16-E.6B'/f'orthogonal_replication_seed{seed}'; meta=json.loads((run/'metadata.json').read_text()); cfg=mod.Config(**meta['config']); torch.set_num_threads(1); model=mod.Model(cfg); model.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)); model.eval()
        for fi,f in enumerate(factors):
            b=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(2600000+seed+fi),'id'); cf=mod.make_batch(a.n,cfg,torch.Generator().manual_seed(1),'id',base=b,flip=f)
            with torch.inference_mode(): ob=model(b,True); oc=model(cf,True)
            h=ob['states'][1]; hc=oc['states'][1]
            for q in [phase_of[f],3]:
                rel_pos,phase_pos,qj=dynamic_positions(b,fi,q)
                _,_,qjc=dynamic_positions(cf,fi,q)
                p0=select_query(ob['policy'].argmax(-1),qj); pc=select_query(oc['policy'].argmax(-1),qjc)
                o0=select_query(torch.sigmoid(ob['outcome']),qj); ocp=select_query(torch.sigmoid(oc['outcome']),qjc)
                gtcf=cf['outcomes_canonical'][:,q]
                conf=p0!=pc
                outcome_diff=(cf['outcomes_canonical'][:,q]-b['outcomes_canonical'][:,q]).abs().sum(-1)>1e-6
                sets={
                    'role_only':('role',), 'entities_only':('entities',), 'phase_only':('phase',),
                    'role_phase':('role','phase'), 'entities_phase':('entities','phase'),
                    'full_relation_coalition':('role','entities','phase')
                }
                for name,parts in sets.items():
                    hp=copy_dynamic(h,hc,rel_pos,phase_pos,parts)
                    with torch.inference_mode(): oo=continue_from(model,hp,1)
                    pp=select_query(oo['policy'].argmax(-1),qj); op=select_query(torch.sigmoid(oo['outcome']),qj)
                    base_dist=(o0-gtcf).abs().mean(-1); patch_dist=(op-gtcf).abs().mean(-1)
                    med=1-patch_dist/(base_dist+1e-8)
                    rows.append(dict(seed=seed,factor=f,phase=q,patch=name,conflict_n=int(conf.sum()),policy_follow=float((pp[conf]==pc[conf]).float().mean()) if conf.any() else np.nan,policy_change=float((pp!=p0).float().mean()),outcome_diff_n=int(outcome_diff.sum()),outcome_closer_cf=float((patch_dist[outcome_diff]<base_dist[outcome_diff]).float().mean()) if outcome_diff.any() else np.nan,outcome_mediation=float(med[outcome_diff].mean()) if outcome_diff.any() else np.nan,model_output_shift=float((op-o0).abs().mean()),cf_model_gap=float((ocp-o0).abs().mean())))
                perm=torch.randperm(a.n,generator=torch.Generator().manual_seed(2700000+seed+fi+q)); hp=copy_shuffled(h,hc,b,cf,rel_pos,phase_pos,fi,q,perm)
                with torch.inference_mode(): oo=continue_from(model,hp,1)
                pp=select_query(oo['policy'].argmax(-1),qj); op=select_query(torch.sigmoid(oo['outcome']),qj); base_dist=(o0-gtcf).abs().mean(-1); patch_dist=(op-gtcf).abs().mean(-1); med=1-patch_dist/(base_dist+1e-8)
                rows.append(dict(seed=seed,factor=f,phase=q,patch='shuffled_full',conflict_n=int(conf.sum()),policy_follow=float((pp[conf]==pc[conf]).float().mean()) if conf.any() else np.nan,policy_change=float((pp!=p0).float().mean()),outcome_diff_n=int(outcome_diff.sum()),outcome_closer_cf=float((patch_dist[outcome_diff]<base_dist[outcome_diff]).float().mean()) if outcome_diff.any() else np.nan,outcome_mediation=float(med[outcome_diff].mean()) if outcome_diff.any() else np.nan,model_output_shift=float((op-o0).abs().mean()),cf_model_gap=float((ocp-o0).abs().mean())))
                hp=random_norm(h,hc,rel_pos,phase_pos,2800000+seed+fi+q)
                with torch.inference_mode(): oo=continue_from(model,hp,1)
                pp=select_query(oo['policy'].argmax(-1),qj); op=select_query(torch.sigmoid(oo['outcome']),qj); base_dist=(o0-gtcf).abs().mean(-1); patch_dist=(op-gtcf).abs().mean(-1); med=1-patch_dist/(base_dist+1e-8)
                rows.append(dict(seed=seed,factor=f,phase=q,patch='random_norm_full',conflict_n=int(conf.sum()),policy_follow=float((pp[conf]==pc[conf]).float().mean()) if conf.any() else np.nan,policy_change=float((pp!=p0).float().mean()),outcome_diff_n=int(outcome_diff.sum()),outcome_closer_cf=float((patch_dist[outcome_diff]<base_dist[outcome_diff]).float().mean()) if outcome_diff.any() else np.nan,outcome_mediation=float(med[outcome_diff].mean()) if outcome_diff.any() else np.nan,model_output_shift=float((op-o0).abs().mean()),cf_model_gap=float((ocp-o0).abs().mean())))
    df=pd.DataFrame(rows); out=root/'analysis/V16-E.6'; out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'coalition_mediation_all.csv',index=False)
    sm=df.groupby(['factor','phase','patch'],as_index=False).agg(mean_policy_follow=('policy_follow','mean'),min_policy_follow=('policy_follow','min'),max_policy_follow=('policy_follow','max'),mean_policy_change=('policy_change','mean'),mean_outcome_closer_cf=('outcome_closer_cf','mean'),mean_outcome_mediation=('outcome_mediation','mean'),mean_model_output_shift=('model_output_shift','mean'),total_conflicts=('conflict_n','sum'),total_outcome_diff=('outcome_diff_n','sum'))
    sm.to_csv(out/'coalition_mediation_summary.csv',index=False); print(sm.to_string(index=False))

if __name__=='__main__': main()
