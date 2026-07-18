from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('v16m', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['v16m'] = mod
    spec.loader.exec_module(mod)
    return mod


def fit_probe(x, y):
    sc = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=1000, solver='liblinear').fit(sc.transform(x), y)
    return sc, clf


def bacc(y, p):
    return float(balanced_accuracy_score(y, p))


def continue_from(model, h, layer_idx):
    for i in range(layer_idx + 1, model.cfg.layers + 1):
        h = model.blocks[i - 1](h)
    h = model.ln(h)
    return model.heads(h[:, 0])


def state_arrays(model, batch):
    with torch.inference_mode():
        out = model(batch, True)
    return [s.cpu().numpy() for s in out['states']], {
        'policy': out['policy'].cpu().numpy(),
        'world': out['world'].cpu().numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[16130,16131,16132])
    ap.add_argument('--cal-n', type=int, default=600)
    ap.add_argument('--eval-n', type=int, default=600)
    ap.add_argument('--k', type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    mod = load_module(root/'src_v0_3/v16e_multicontext.py')
    factors = ['identity','beneficiary','concern']
    home = {'identity':0,'beneficiary':1,'concern':2}
    token_names = ['cls','entity_slot0','entity_slot1','identity_role','beneficiary_role','concern_context','query']
    patch_sets = {
        'cls':[0], 'entity_both':[1,2], 'identity_role':[3], 'beneficiary_role':[4],
        'concern_context':[5], 'query':[6], 'all_role_context':[3,4,5], 'all_noncls':[1,2,3,4,5,6]
    }
    probe_rows=[]; angle_rows=[]; patch_rows=[]

    for seed in args.seeds:
        run = root/'raw/pilot_v0_3'/f'behavior_only_seed{seed}'
        meta = json.loads((run/'metadata.json').read_text())
        cfg = mod.Config(**meta['config'])
        torch.set_num_threads(cfg.threads)
        model = mod.Model(cfg)
        model.load_state_dict(torch.load(run/'checkpoint.pt', map_location='cpu'))
        model.eval()
        n_states = cfg.layers + 2

        # Cache ordinary train/eval activations for all queries.
        cal_cache={}; eval_cache={}
        for q in range(4):
            cb=mod.make_batch(args.cal_n,cfg,torch.Generator().manual_seed(900000+seed+q),query=q)
            eb=mod.make_batch(args.eval_n,cfg,torch.Generator().manual_seed(910000+seed+q),query=q)
            cal_cache[q]=(cb,state_arrays(model,cb)[0])
            eval_cache[q]=(eb,state_arrays(model,eb)[0])

        for factor in factors:
            tq=home[factor]; cb,cstates=cal_cache[tq]
            for layer in range(n_states):
                for tok,tname in enumerate(token_names):
                    sc,clf=fit_probe(cstates[layer][:,tok],cb[factor].numpy())
                    for test_q in range(4):
                        eb,estates=eval_cache[test_q]
                        pred=clf.predict(sc.transform(estates[layer][:,tok]))
                        probe_rows.append(dict(seed=seed,factor=factor,train_query=tq,test_query=test_q,layer=layer,token=tname,bacc=bacc(eb[factor].numpy(),pred)))

        # Cache factor counterfactual activations at each factor's home query.
        paired={}
        for factor in factors:
            q=home[factor]
            b=mod.make_batch(args.cal_n,cfg,torch.Generator().manual_seed(800000+seed+q),query=q)
            cf=mod.make_batch(args.cal_n,cfg,torch.Generator().manual_seed(1),query=q,base=b,flip=factor)
            sb,_=state_arrays(model,b); scf,_=state_arrays(model,cf)
            paired[factor]=(b,sb,scf)

        for layer in range(n_states):
            for tok,tname in enumerate(token_names):
                bases={}
                for factor in factors:
                    b,sb,scf=paired[factor]
                    sign=np.where(b[factor].numpy()==0,1.0,-1.0)[:,None]
                    d=(scf[layer][:,tok]-sb[layer][:,tok])*sign
                    d-=d.mean(0,keepdims=True)
                    _,_,vt=np.linalg.svd(d,full_matrices=False)
                    bases[factor]=vt[:args.k].T
                for i,f1 in enumerate(factors):
                    for f2 in factors[i+1:]:
                        s=np.linalg.svd(bases[f1].T@bases[f2],compute_uv=False)
                        s=np.clip(s,0,1); angles=np.degrees(np.arccos(s))
                        angle_rows.append(dict(seed=seed,layer=layer,token=tname,factor_a=f1,factor_b=f2,k=args.k,
                                               min_angle_deg=float(angles.min()),mean_angle_deg=float(angles.mean()),
                                               max_cos=float(s.max()),projection_overlap=float(np.sum(s*s)/args.k)))

        # Path patching: home query and integrated query only.
        for target in factors:
            for q in sorted(set([home[target],3])):
                b=mod.make_batch(args.eval_n,cfg,torch.Generator().manual_seed(920000+seed+31*q+7*home[target]),query=q)
                cf=mod.make_batch(args.eval_n,cfg,torch.Generator().manual_seed(1),query=q,base=b,flip=target)
                with torch.inference_mode(): ob=model(b,True); oc=model(cf,True)
                p0=ob['policy'].argmax(-1).cpu().numpy(); pc=oc['policy'].argmax(-1).cpu().numpy(); conf=p0!=pc
                w0=(ob['world'].cpu().numpy()>0).astype(int)
                for layer in range(cfg.layers+1):
                    h=ob['states'][layer].clone(); hc=oc['states'][layer]
                    for pname,idxs in patch_sets.items():
                        hp=h.clone(); hp[:,idxs]=hc[:,idxs]
                        with torch.inference_mode(): out=continue_from(model,hp,layer)
                        pp=out['policy'].argmax(-1).cpu().numpy(); w=(out['world'].cpu().numpy()>0).astype(int)
                        patch_rows.append(dict(seed=seed,layer=layer,target_factor=target,query=q,source_patch=pname,
                                               conflict_n=int(conf.sum()),follow=float((pp[conf]==pc[conf]).mean()) if conf.any() else np.nan,
                                               policy_change=float((pp!=p0).mean()),world_stability=float((w==w0).mean())))

    outdir=root/'analysis/p2a_v0_3'; outdir.mkdir(parents=True,exist_ok=True)
    probes=pd.DataFrame(probe_rows); angles=pd.DataFrame(angle_rows); patches=pd.DataFrame(patch_rows)
    probes.to_csv(outdir/'layer_token_probe.csv',index=False)
    angles.to_csv(outdir/'factor_subspace_angles.csv',index=False)
    patches.to_csv(outdir/'layer_token_path_patching.csv',index=False)
    ps=(probes[probes.test_query==3].groupby(['factor','layer','token'],as_index=False).bacc.mean().sort_values(['factor','bacc'],ascending=[True,False]))
    ps.to_csv(outdir/'integrated_query_probe_summary.csv',index=False)
    ang=angles.groupby(['layer','token','factor_a','factor_b'],as_index=False).agg(projection_overlap=('projection_overlap','mean'),min_angle_deg=('min_angle_deg','mean'),mean_angle_deg=('mean_angle_deg','mean'))
    ang.to_csv(outdir/'factor_subspace_angle_summary.csv',index=False)
    pat=(patches[patches.conflict_n>0].groupby(['layer','target_factor','query','source_patch'],as_index=False)
         .agg(follow=('follow','mean'),policy_change=('policy_change','mean'),world_stability=('world_stability','mean'),conflict_n=('conflict_n','sum')))
    pat.to_csv(outdir/'path_patching_summary.csv',index=False)
    print('Wrote',outdir)
    print('\nTop integrated-query probe locations:')
    print(ps.groupby('factor').head(6).to_string(index=False))
    print('\nIntegrated-policy q3 path patches:')
    print(pat[pat.query==3].sort_values(['target_factor','layer','follow'],ascending=[True,True,False]).groupby(['target_factor','layer']).head(3).to_string(index=False))

if __name__=='__main__': main()
