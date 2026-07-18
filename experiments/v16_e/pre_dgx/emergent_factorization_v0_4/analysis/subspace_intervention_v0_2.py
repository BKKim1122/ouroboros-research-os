from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('v16e_mod', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['v16e_mod'] = mod
    spec.loader.exec_module(mod)
    return mod


def fit_probe(x, y):
    sc = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=2000, solver='liblinear', random_state=0).fit(sc.transform(x), y)
    return sc, clf


def pred_probe(sc, clf, x):
    return clf.predict(sc.transform(x))


def orthonormal_random_subspace(d, k, rng, avoid=None):
    A = rng.normal(size=(d, k))
    if avoid is not None:
        A = A - avoid @ (avoid.T @ A)
    q, _ = np.linalg.qr(A)
    return q[:, :k]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--condition', required=True)
    ap.add_argument('--layer', type=int, default=2)
    ap.add_argument('--cal-n', type=int, default=5000)
    ap.add_argument('--eval-n', type=int, default=3000)
    args=ap.parse_args()
    root=Path(args.root)
    mod=load_module(root/'src'/'v16e_emergent_factorization.py')
    meta=json.loads((root/'raw'/'pilot'/f'{args.condition}_seed{args.seed}'/'metadata.json').read_text())
    cfg=mod.Config(**meta['config'])
    torch.set_num_threads(cfg.torch_threads)
    cfg.eval_n=args.eval_n
    model=mod.UnifiedTransformer(cfg,args.condition)
    model.load_state_dict(torch.load(root/'raw'/'pilot'/f'{args.condition}_seed{args.seed}'/'checkpoint.pt',map_location='cpu'))
    model.eval()

    # Separate calibration and evaluation sets.
    cal=mod.make_batch(args.cal_n,cfg,torch.Generator().manual_seed(810000+args.seed))
    ev=mod.make_batch(args.eval_n,cfg,torch.Generator().manual_seed(820000+args.seed))
    cal_layers,_=mod.collect_layers(model,cal)
    ev_layers,ev_o=mod.collect_layers(model,ev)
    zcal=cal_layers[args.layer]; zev=ev_layers[args.layer]

    # Probes for target transfer / off-target stability.
    probes={}
    for f in ['identity','beneficiary','concern']:
        probes[f]=fit_probe(zcal,cal[f].numpy())

    rows=[]
    rng=np.random.default_rng(830000+args.seed)
    for factor in ['identity','beneficiary','concern']:
        calcf=mod.make_batch(args.cal_n,cfg,torch.Generator().manual_seed(1),base=cal,flip_factor=factor)
        evcf=mod.make_batch(args.eval_n,cfg,torch.Generator().manual_seed(2),base=ev,flip_factor=factor)
        calcf_layers,_=mod.collect_layers(model,calcf)
        evcf_layers,evcf_o=mod.collect_layers(model,evcf)
        zcalcf=calcf_layers[args.layer]; zevcf=evcf_layers[args.layer]
        # Align all calibration differences to class 0 -> class 1.
        sign=np.where(cal[factor].numpy()==0,1.0,-1.0)[:,None]
        D=(zcalcf-zcal)*sign
        D=D-D.mean(0,keepdims=True)
        _,_,vt=np.linalg.svd(D,full_matrices=False)
        pred0=ev_o['policy'].argmax(-1); predcf=evcf_o['policy'].argmax(-1); conflict=pred0!=predcf
        for k in [1,2,4,8,16]:
            k=min(k,vt.shape[0])
            U=vt[:k].T
            delta=zevcf-zev
            zsw=zev+(delta@U)@U.T
            R=orthonormal_random_subspace(zev.shape[1],k,rng,avoid=U)
            zrd=zev+(delta@R)@R.T
            with torch.no_grad():
                osw=model.forward_from_layer_cls(ev,args.layer,torch.tensor(zsw,dtype=torch.float32))
                ord_=model.forward_from_layer_cls(ev,args.layer,torch.tensor(zrd,dtype=torch.float32))
            ps=osw['policy'].numpy().argmax(-1); pr=ord_['policy'].numpy().argmax(-1)
            target_sc,target_clf=probes[factor]
            target_transfer=float((pred_probe(target_sc,target_clf,zsw)==evcf[factor].numpy()).mean())
            off=[]
            for other in ['identity','beneficiary','concern']:
                if other==factor: continue
                sc,cl=probes[other]
                off.append(float((pred_probe(sc,cl,zsw)==pred_probe(sc,cl,zev)).mean()))
            world0=(ev_o['world']>0).astype(int); worldsw=(osw['world'].numpy()>0).astype(int)
            rows.append(dict(seed=args.seed,condition=args.condition,layer=args.layer,factor=factor,k=k,
                conflict_n=int(conflict.sum()),counterfactual_follow=float((ps[conflict]==predcf[conflict]).mean()),
                random_follow=float((pr[conflict]==predcf[conflict]).mean()),target_probe_transfer=target_transfer,
                offtarget_probe_stability=float(np.mean(off)),world_prediction_stability=float((world0==worldsw).mean()),
                policy_change_rate=float((ps!=pred0).mean()),cf_policy_change_rate=float(conflict.mean())))
    out=pd.DataFrame(rows)
    od=root/'raw'/'pilot'/f'{args.condition}_seed{args.seed}'
    out.to_csv(od/'subspace_interventions.csv',index=False)
    print(out.groupby(['factor','k'])[['counterfactual_follow','random_follow','target_probe_transfer','offtarget_probe_stability','world_prediction_stability']].mean().round(4).to_string())

if __name__=='__main__': main()
