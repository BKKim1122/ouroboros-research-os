from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F


def loadmod(path):
    spec=importlib.util.spec_from_file_location('v16e6',str(path)); m=importlib.util.module_from_spec(spec); sys.modules['v16e6']=m; spec.loader.exec_module(m); return m


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--chunk',type=int,default=200); ap.add_argument('--target',type=int,default=800)
    a=ap.parse_args(); root=Path(a.root); mod=loadmod(root/'src/v16e6_surface_transition_robustness.py'); cfg=mod.Config(steps=a.target,threads=1); mod.seed_all(a.seed); torch.set_num_threads(1)
    out=root/'raw/V16-E.6'/f'surface_transition_seed{a.seed}'; out.mkdir(parents=True,exist_ok=True); state_path=out/'training_state.pt'
    model=mod.Model(cfg); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4); g=torch.Generator()
    history=[]; start=0
    if state_path.exists():
        st=torch.load(state_path,map_location='cpu',weights_only=False); model.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); g.set_state(st['generator_state']); history=st['history']; start=int(st['step'])
    else:
        g.manual_seed(1600000+a.seed)
    end=min(start+a.chunk,a.target)
    for step in range(start,end):
        b=mod.make_batch(cfg.batch,cfg,g,'train'); o=model(b); ce_raw=F.cross_entropy(o['policy'].reshape(-1,4),b['actions'].reshape(-1),reduction='none').reshape(-1,4); phase_w_can=torch.tensor([1.45,1.45,0.45,1.45]); phase_w_q=phase_w_can[b['phase_order']]; ce=(ce_raw*phase_w_q).sum()/phase_w_q.sum(); lo=F.mse_loss(torch.sigmoid(o['outcome']),b['outcomes']); loss=ce+cfg.outcome_weight*lo; opt.zero_grad(); loss.backward(); opt.step()
        if step%25==0 or step==a.target-1: history.append(dict(step=step,loss=float(loss.detach()),action_acc=float((o['policy'].argmax(-1)==b['actions']).float().mean()),outcome_mse=float(lo.detach())))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),generator_state=g.get_state(),history=history,step=end),state_path)
    print(json.dumps({'seed':a.seed,'start':start,'end':end,'loss':float(loss.detach()),'action_acc':float((o['policy'].argmax(-1)==b['actions']).float().mean())}),flush=True)
    if end>=a.target:
        torch.save(model.state_dict(),out/'checkpoint.pt'); pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        (out/'metadata.json').write_text(json.dumps(dict(version='V16-E.6',seed=a.seed,config=mod.asdict(cfg),training_mode='chunked',finished_utc=mod.datetime.now(mod.timezone.utc).isoformat()),indent=2))
        rows=[]
        model.eval()
        for mode in ['id','encoding_ood','factor_ood','rule_ood']: rows.extend(mod.evaluate_mode(model,cfg,a.seed,mode))
        pd.DataFrame(rows).to_csv(out/'base_metrics.csv',index=False); print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__': main()
