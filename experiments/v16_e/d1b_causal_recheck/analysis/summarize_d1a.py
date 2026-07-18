from pathlib import Path
import json, pandas as pd, numpy as np

root = Path(__file__).resolve().parents[1]
work = root / "workspace"
rows=[]
for p in sorted((work/"raw"/"V16-E.C1").glob("compositional_seed*/base_metrics.csv")):
    df=pd.read_csv(p); rows.append(df)
if not rows:
    raise SystemExit("No D1a base_metrics.csv files found")
df=pd.concat(rows,ignore_index=True)
df.to_csv(root/"analysis"/"D1A_BASE_METRICS_ALL.csv",index=False)
summary=(df.groupby(["eval_mode","name"])["action_acc"]
         .agg(["mean","std","min","max","count"]).reset_index())
summary.to_csv(root/"analysis"/"D1A_BASE_METRICS_SUMMARY.csv",index=False)

# Compare with source C1 metrics if available beside the launcher root.
source = root.parent / "analysis" / "V16-E.C1" / "base_metrics_all.csv"
comparison=[]
if source.exists():
    c1=pd.read_csv(source)
    c1=c1[c1.rule_mode.eq("compositional")]
    for mode in ["id","encoding_ood","factor_ood","rule_ood"]:
        for name in ["continuity","allocation","protection","integrated"]:
            a=c1[(c1.eval_mode==mode)&(c1.name==name)].action_acc.mean()
            b=df[(df.eval_mode==mode)&(df.name==name)].action_acc.mean()
            comparison.append(dict(eval_mode=mode,name=name,c1_mean=a,d1a_mean=b,delta=b-a))
    pd.DataFrame(comparison).to_csv(root/"analysis"/"D1A_VS_C1.csv",index=False)

# Seed-level convergence from final history row.
conv=[]
for p in sorted((work/"raw"/"V16-E.C1").glob("compositional_seed*/history.csv")):
    seed=int(p.parent.name.split("seed")[-1])
    h=pd.read_csv(p)
    last=h.iloc[-1]
    conv.append(dict(seed=seed,final_step=int(last.step),final_action_acc=float(last.action_acc),final_loss=float(last.loss)))
pd.DataFrame(conv).to_csv(root/"analysis"/"D1A_CONVERGENCE.csv",index=False)

get=lambda mode,name: float(df[(df.eval_mode==mode)&(df.name==name)].action_acc.mean())
result={
  "version":"V16-E.D1a",
  "status":"exploratory_post_confirmatory",
  "n_seeds":int(df.seed.nunique()),
  "id_allocation":get("id","allocation"),
  "rule_ood_allocation":get("rule_ood","allocation"),
  "rule_ood_protection":get("rule_ood","protection"),
  "diagnostic_pattern":None
}
id_ok=result["id_allocation"]>=0.95
rule_ok=(result["rule_ood_allocation"]>=0.90 and result["rule_ood_protection"]>=0.90)
if id_ok and not rule_ok: pat="convergence_repaired_compositionality_remains"
elif id_ok and rule_ok: pat="uniform_extension_repairs_both"
elif not id_ok and not rule_ok: pat="optimization_and_compositionality_both_remain"
else: pat="mixed_unexpected_pattern"
result["diagnostic_pattern"]=pat
(root/"analysis"/"D1A_SUMMARY.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
