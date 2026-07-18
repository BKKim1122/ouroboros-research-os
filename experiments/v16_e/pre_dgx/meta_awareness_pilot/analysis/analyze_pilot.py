from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
r=pd.read_csv(ROOT/'analysis/pilot_eval_raw.csv')
f=r[(r.condition=='full_monitor')&(r.monitor_mode=='factual')]
b=r[(r.condition=='blind_cue_control')&(r.monitor_mode=='factual')]

# Drift vs recovery/policy
x=f[(f.report_gain==1)&(f.feedback_gain.isin([0,1]))].groupby(['drift_strength','feedback_gain'],as_index=False)[['recovery_rate','policy_acc','pre_focus_acc','post_focus_acc']].mean()
for metric,title,ylabel,name in [
    ('recovery_rate','Closed-loop recovery under attention drift','Recovery rate','v16_pilot_recovery_by_drift.png'),
    ('policy_acc','Policy performance under attention drift','Bit accuracy','v16_pilot_policy_by_drift.png')]:
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    for g,grp in x.groupby('feedback_gain'):
        ax.plot(grp.drift_strength,grp[metric],marker='o',label=f'feedback gain={g:g}')
    ax.set_xlabel('Drift strength'); ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(-0.03,1.03); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures'/name,dpi=180); plt.close(fig)

# Report gain dissociation
x=f[(f.drift_strength==4.5)&(f.feedback_gain==1)].groupby('report_gain',as_index=False)[['meta_stream_acc','meta_attention_mae','policy_acc','recovery_rate']].mean()
fig,ax=plt.subplots(figsize=(7.2,4.6))
ax.plot(x.report_gain,x.meta_stream_acc,marker='o',label='Meta attended-stream accuracy')
ax.plot(x.report_gain,x.policy_acc,marker='o',label='Policy accuracy')
ax.plot(x.report_gain,x.recovery_rate,marker='o',label='Recovery rate')
ax.set_xlabel('Report-access gain'); ax.set_ylabel('Metric'); ax.set_title('Report access dissociates from causal feedback use'); ax.set_ylim(-.03,1.03); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures'/'v16_pilot_report_gain_dissociation.png',dpi=180); plt.close(fig)

# Full versus blind at strong drift
x=pd.concat([f,b]).query("drift_strength==4.5 and report_gain==1 and feedback_gain==1").groupby('condition',as_index=False)[['meta_stream_acc','recovery_rate','policy_acc','world_acc']].mean()
fig,ax=plt.subplots(figsize=(7.2,4.6))
xx=range(len(x)); width=.18
for j,m in enumerate(['meta_stream_acc','recovery_rate','policy_acc','world_acc']):
    ax.bar([i+(j-1.5)*width for i in xx],x[m],width,label=m)
ax.set_xticks(list(xx)); ax.set_xticklabels(x.condition); ax.set_ylim(0,1.05); ax.set_title('Attention-state monitor versus cue-only control'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/'figures'/'v16_pilot_full_vs_blind.png',dpi=180); plt.close(fig)
