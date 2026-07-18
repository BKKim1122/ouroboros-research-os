import argparse, json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); root=Path(a.root)
f=root/'workspace'/'analysis'/'V16-E.D1b'/'causal_metrics_all.csv'
df=pd.read_csv(f)
matched=df[~df['patch'].str.contains('random')].copy(); random=df[df['patch'].str.contains('random')].copy()
summary=matched.groupby(['context','kind','target'],as_index=False).agg(
 mean_policy_follow=('policy_follow_cf','mean'), min_policy_follow=('policy_follow_cf','min'),
 mean_non_target_change=('non_target_policy_change','mean'), mean_outcome_mediation=('outcome_mediation','mean'))
r=random.groupby(['context','kind','target'],as_index=False).agg(random_policy_follow=('policy_follow_cf','mean'))
summary=summary.merge(r,on=['context','kind','target'],how='left')
summary['matched_minus_random']=summary.mean_policy_follow-summary.random_policy_follow
out=root/'analysis'; out.mkdir(exist_ok=True)
summary.to_csv(out/'D1B_CAUSAL_SUMMARY.csv',index=False)
obj={'rows':len(df),'summary':summary.to_dict(orient='records')}
(out/'D1B_SUMMARY.json').write_text(json.dumps(obj,indent=2))
report=['# V16-E.D1b 자동 요약','', 'D1a 2600-step 체크포인트의 인과구조 재검증 결과입니다.','',summary.to_markdown(index=False)]
(root/'reports'/'V16-E.D1b_RESULTS_KO.md').write_text('\n'.join(report))
print(summary.to_string(index=False))
