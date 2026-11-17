from pathlib import Path
import json,shutil,hashlib
import pandas as pd,numpy as np
P=Path(__file__).resolve().parents[1];R=P.parent;rows=[];summ=[]
for topology in ['scale_free','small_world']:
 src=P/'original_data'/topology
 p=src/f'experiment_2_{topology}_raw.csv';df=pd.read_csv(p);df=df[df.A_env.isin([.6,.7])].copy();df['dataset']=topology
 assert not df.duplicated(['model','A_env','run_id','seed']).any()
 for _,r in df.iterrows():
  ts=np.arange(int(r.steps-r.steady_window+1),int(r.steps)+1,int(r.sample_interval));mid=len(ts)//2
  for metric in ['tail_drift_x','tail_drift_D']:rows.append({'dataset':topology,'model':r.model,'A_env':r.A_env,'run_id':int(r.run_id),'seed':int(r.seed),'metric':metric,'signed_drift':r[metric],'sample_count':len(ts),'half1_start':int(ts[0]),'half1_end':int(ts[mid-1]),'half2_start':int(ts[mid]),'half2_end':int(ts[-1]),'sample_interval':int(r.sample_interval)})
d=pd.DataFrame(rows);d.to_csv(P/'analysis/topology_drift_seed_details.csv',index=False)
for key,g in d.groupby(['dataset','model','A_env','metric']):
 v=g.signed_drift.abs();q=v.quantile([.25,.5,.75],interpolation='linear');summ.append(dict(zip(['dataset','model','A_env','metric'],key),n=len(g),absolute_Q1=q.loc[.25],absolute_median=q.loc[.5],absolute_Q3=q.loc[.75],absolute_max=v.max(),increasing=int((g.signed_drift>0).sum()),decreasing=int((g.signed_drift<0).sum()),zero=int((g.signed_drift==0).sum())))
pd.DataFrame(summ).to_csv(P/'analysis/topology_drift_summary.csv',index=False)
print(pd.DataFrame(summ).sort_values('absolute_max',ascending=False).head(8).to_string(index=False))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import MAP,ORDER
coords=[]
for topology in ['scale_free','small_world']:
 fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for i,metric in enumerate(['tail_drift_x','tail_drift_D']):
  for j,A in enumerate([.6,.7]):
   ax=axs[i,j]
   for k,(raw,label) in enumerate(MAP.items()):
    g=d[(d.dataset==topology)&(d.model==raw)&(d.A_env==A)&(d.metric==metric)].sort_values('seed');xs=k+np.linspace(-.15,.15,len(g));ax.scatter(xs,g.signed_drift,s=20);q=g.signed_drift.quantile([.25,.5,.75]);ax.vlines(k,q.loc[.25],q.loc[.75],color='black',lw=3);ax.plot([k-.1,k+.1],[q.loc[.5],q.loc[.5]],color='black')
    coords.extend([dict(dataset=topology,metric=metric,A_env=A,seed=int(r.seed),run_id=int(r.run_id),plot_x=x,value=r.signed_drift) for (_,r),x in zip(g.iterrows(),xs)])
   ax.axhline(0,color='gray',ls=':');ax.set_xticks(range(4),ORDER,rotation=25,ha='right',fontsize=8);ax.set_ylabel(metric);ax.set_title(f'{topology}, A = {A:g}')
 for ext in ['png','pdf','svg']:fig.savefig(P/'figures'/f'{topology}_drift.{ext}',dpi=220)
 plt.close(fig)
pd.DataFrame(coords).to_csv(P/'analysis/topology_drift_plot_coordinates.csv',index=False)
