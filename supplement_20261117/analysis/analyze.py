"""CSV-only audit, seed summaries and drift figures; never starts simulation."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
P=Path(__file__).resolve().parents[1]
MAP={'Payoff-only':'Payoff-only diffusion','Payoff+MemoryAffinity':'Proxy-coupled diffusion','Payoff+ConcentrationGate':'Concentration gate','Full Model':'Full model'}
ORDER=list(MAP.values());COLORS=['#707070','#2171b5','#d97726','#238b45']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(df,name):df.to_csv(P/'analysis'/name,index=False)
def stats(df,groups,metrics):
 rows=[]
 for key,g in df.groupby(groups,sort=False):
  row=dict(zip(groups,key if isinstance(key,tuple) else [key]));row['n']=len(g)
  for m in metrics:
   v=g[m].to_numpy();q=np.quantile(v,[.25,.5,.75],method='linear');row.update({m+'_Q1':q[0],m+'_median':q[1],m+'_Q3':q[2]})
  if 'high_assurance_low_debt'in g:row.update(joint_count=int(g.high_assurance_low_debt.sum()),joint_fraction=g.high_assurance_low_debt.mean())
  rows.append(row)
 return pd.DataFrame(rows)
def quality(df,keys):
 issues=[]
 for k in ['mean_x','mean_D','A_env','run_id','seed']:
  if k not in df:issues.append({'kind':'missing_column','column':k});continue
  vals=pd.to_numeric(df[k],errors='coerce');bad=~np.isfinite(vals)
  if k in ['mean_x','mean_D','A_env']:bad|=(vals<0)|(vals>1)
  for i in df.index[bad]:issues.append({'row':int(i),'kind':'invalid_value','column':k,'value':str(df.loc[i,k])})
 if df.duplicated(keys).any():issues.append({'kind':'duplicate_identity','rows':df.index[df.duplicated(keys,keep=False)].tolist()})
 return issues

def plot_points(df,name,order,As,colors):
 coords=[];fig,axes=plt.subplots(2,len(As),figsize=(max(7,3.8*len(As)),7),squeeze=False,layout='constrained')
 for col,A in enumerate(As):
  for row,m in enumerate(['mean_x','mean_D']):
   ax=axes[row,col]
   for j,label in enumerate(order):
    g=df[(df.configuration==label)&np.isclose(df.A_env,A)].sort_values(['seed','run_id']);jit=np.linspace(-.17,.17,len(g)) if len(g)>1 else np.zeros(len(g));xs=j+jit
    ax.scatter(xs,g[m],s=22,alpha=.8,color=colors[j],zorder=3)
    if len(g):
     q=np.quantile(g[m],[.25,.5,.75],method='linear');ax.vlines(j,q[0],q[2],color='black',lw=3);ax.plot([j-.13,j+.13],[q[1],q[1]],color='black',lw=2)
    for (_,r),x in zip(g.iterrows(),xs):coords.append({'A_env':A,'configuration':label,'metric':m,'run_id':r.run_id,'seed':r.seed,'plot_x':x,'value':r[m]})
   ax.axhline(.8 if m=='mean_x' else .2,color='#888888',ls=':',lw=1);ax.set_ylim(-.025,1.025);ax.set_xticks(range(len(order)),order,rotation=23,ha='right',fontsize=8);ax.set_ylabel('Quality assurance, mean x' if m=='mean_x' else 'Technical debt, mean D');ax.set_title(f'A = {A:g}');ax.spines[['top','right']].set_visible(False)
 for ext in ['png','pdf','svg']:fig.savefig(P/'figures'/f'{name}.{ext}',dpi=220)
 plt.close(fig);save(pd.DataFrame(coords),name+'_plot_coordinates.csv')

def main():
 core=pd.read_csv(P/'original_data/core/experiment_2_raw.csv');issues=quality(core,['model','A_env','run_id','seed']);(P/'docs/core_quality_issues.json').write_text(json.dumps(issues,indent=2));assert not issues,'Resolve audit issues before statistics'
 expected={'Payoff-only':(1.,0.,0.),'Payoff+MemoryAffinity':(.88,.12,0.),'Payoff+ConcentrationGate':(1.,0.,.18),'Full Model':(.88,.12,.18)}
 for model,g in core.groupby('model'):
  assert model in expected
  for k,v in zip(['w_payoff','w_memory','alpha_conc'],expected[model]):assert np.allclose(g[k],v)
 assert all(core[k].eq(v).all() for k,v in [('grid_size',100),('steps',50000),('steady_window',500)])
 assert np.all(core.seed==20260514+10000*core.run_id)
 core['configuration']=core.model.map(MAP);core['high_assurance_low_debt']=(core.mean_x>.8)&(core.mean_D<.2)
 detail=core[core.A_env.isin([.6,.7])].copy();save(detail,'core_seed_details.csv');summ=stats(detail,['configuration','A_env'],['mean_x','mean_D']);save(summ,'core_summary.csv');plot_points(detail,'core_seed_points',ORDER,[.6,.7],COLORS)
 headers=['配置','A','有效实现数','保障中位数 [Q1, Q3]','债务中位数 [Q1, Q3]','联合达标数/有效实现数'];lines=['| '+' | '.join(headers)+' |','|'+'---|'*6]
 for _,r in summ.iterrows():
  cells=[r.configuration,f'{r.A_env:g}',str(r.n)]+[f'{r[m+"_median"]:.6f} [{r[m+"_Q1"]:.6f}, {r[m+"_Q3"]:.6f}]' for m in ['mean_x','mean_D']]+[f'{r.joint_count}/{r.n} ({r.joint_fraction:.0%})'];lines.append('| '+' | '.join(cells)+' |')
 (P/'analysis/core_summary.md').write_text('\n'.join(lines)+'\n\nQuantiles: NumPy method="linear" (Hyndman-Fan type 7). Strict joint criterion uses unrounded per-realization means.\n',encoding='utf8')
 drifts=[]
 for label in ['dynamic','fixed']:
  df=pd.read_csv(P/f'original_data/{label}/experiment_1_raw.csv');q=quality(df,['A_env','run_id','seed']);assert not q
  df['dataset']=label;df['window_start']=df.steps-df.steady_window+1;df['window_end']=df.steps;df['sample_interval']=1;df['half1_start']=49501;df['half1_end']=49750;df['half2_start']=49751;df['half2_end']=50000
  drifts.append(df)
 drift=pd.concat(drifts,ignore_index=True);save(drift,'drift_all_saved_sweep_rows.csv');focus=drift[drift.A_env.isin([.6,.7])].copy();save(focus,'drift_seed_details.csv');long=focus.melt(id_vars=['dataset','A_env','run_id','seed'],value_vars=['tail_drift_x','tail_drift_D'],var_name='metric',value_name='signed_drift');long['absolute_drift']=long.signed_drift.abs();ds=stats(long,['dataset','A_env','metric'],['absolute_drift']);extra=long.groupby(['dataset','A_env','metric']).agg(absolute_max=('absolute_drift','max'),increasing=('signed_drift',lambda x:int((x>0).sum())),decreasing=('signed_drift',lambda x:int((x<0).sum())),zero=('signed_drift',lambda x:int((x==0).sum()))).reset_index();ds=ds.merge(extra);save(ds,'drift_summary.csv')
 fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained');dc=[]
 for i,m in enumerate(['tail_drift_x','tail_drift_D']):
  for j,A in enumerate([.6,.7]):
   ax=axs[i,j]
   for k,label in enumerate(['dynamic','fixed']):
    g=focus[(focus.dataset==label)&(focus.A_env==A)].sort_values('seed');xs=k+np.linspace(-.13,.13,len(g));ax.scatter(xs,g[m],s=25);q=np.quantile(g[m],[.25,.5,.75]);ax.vlines(k,q[0],q[2],lw=3,color='black');ax.plot([k-.1,k+.1],[q[1],q[1]],color='black');dc.extend([{'dataset':label,'A_env':A,'metric':m,'seed':int(r.seed),'run_id':int(r.run_id),'plot_x':x,'value':r[m]} for (_,r),x in zip(g.iterrows(),xs)])
   ax.axhline(0,c='gray',ls=':');ax.set_xticks([0,1],['Environment-dependent','Fixed rate']);ax.set_title(f'A = {A:g}');ax.set_ylabel('Signed assurance drift' if i==0 else 'Signed debt drift');ax.spines[['top','right']].set_visible(False)
 for ext in ['png','pdf','svg']:fig.savefig(P/'figures'/f'tail_drift.{ext}',dpi=220)
 plt.close(fig);save(pd.DataFrame(dc),'drift_plot_coordinates.csv')
 settings={'quantiles':'linear / Hyndman-Fan type 7','joint_criterion':'mean_x > 0.8 and mean_D < 0.2, per realization, unrounded','jitter':'deterministic equally spaced horizontal offsets, sorted seed then run_id; vertical values unchanged','drift':'second-half minus first-half mean; 49501..49750 vs 49751..50000, sampled every round','formal_pairing':['A_env','run_id','seed'],'ci':'Student-t of per-seed differences, ddof=1; two-sided 95%'};(P/'configs/analysis_settings.json').write_text(json.dumps(settings,indent=2));(P/'configs/configuration_mapping.json').write_text(json.dumps(MAP,indent=2));pd.DataFrame({'run_id':range(10),'seed':[20260514+10000*i for i in range(10)]}).to_csv(P/'configs/seeds.csv',index=False)
 formal=P/'new_experiments/formal/gate_raw.csv'
 if formal.exists():
  new=pd.read_csv(formal);new['configuration']=new.model;new['high_assurance_low_debt']=(new.mean_x>.8)&(new.mean_D<.2);save(new,'gate_seed_details.csv');save(stats(new,['configuration','A_env'],['mean_x','mean_D']),'gate_summary.csv');pairs=new[new.screening_mode==3].merge(new[new.screening_mode==1],on=['A_env','run_id','seed'],suffixes=('_full','_reliability'),validate='one_to_one');paired=[];rr=[]
  for A,g in pairs.groupby('A_env'):
   for metric in ['mean_x','mean_D']:
    d=g[metric+'_full']-g[metric+'_reliability'];n=len(d);err=t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n) if n>1 else np.nan;rr.append({'A_env':A,'metric':metric,'n_pairs':n,'difference_mean':d.mean(),'CI95_low':d.mean()-err,'CI95_high':d.mean()+err})
    for idx,v in d.items():paired.append({'A_env':A,'run_id':int(g.loc[idx,'run_id']),'seed':int(g.loc[idx,'seed']),'metric':metric,'difference':v})
  save(pd.DataFrame(rr),'gate_paired_summary.csv');save(pd.DataFrame(paired),'gate_paired_details.csv')
  if len(new)==90:
   plot_points(new,'gate_seed_points',list(new.configuration.unique()),[.5,.6,.7],COLORS[:3]);fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained');pairsdf=pd.DataFrame(paired);difference_coords=[]
   for ax,m in zip(axs,['mean_x','mean_D']):
    for j,A in enumerate([.5,.6,.7]):
     g=pairsdf[(pairsdf.metric==m)&(pairsdf.A_env==A)].sort_values('seed');ax.scatter(j+np.linspace(-.13,.13,len(g)),g.difference);difference_coords.extend([{'A_env':A,'metric':m,'run_id':int(z.run_id),'seed':int(z.seed),'plot_x':xx,'difference':z.difference} for (_,z),xx in zip(g.iterrows(),j+np.linspace(-.13,.13,len(g)))]);r=next(r for r in rr if r['A_env']==A and r['metric']==m);ax.errorbar(j,r['difference_mean'],yerr=[[r['difference_mean']-r['CI95_low']],[r['CI95_high']-r['difference_mean']]],fmt='ks',capsize=5)
    ax.axhline(0,c='gray',ls=':');ax.set_xticks([0,1,2],['0.5','0.6','0.7']);ax.set_xlabel('A');ax.set_ylabel(r'$\Delta\langle x\rangle$ (full minus reliability-only)' if m=='mean_x' else r'$\Delta\langle D\rangle$ (full minus reliability-only)')
   for ext in ['png','pdf','svg']:fig.savefig(P/'figures'/f'gate_paired_differences.{ext}',dpi=220)
   plt.close(fig);save(pd.DataFrame(difference_coords),'gate_difference_plot_coordinates.csv')
 print(summ.to_string(index=False));print(ds.to_string(index=False))
if __name__=='__main__':main()
