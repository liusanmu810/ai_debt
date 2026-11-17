from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis'
D=ROOT
df=pd.read_csv(D/'new_experiments/formal/gate_raw.csv');assert len(df)==90
plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,3,figsize=(9.6,4.0),layout='constrained');colors=['#737373','#2171b5','#d97726'];coords=[]
for j,A in enumerate([.5,.6,.7]):
 for i,metric in enumerate(['mean_x','mean_D']):
  ax=axs[i,j]
  for k,mode in enumerate([0,1,3]):
   g=df[(df.screening_mode==mode)&(df.A_env==A)].sort_values(['seed','run_id']);xs=k+np.linspace(-.17,.17,len(g));ax.scatter(xs,g[metric],s=22,color=colors[k],alpha=.85,zorder=3);q=np.quantile(g[metric],[.25,.5,.75],method='linear');ax.vlines(k,q[0],q[2],lw=2.4,color='black');ax.plot([k-.13,k+.13],[q[1],q[1]],lw=2,color='black')
   coords.extend([{'A_env':A,'metric':metric,'screening_mode':mode,'run_id':int(r.run_id),'seed':int(r.seed),'plot_x':x,'value':r[metric]} for (_,r),x in zip(g.iterrows(),xs)])
  ax.axhline(.8 if i==0 else .2,color='.65',ls=':',lw=1);ax.set_ylim(-.025,1.025);ax.set_xlim(-.45,2.45);ax.set_xticks([0,1,2],['NG','RG','FG']);ax.tick_params(labelsize=11);ax.set_title(f'$A={A:g}$',fontsize=14);ax.set_ylabel(r'$\langle x\rangle$' if i==0 else r'$\langle D\rangle$',fontsize=15);ax.spines[['top','right']].set_visible(False)
for ext in ['pdf','png','svg']:fig.savefig(ROOT/'figures'/f'fig_gate_forms_seed_points.{ext}',dpi=300)
pd.DataFrame(coords).to_csv(OUT/'manuscript_gate_plot_coordinates.csv',index=False)
assert len(coords)==180
print('Saved 90 realizations x 2 metrics; vertical data unchanged.')
