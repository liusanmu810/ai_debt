"""Validate archived inputs and complete formal output; report status separately."""
import json,hashlib,sys,importlib.metadata as metadata
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
P=Path(__file__).resolve().parents[1]
def main():
 checks={};core=pd.read_csv(P/'original_data/core/experiment_2_raw.csv');target=core[core.A_env.isin([.6,.7])];checks['core_target_rows']=len(target);assert len(target)==80
 assert not target.duplicated(['model','A_env','run_id','seed']).any();assert target.groupby(['model','A_env']).size().eq(10).all()
 for k in ['mean_x','mean_D']:assert np.isfinite(target[k]).all() and target[k].between(0,1).all()
 classified=(target.mean_x>.8)&(target.mean_D<.2);summary=pd.read_csv(P/'analysis/core_summary.csv');assert summary.joint_count.sum()==classified.sum();checks['core_joint_total']=int(classified.sum())
 for lab in ['dynamic','fixed']:
  df=pd.read_csv(P/f'original_data/{lab}/experiment_1_raw.csv');assert not df.duplicated(['A_env','run_id','seed']).any();assert df.groupby('A_env').size().eq(10).all();assert (df.seed==20260514+10000*df.run_id).all()
  for metric in ['tail_drift_x','tail_drift_D']:assert np.isfinite(df[metric]).all()
 checks['source_input_sha256_verified']=True
 for r in json.loads((P/'docs/provenance.json').read_text()):assert hashlib.sha256((P/r['archived']).read_bytes()).hexdigest()==r['sha256']
 smoke=P/'new_experiments/smoke/completion.json';checks['functional_test_passed']=smoke.exists() and json.loads(smoke.read_text())['records']==3
 sm=pd.read_csv(P/'new_experiments/smoke/gate_raw.csv');assert set(sm.screening_mode)=={0,1,3};assert sm.experiment_kind.eq('functional_test').all();assert sm.grid_size.eq(10).all() and sm.steps.eq(40).all() and sm.steady_window.eq(20).all()
 for col in ['mean_x','mean_D']:assert sm[col].between(0,1).all()
 init=json.loads((P/'new_experiments/smoke/initialization_checks.json').read_text());assert init['alpha_zero_mode0_mode3_identical'];assert len({tuple(r[k] for k in ['x','a','D','M','rng']) for r in init['states']})==1
 formal=P/'new_experiments/formal/gate_raw.csv';checks['formal_complete']=False
 if formal.exists():
  df=pd.read_csv(formal);cfg=json.loads((P/'configs/gate_comparison.json').read_text());hashes=json.loads((P/'new_experiments/formal/code_hashes.json').read_text());assert not df.duplicated(['model','A_env','run_id','seed']).any();assert df.experiment_kind.eq('formal').all()
  for name,sha in hashes.items():
   file=P/('analysis' if name=='run_gate_comparison.py' else 'code')/name;assert hashlib.sha256(file.read_bytes()).hexdigest()==sha
  for _,r in df.iterrows():
   assert r.grid_size==100 and r.steps==50000 and r.steady_window==500 and r.sample_interval==1
   assert r.seed==cfg['seeds'][int(r.run_id)] and r.A_env in cfg['A_values']
   expected=dict(cfg['base_params'],**cfg['groups'][r.model]);assert json.loads(r.parameters_json)==expected;assert r.screening_mode==expected['screening_mode'];assert r.model_sha256==hashes['MODEL.py']
   assert 0<=r.mean_x<=1 and 0<=r.mean_D<=1
  checks['formal_rows']=len(df);checks['formal_complete']=len(df)==90
  if len(df)==90:
   assert df.groupby(['model','A_env']).size().eq(10).all();paired=df[df.screening_mode==3].merge(df[df.screening_mode==1],on=['A_env','run_id','seed'],suffixes=('_f','_r'),validate='one_to_one');assert len(paired)==30
   ci=pd.read_csv(P/'analysis/gate_paired_summary.csv')
   for _,r in ci.iterrows():
    g=paired[paired.A_env==r.A_env];d=g[r.metric+'_f']-g[r.metric+'_r'];se=d.std(ddof=1)/np.sqrt(len(d));half=t.ppf(.975,len(d)-1)*se;assert np.isclose(r.difference_mean,d.mean());assert np.isclose(r.CI95_low,d.mean()-half);assert np.isclose(r.CI95_high,d.mean()+half)
   checks['formal_results_verified']=True
 (P/'docs/validation.json').write_text(json.dumps(checks,indent=2));(P/'docs/analysis_environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{k:metadata.version(k) for k in ['numpy','pandas','scipy','matplotlib']}},indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
