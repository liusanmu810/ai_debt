"""Bounded gate comparison. Default validates only; --smoke or --formal is explicit."""
import argparse,hashlib,json,sys,time,platform,subprocess,importlib.metadata as metadata
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import core_driver
from MODEL import AITechnicalDebtMDAE_GPU
from numba import cuda

def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def environment():
 info={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{p:metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','numba']},'cuda_available':cuda.is_available()}
 try:info['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True);info['cuda_runtime']=cuda.runtime.get_version();info['cuda_driver']=cuda.cudadrv.driver.driver.get_version()
 except Exception as e:info['cuda_detail_error']=str(e)
 return info

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs/gate_comparison.json');g=ap.add_mutually_exclusive_group();g.add_argument('--smoke',action='store_true');g.add_argument('--formal',action='store_true');a=ap.parse_args();cfg=json.loads(a.config.read_text());out=ROOT/'new_experiments'/('smoke' if a.smoke else 'formal');out.mkdir(exist_ok=True,parents=True)
 env=environment();(out/'environment.json').write_text(json.dumps(env,indent=2)); hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'code/MODEL.py',ROOT/'code/core_driver.py',Path(__file__)]};(out/'code_hashes.json').write_text(json.dumps(hashes,indent=2));(out/'config.json').write_text(json.dumps(cfg,indent=2))
 assert cfg['base_params']['selection_rule']==0 and cfg['base_params']['w_memory']==.12 and cfg['base_params']['w_payoff']==.88
 assert cfg['seeds']==[cfg['seed_base']+i*10000 for i in range(cfg['runs'])]
 if not (a.smoke or a.formal):print('Configuration validated. No simulation requested.');return
 if not cuda.is_available():raise RuntimeError('CUDA unavailable; no simulations run')
 n,steps,window=(10,40,20) if a.smoke else (cfg['grid_size'],cfg['steps'],cfg['steady_window'])
 As=[.6] if a.smoke else cfg['A_values'];runs=1 if a.smoke else cfg['runs']; init=[]
 if a.smoke:
  for name,over in cfg['groups'].items():
   params=dict(cfg['base_params'],**over);m=AITechnicalDebtMDAE_GPU(grid_size=n,seed=cfg['seeds'][0],A_env=.6,**params)
   rec={'model':name,'screening_mode':m.screening_mode,**{k:digest(getattr(m,'d_'+k).copy_to_host()) for k in ['x','a','D','M']},'rng':digest(m.rng_states.copy_to_host())};init.append(rec)
  assert len({tuple(r[k] for k in ['x','a','D','M','rng']) for r in init})==1
  # Mode 0 and alpha=0 mode 3 must be numerically identical without changing RNG draws.
  q=dict(cfg['base_params'],alpha_conc=0.0);r0=core_driver.run_single_simulation('zero',.6,0,n,steps,window,cfg['seed_base'],dict(q,screening_mode=0));r3=core_driver.run_single_simulation('zero',.6,0,n,steps,window,cfg['seed_base'],dict(q,screening_mode=3));assert all(r0[k]==r3[k] for k in ['mean_x','mean_a','mean_D','mean_M','mean_payoff'])
  (out/'initialization_checks.json').write_text(json.dumps({'states':init,'alpha_zero_mode0_mode3_identical':True,'note':'Matched initialization and initial RNG states verified; identical stochastic inputs throughout divergent trajectories are not assumed.'},indent=2))
 path=out/'gate_raw.csv'; done=pd.read_csv(path) if path.exists() else pd.DataFrame(); records=done.to_dict('records')
 for A in As:
  for run in range(runs):
   for name,over in cfg['groups'].items():
    if any(r['model']==name and r['A_env']==A and r['run_id']==run for r in records):continue
    params=dict(cfg['base_params'],**over);record=core_driver.run_single_simulation(name,A,run,n,steps,window,cfg['seed_base'],params)
    record.update(screening_mode=over['screening_mode'],sample_interval=1,window_start=steps-window+1,window_end=steps,parameters_json=json.dumps(params,sort_keys=True),model_sha256=hashes['MODEL.py'],driver_sha256=hashes['core_driver.py'],experiment_kind='functional_test' if a.smoke else 'formal',initialization='antithetic uniform x/a shuffled, uniform low debt, M=0.5')
    records.append(record);pd.DataFrame(records).to_csv(path,index=False);print(f'{len(records)}/{len(As)*runs*len(cfg["groups"])} A={A} seed={record["seed"]} mode={over["screening_mode"]} x={record["mean_x"]:.6f} D={record["mean_D"]:.6f}',flush=True)
 (out/'completion.json').write_text(json.dumps({'complete':True,'records':len(records),'kind':'functional_test' if a.smoke else 'formal'},indent=2))
if __name__=='__main__':main()
