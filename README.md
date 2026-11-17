# AI-induced technical debt in networked software teams

Research code for **AI-Induced Technical Debt in Networked Software Teams: An Evolutionary Public-Goods Game Model**.

The model couples continuous AI-use and assurance strategies with persistent technical debt, an endogenous reliability proxy, and reliability-conditioned concentration control. This repository contains the simulation kernels, experiment drivers, plotting code, and saved experiment configurations.

## Contents

| Directory | Purpose |
|---|---|
| `lattice/` | Periodic square-lattice model, environmental sweeps, four-configuration ablation, temporal/spatial trajectories, parameter maps, and plotting |
| `topology_robustness/` | Scale-free and small-world model kernels and experiment drivers |
| `configs/` | Saved configurations and metadata corresponding to the manuscript experiments |
| `docs/` | Figure-to-code mapping and reproduction details |

The three core model files are `lattice/MODEL.py`, `topology_robustness/scale_free_model.py`, and `topology_robustness/small_world_model.py`.

## Reproducibility package

The versioned [research artifact](supplement_20261117/README.md) contains selected realization-level data, frozen execution sources, CPU analysis scripts, figures and audit records. Start with its CPU-only working-copy instructions; viewing results does not require a GPU. It covers the original four-configuration seed comparison, both Figure 2 update-rule datasets, bounded NG/RG/FG gate comparisons, and selected topology drift diagnostics. It does not archive all inputs for every manuscript figure. See the [current manuscript mapping](supplement_20261117/docs/manuscript_mapping.md). The directory date is an artifact identifier, not a separate manuscript appendix. The data and archived-code snapshot is publicly available at [commit `3af6ada6f63b`](https://github.com/liusanmu810/ai_debt/tree/3af6ada6f63bdbeac3825b6fa9b56e1cab38a436/supplement_20261117). Anonymous download and SHA256 verification confirmed all 100 files in that snapshot's manifest. Documentation-only updates preserve this manuscript-linked data version.

## Environment

Use Python with the packages in `requirements.txt`. Simulations use **Numba CUDA and an NVIDIA GPU**. The parameter-map driver checks for Numba 0.63 or later. Install a CUDA-capable environment appropriate for the GPU before launching simulations. Plotting saved outputs uses NumPy, pandas, SciPy, and Matplotlib and does not require running simulations.

```bash
python -m pip install -r requirements.txt
python -c "from numba import cuda; print(cuda.is_available())"
```

The requirements file lists dependencies and is not a locked simulation environment. GPU/driver and package versions should be recorded for each new run.

## Run the main experiments

Run the following from the `lattice` directory so that generated outputs and plotting defaults use the same relative paths.

```bash
cd lattice

# Environmental sweep, including the denser adoption interval.
python experiment_1_environment_sweep.py --grid-size 100 --steps 50000 --steady-window 500 --runs 10 --transition-runs 10 --seed-base 20260514 --A-values 0 0.1 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1

# Fixed proxy-update rate comparison.
python experiment_1_environment_sweep.py --grid-size 100 --steps 50000 --steady-window 500 --runs 10 --transition-runs 10 --seed-base 20260514 --memory-update-mode fixed --fixed-lambda 0.34 --A-values 0 0.1 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1

# The manuscript uses exactly these four configurations.
python experiment_2_ablation.py --models Payoff-only Payoff+MemoryAffinity Payoff+ConcentrationGate "Full Model" --grid-size 100 --steps 50000 --steady-window 500 --runs 10 --seed-base 20260514 --output-dir results_exp2_four_group_concentration_gate

# Representative temporal trajectory.
python experiment_3_temporal_dynamics.py --grid-size 100 --steps 50000 --A-env 0.6 --seed 20260514 --model-set core

# Representative spatial snapshots.
python experiment_3_spatial_snapshots.py --grid-size 200 --steps 100000 --A-env 0.6 --seed 20260514 --model-set core --snapshot-times 0 100 1000 20000 100000

# All three parameter planes. Each uses 50 x 50 parameter points and 10 realizations.
python run_experiment_4_all.py
```

The full parameter scans are substantial GPU jobs. `--help` describes command-line options; the three parameter-plane files also contain explicit configuration constants.

The ablation driver contains an additional `Payoff-Fermi` comparator. Its default model list is broader than the manuscript. Use the explicit four-model command above for the reported comparison.

For topology comparisons, run from `topology_robustness`:

```bash
python experiment2_scale_free.py --num-nodes 10000 --steps 50000 --steady-window 500 --runs 10 --seed-base 20260514 --scale-free-m 2
python experiment2_small_world.py --num-nodes 10000 --steps 50000 --steady-window 500 --runs 10 --seed-base 20260514 --small-world-k 4 --rewiring-probability 0.1
```

See each driver's `--help` for additional options and plotting-only mode. Saved configuration JSON files document the network settings used for the paper.

## Reproduction notes

- `configs/` files document saved runs; drivers do not automatically ingest them as universal configuration files.
- Some saved result-directory names retain `A0.8`; the manuscript temporal and spatial environmental setting is **A = 0.6**. The representative temporal CSV records seed 20261114, whereas its adjacent JSON and the illustrative command above specify 20260514. This conflict remains unresolved; that command is not a verified reconstruction of the plotted temporal trajectory. The new gate comparison does not resolve or replace its seed provenance.
- The reliability-conditioned gate uses the normalized score **M_prime**, with signal `rho * (1 - M_prime)`. Parameter-output metadata has been clarified to distinguish this from the unnormalized proxy; model calculations are unchanged.
- Parameter-map smoothing affects display only. Numerical comparisons use unsmoothed values.
- Current plotting sources include the manuscript's larger heatmap/topology labels and separated spatial colorbar endpoints.

## Data and manuscript files

The original source upload contains simulation code and run configurations. The reproducibility package now adds the selected realization-level CSVs described above; node-level fields, complete trajectories and full parameter-map arrays remain outside this artifact. Manuscript drafts, downloaded papers and revision history are not bundled.

The legacy `upload-manifest.json` records the original source-upload snapshot, not the current repository contents. The new artifact has its own `supplement_20261117/MANIFEST_SHA256.csv`; these manifests have different scopes.

## License

No distribution license has yet been specified by the authors. This upload does not add an inferred MIT, GPL, or other license.
