# Square-lattice experiments

`MODEL.py` implements the CUDA model. Run the experiment scripts from this directory. The repository README lists manuscript-scale commands and explicitly selects the four published ablation configurations.

Parameter-map generation uses `exp4_common.py` and the three `experiment_4_*_data.py` files. `run_experiment_4_all.py` runs these three planes sequentially. The plotting scripts read saved CSV/NPZ outputs; no raw outputs are included in this initial source upload.
