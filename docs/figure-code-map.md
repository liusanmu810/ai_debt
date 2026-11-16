# Figure-to-code map

| Paper figure | Simulation / analysis source | Saved outputs used by plotting |
|---|---|---|
| Fig. 1, environmental response | `lattice/experiment_1_environment_sweep.py` | Per-realization raw CSV and summary CSV; `plot_experiment_1_nature_style.py` redraws saved results |
| Fig. 2, proxy-update comparison | Environmental sweep with environment-dependent and fixed-rate settings | Two raw CSV files; `plot_fixed_memory_rate_comparison.py` calculates realization-level summaries and plots the comparison |
| Fig. 3, four-configuration ablation | `lattice/experiment_2_ablation.py`, with explicit four-model selection | Raw ablation CSV and summary; `plot_experiment_2_nature_style.py` supplies core panels |
| Fig. 4, model payoff | Same four-configuration ablation | `plot_payoff_trajectories` in the ablation plotting module |
| Fig. 5, representative trajectory | `lattice/experiment_3_temporal_dynamics.py` | Temporal CSV and config; plotted coordinate is `log10(t+1)` with original round labels |
| Fig. 6, spatial snapshots | `lattice/experiment_3_spatial_snapshots.py` | Two `snapshots_A0.6_*.npz` files; unchanged normalized fields and fixed color scales |
| Fig. 7, A / gate strength | `lattice/experiment_4_A_alpha_data.py`, `exp4_common.py` | `experiment_4_A_alpha_heatmap_data.npz` |
| Fig. 8, A / proxy weight | `lattice/experiment_4_A_wM_data.py`, `exp4_common.py` | `experiment_4_A_wM_heatmap_data.npz` |
| Fig. 9, proxy weight / gate strength | `lattice/experiment_4_wM_alpha_data.py`, `exp4_common.py` | `experiment_4_wM_alpha_heatmap_data.npz` |
| Fig. 10, network comparison | `topology_robustness/experiment2_scale_free.py`, `experiment2_small_world.py`, and `experiment2_core.py` | Topology-specific raw CSV, summary CSV, and metadata JSON |

All three parameter maps use `lattice/plot_experiment_4_heatmaps.py` for plotting. Their display smoothing is inherited from the manuscript workflow and does not change the saved numerical arrays.

The code includes a broader set of optional comparators than the paper. The README explicitly selects the four reported ablation configurations. Representative trajectories are single paired realizations; the main sweeps, ablation, maps, and topology comparisons use the replication settings recorded in their configurations.

## Scope of this source upload

The GPU model kernels are copied without scientific changes. Local cleanup renames two experiment entry points, removes machine-specific environment advice from error messages, clarifies a normalized-proxy metadata label, and incorporates the final paper's plotting typography. No simulation output is regenerated during packaging.

Six JSON files are copied from saved experiment metadata. The ablation protocol is documented by the explicit README command, and the three parameter-map protocols remain explicit constants in their generator files. The JSON files are provenance records rather than a new runtime configuration format.

Original raw CSV/NPZ files, manuscript PDFs/TeX, downloaded literature, temporary rendering images, and revision history are outside this initial source-only release.
