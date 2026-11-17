# Data audit and provenance

This public research artifact contains selected realization-level evidence from the completed local archive. The original ZIP and unchanged extraction are retained privately. Public changes are documented separately and do not claim byte identity with that ZIP. The repository baseline is recorded in `source_repository.json`; adding this package does not replace the existing simulation directories.

## Included datasets

| Alias | Relative CSV | Rows | Target scope |
|---|---|---:|---|
| core | `original_data/core/experiment_2_raw.csv` | 440 | Original four configurations, 11 A values, 10 realizations per condition; target A=0.6/0.7 has 80 records |
| dynamic | `original_data/dynamic/experiment_1_raw.csv` | 160 | Full model, environment-dependent proxy updating, 16 A values ×10 |
| fixed | `original_data/fixed/experiment_1_raw.csv` | 160 | Full model, fixed lambda_M=0.34, 16 A values ×10 |
| scale_free | `original_data/scale_free/experiment_2_scale_free_raw.csv` | 440 | Original four configurations on 10,000-node scale-free networks; metadata retained |
| small_world | `original_data/small_world/experiment_2_small_world_raw.csv` | 440 | Original four configurations on 10,000-node small-world networks; metadata retained |
| formal | `new_experiments/formal/gate_raw.csv` | 90 | NG/RG/FG × A=0.5/0.6/0.7 ×10 seeds |
| smoke | `new_experiments/smoke/gate_raw.csv` | 3 | 10×10,40-round functional checks only; excluded from formal statistics |

All five historical CSVs and the formal/smoke numerical CSVs are byte-preserved. Metadata keep scientific parameters and identities; machine-specific paths and runtime snapshots were redacted separately. `provenance.json` uses forward-slash relative paths; `source-workspace/` labels identify original locations without being required runtime paths. The wider historical workspace inventory remains provenance context, not additional observations merged into the included datasets.

The 80-row target core subset has 10 realizations in each configuration/A condition, no duplicate model/A/run/seed identities, no missing/non-numeric endpoints, no endpoint outside [0,1], and no mixed grid/horizon/window/weight parameters. The Figure 2 target subset has 20 observations per update rule. All seeds follow seed=20260514+10000 run_id. Full-model core and dynamic-sweep endpoints match exactly for overlapping identities; they are not pooled as additional independent samples.

## Meaning of fields and groups

`A_env` is external AI environmental intensity, not elapsed time. `run_id` and `seed` identify one independent realization. `mean_x` and `mean_D` are that realization's tail-window averages of population means. `std_*_time` measures within-realization time variation. Nodes, repeated exports and time observations do not increase n.

Original configurations map as follows: Payoff-only uses w_P=1,w_M=0,alpha=0; Payoff+MemoryAffinity uses .88,.12,0; Payoff+ConcentrationGate uses 1,0,.18; Full Model uses .88,.12,.18. Display names are in `configs/configuration_mapping.json`. Current reference driver definitions include deterministic maximum-stimulus selection; the historical core CSV itself lacks screening_mode,selection_rule, most baseline parameters and execution hashes. A current reference file cannot retrospectively certify its historical execution version.

For new formal records, complete merged parameters and execution hashes are saved per row. Mode0 has no gate, mode1 discounts by 1−M_prime, and mode3 by rho(1−M_prime). All three keep proxy coupling. The frozen model passes screening_mode to the kernel; the wrapper uses the original core driver's simulation/averaging function. No archived model file was replaced by the repository HEAD.

## Tail drift and source gaps

Sweep drift compares the final window's second 250 round observations with its first 250: rounds49751–50000 minus49501–49750. Topology drift compares the second 25 with first 25 samples at interval10:49501,…,49991. These definitions are not pooled. Signed changes and absolute-magnitude summaries are both retained; no convergence threshold is fitted.

Historical core drift and dense tail histories are missing. Its window means and standard deviations cannot reconstruct half-window means. A separate representative temporal CSV identifies seed20261114 for both trajectories, while adjacent JSON specifies20260514; historical logs do not resolve this conflict. It is not backfilled using formal gate seeds. Legacy gate-form outputs with Debt-only/rho_D labels and different horizons are not treated as the current reliability-only experiment.

Historical run-time package versions/commits not recorded in old datasets remain unknown. New formal execution environment and source hashes retain their historical meaning after path redaction. The CPU review is recorded separately in `cpu_review.json`; it is not a new GPU simulation or a replacement execution environment.
