# Missing or non-reusable information

1. Original core ablation has no tail_drift_x/D and no retained per-round500-point tail history for all10 seeds. It cannot supply core drift by reconstruction.
2. Old run-time code hashes/commits and exact Python/CUDA/package versions are not present. Current-source hashes do not retrospectively certify historical execution versions.
3. Old core CSV omits screening_mode,selection_rule and most baseline parameters; current driver definitions are consistent intended settings, not an archived execution contract. No historical rows reused in the new comparison.
4. Representative temporal CSV seed20261114 conflicts with nearby JSON seed20260514. It is not used for core drift.
5. Old gate-form CSV has Debt-only/rho_D fields and A=.8,10000 rounds; these observations cannot be interpreted as the current mode1 reliability-only condition.
6. Input folders named no_gate or refined may retain alpha=.18 and use10000-round horizons. Names alone are insufficient to assign a mechanism. They are inventoried, not merged.
7. The workspace is not a Git checkout. Remote HEAD was not independently fetched in this round, and no remote files were modified.

No target core condition is missing: four configurations x two A values x10 realizations =80 valid records. Both Figure2 update rules have10 realizations at each target A. Historical missing drift remains missing, not zero.
