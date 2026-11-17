# Reproducibility package: realization-level outcomes and gate comparisons

This research artifact supports selected results in **AI-Induced Technical Debt in Networked Software Teams: An Evolutionary Public-Goods Game Model**. Its dated directory name is a version identifier; it does not denote a separate manuscript appendix. Data and archived execution sources are kept together with their relative paths. The package is publicly available in [manuscript-linked commit `3af6ada6f63b`](https://github.com/liusanmu810/ai_debt/tree/3af6ada6f63bdbeac3825b6fa9b56e1cab38a436/supplement_20261117). All 100 manifest-listed files in that immutable snapshot were verified through anonymous HTTPS access and matching SHA256 values. Later documentation-only commits do not change that cited data snapshot.

## Scope and results

The package includes five historical datasets (`original_data/core`, `dynamic`, `fixed`, `scale_free`, `small_world`), the completed bounded gate comparison, and clearly separated historical functional tests. It supports manuscript Table 3, Table 4, the gate-form seed figure, Figure 2 update-rule diagnostics and the reported topology drift diagnostics. It does **not** supply every input for the paper's parameter maps, complete trajectories or node-level spatial fields.

The original four-configuration comparison contains payoff-only diffusion, proxy-coupled diffusion, concentration gate and full model. In the new comparison:

- **NG** retains proxy coupling without a gate; it is not payoff-only diffusion.
- **RG** applies reliability-only gating; it is not the original concentration-gate configuration, whose direct proxy weight is zero.
- **FG** applies reliability-conditioned gating with true local concentration.

The formal comparison has **3 groups × 3 A values × 10 matched seeds = 90 records**, using A=0.5, 0.6 and 0.7, a periodic 100×100 lattice, 50,000 rounds, the final 500 rounds sampled every round, w_M=0.12 and w_P=0.88. The gated groups share nominal alpha_conc=0.18. Both RG and FG meet the joint criterion in all ten realizations at A=0.5 and 0.6; NG does so in none. At A=0.7 all three groups have zero joint successes. All six FG-minus-RG confidence intervals include zero. This comparison has not established an additional benefit of local concentration at these settings and is not an equivalence test. Equal nominal alpha changes penalty strength as well as form when rho is included.

`new_experiments/smoke/` contains a **10×10, 40-round functional test**, with a final-20-round summary, three mode records and initial-state/RNG checks. Those observations never contribute to formal n. Repeating the same condition and seed does not add a new independent sample.

## CPU-only reproduction from saved CSVs

The analysis writes tables, figures and some metadata. **Never run the following scripts directly inside the frozen artifact.** Create a new, untracked sibling work copy first. Run these commands from the repository root; copytree refuses to replace an existing destination:

```sh
python -c "from pathlib import Path; import shutil; shutil.copytree(Path('supplement_20261117'), Path('../ai_debt_cpu_review/supplement_20261117'))"
cd ../ai_debt_cpu_review/supplement_20261117
python analysis/analyze.py
python analysis/topology_drift.py
python analysis/validate_package.py
python analysis/plot_manuscript_gate.py
```

CPU analysis requires NumPy, pandas, SciPy and Matplotlib; it does not import the GPU model. Tables and coordinates are written under `analysis/`, figures under `figures/`. `validate_package.py` overwrites `docs/validation.json` and `docs/analysis_environment.json` **in the work copy**, so these are the current CPU review records, not the archived historical execution environment. Preserve the frozen originals when comparing. The review in this artifact used predeclared atol=1e-12 and rtol=1e-10 for recomputed floating-point summaries, exact identity/seed/parameter/source-hash checks, and no image-binary equality test. See [the review report](docs/cpu_review.json) and [the manuscript mapping](docs/manuscript_mapping.md).

## Statistical conventions

One independent realization is one statistical unit. Joint success uses `mean_x > 0.8 AND mean_D < 0.2` in the **same unrounded row**. Quartiles use linear interpolation (Hyndman–Fan type 7). Paired contrasts join A_env, run_id and seed one-to-one; differences are **FG−RG**. Two-sided 95% Student-t intervals use the mean and sample standard deviation (ddof=1) of the ten realization-level differences, with nine degrees of freedom. Group confidence-interval endpoints are not subtracted.

Saved sweep drift is the mean of rounds 49,751–50,000 minus the mean of rounds 49,501–49,750, sampled every round. Topology drift uses rounds 49,501, 49,511, …, 49,991, with 25 samples in each half (first half ends 49,741; second begins 49,751). Signs are retained and absolute magnitudes summarized by median, quartiles and maximum. These protocols remain separate. Small drift within the recorded window is not a convergence test. The old core ablation lacks drift observations; `core_drift_unavailable.csv` records missing values rather than zeros.

## GPU simulation from scratch (separate optional task)

No GPU experiment was executed during this publication preparation. `requirements-observed-gpu.txt` records the historical dependency versions; the observed hardware/software information is in `new_experiments/formal/environment.json`. It retains the actual targeted gate-form runs' GPU model, Python/dependency, CUDA runtime, CUDA driver and display-driver versions while omitting machine-specific paths and device/process snapshots. These records do not recover execution environments for every historical experiment. Compatibility must be checked on the reader's machine.

The frozen runner writes environment/configuration/hash files **even without --formal**, so do not call it merely to validate archived settings. To run the 90-condition protocol from scratch, create another new sibling copy while omitting the historical formal-output directory; leave the published artifact untouched:

```sh
# Start from the repository root, in an appropriately configured GPU environment.
python -c "from pathlib import Path; import shutil; shutil.copytree(Path('supplement_20261117'), Path('../ai_debt_gpu_run/supplement_20261117'), ignore=shutil.ignore_patterns('formal', 'formal_log.txt'))"
cd ../ai_debt_gpu_run/supplement_20261117
python analysis/run_gate_comparison.py --formal
python analysis/analyze.py
python analysis/topology_drift.py
python analysis/validate_package.py
```

The runner checkpoints each identity and resumes an unchanged configuration. Never combine protocols in one output directory. Historical smoke records remain separate; keeping them does not constitute a new functional test on another GPU. No full environmental scan, parameter grid, noise/delay or network-separation experiment is launched by these commands.

## Sources, integrity and missing information

`code/` contains frozen MODEL.py and historical/reference drivers, without replacing the repository's main model directories. `new_experiments/formal/code_hashes.json` preserves historical execution SHA256 values. The current repository commit is not substituted for those hashes. Current source hashes cannot retrospectively prove the source version used for historical datasets whose execution hashes were not saved.

`docs/provenance.json` uses POSIX-style relative artifact paths and `source-workspace/` provenance aliases, which are labels for unavailable original locations, not paths required to run analysis. The unresolved representative temporal seed conflict remains recorded: the trajectory CSV identifies 20261114, adjacent JSON identifies 20260514. Its trajectory data are outside this artifact and no seed has been borrowed from the new comparison.

`MANIFEST_SHA256.csv` describes the **public, redacted artifact** and excludes itself. It is distinct from the original ZIP manifest retained privately. Archived numerical CSVs, four frozen code files, the execution runner, original execution hash JSONs and initial-state/RNG checks are byte-preserved. Path/environment/document redactions are recorded in `docs/redactions.json` and the complete public-change log. `.gitattributes` disables text/filter/encoding conversion for this artifact so Git preserves hashed bytes. Figures are previews/vector views and do not replace CSV source data.

See [data audit](docs/data_audit.md), [missing items](docs/missing_items.md), [result note](docs/results_note.md), [file list](docs/public_file_list.csv) and [publication checklist](docs/publication_checklist.md). Workspace-scanning/packaging tools, draft prose suggestions, manuscript files, private reviews, caches and the original ZIP are excluded. No license has been selected; author confirmation is still required before assigning one.
