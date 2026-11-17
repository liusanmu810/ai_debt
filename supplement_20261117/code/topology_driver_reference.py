"""Shared engine for Experiment 2 topology-robustness replications.

Scientific contract
-------------------
Claim: the four-group ablation ranking for AI use, quality assurance, and
technical debt is robust to changing the interaction topology. The two
ablated factors are reliability-memory affinity and reliability-conditioned
concentration regulation. The formal gate is rho * (1 - M'); technical debt
does not enter the gate.

The topology-specific entry points keep the dynamics, calibration, A grid,
initialisation, seeds, aggregation, and plotting identical.  Only the sparse
network generator changes.  Seed-level results are appended after every
simulation so interrupted formal runs can resume safely.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import cuda
from scipy.stats import t as student_t


MODEL_ORDER = [
    "Payoff-only",
    "Payoff+MemoryAffinity",
    "Payoff+ConcentrationGate",
    "Full Model",
]

MODEL_LABELS = {
    "Payoff-only": "Payoff only",
    "Payoff+MemoryAffinity": "Proxy-coupled diffusion",
    "Payoff+ConcentrationGate": "Concentration gate",
    "Full Model": "Full model",
}

MODEL_COLORS = {
    "Payoff-only": "#5C5C5C",
    "Payoff+MemoryAffinity": "#6F63C2",
    "Payoff+ConcentrationGate": "#2C9B52",
    "Full Model": "#1F77B4",
}

MODEL_MARKERS = {
    "Payoff-only": "o",
    "Payoff+MemoryAffinity": "D",
    "Payoff+ConcentrationGate": "s",
    "Full Model": "o",
}

REGULATION_FORMULA = "Stim = Aff * exp[-alpha_conc * rho * (1 - M_prime)]"

CORE_METRICS = ("mean_a", "mean_x", "mean_D")
ALL_METRICS = ("mean_x", "mean_a", "mean_D", "mean_M")


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def get_base_params(init_D_max: float = 0.02) -> Dict[str, float]:
    """Return the calibrated square-lattice Experiment 1--4 parameters."""
    return {
        "delta_D": 0.06,
        "beta_a": 0.095,
        "omega_D": 0.42,
        "beta_N": 0.010,
        "mu_x": 0.040,
        "kappa_D": 0.32,
        "spillover_include_self": False,
        "r_public": 3.2,
        "theta_D": 0.55,
        "b_ai": 2.30,
        "chi_x": 0.55,
        "c_x": 1.00,
        "c_a": 0.21,
        "tau_D": 0.45,
        "lambda_0": 0.10,
        "lambda_A": 0.48,
        "u_D": 2.80,
        "w_payoff": 0.88,
        "w_memory": 0.12,
        "K_P": 1.20,
        "K_M": 1.20,
        "delta": 0.30,
        "lambda_a": 1.00,
        "concentration_include_self": False,
        "alpha_conc": 0.18,
        "sigma_min": 0.001,
        "sigma_max": 0.055,
        "learning_rate": 0.13,
        "init_D_max": float(init_D_max),
    }


def get_ablation_configs() -> Dict[str, Dict[str, float]]:
    """Return the same 2 x 2 mechanism design as square-grid Experiment 2.

    ``w_memory`` controls whether reliability memory enters affinity.
    ``alpha_conc`` controls whether the concentration gate is active.
    Reliability memory itself always evolves because the active gate uses
    ``rho * (1 - M')`` even when memory affinity is ablated.
    """
    return {
        "Payoff-only": {
            "w_payoff": 1.00,
            "w_memory": 0.00,
            "alpha_conc": 0.00,
        },
        "Payoff+MemoryAffinity": {
            "w_payoff": 0.88,
            "w_memory": 0.12,
            "alpha_conc": 0.00,
        },
        "Payoff+ConcentrationGate": {
            "w_payoff": 1.00,
            "w_memory": 0.00,
            "alpha_conc": 0.18,
        },
        "Full Model": {
            "w_payoff": 0.88,
            "w_memory": 0.12,
            "alpha_conc": 0.18,
        },
    }


def validate_ablation_configs(configs: Mapping[str, Mapping[str, float]]) -> None:
    """Reject incomplete or non-identifiable ablation definitions."""
    if list(configs) != MODEL_ORDER:
        raise ValueError(
            "Ablation groups must match MODEL_ORDER exactly: " + ", ".join(MODEL_ORDER)
        )
    observed = set()
    for model_name, config in configs.items():
        if not math.isclose(
            float(config["w_payoff"]) + float(config["w_memory"]),
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Affinity weights do not sum to one for {model_name}.")
        factors = (
            bool(float(config["w_memory"]) > 0.0),
            bool(float(config["alpha_conc"]) > 0.0),
        )
        if factors in observed:
            raise ValueError(f"Duplicate ablation factor combination for {model_name}: {factors}.")
        observed.add(factors)
    if observed != {(False, False), (True, False), (False, True), (True, True)}:
        raise ValueError("Ablation configurations must form a complete 2 x 2 design.")


def make_parser(description: str, default_output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=("both", "run", "plot"), default="both")
    parser.add_argument("--num-nodes", type=int, default=10000)
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Evolution steps per realization; matched to the square-grid formal run.",
    )
    parser.add_argument(
        "--steady-window",
        type=int,
        default=500,
        help="Final steps used for late-time sampling.",
    )
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260514)
    parser.add_argument("--init-D-max", type=float, default=0.02)
    parser.add_argument(
        "--A-values",
        type=float,
        nargs="+",
        default=[round(value, 2) for value in np.linspace(0.0, 1.0, 11)],
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def configure_smoke_test(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.num_nodes = 256
    args.steps = 40
    args.steady_window = 20
    args.sample_interval = 5
    args.runs = 1
    args.A_values = [0.0, 1.0]
    args.output_dir = Path(f"{args.output_dir}_smoke")


def validate_args(args: argparse.Namespace) -> None:
    if args.num_nodes < 4:
        raise ValueError("num-nodes must be at least 4.")
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if not (0 < args.steady_window <= args.steps):
        raise ValueError("steady-window must lie in [1, steps].")
    if args.sample_interval <= 0:
        raise ValueError("sample-interval must be positive.")
    if args.runs <= 0:
        raise ValueError("runs must be positive.")
    if not (0.0 <= args.init_D_max <= 1.0):
        raise ValueError("init-D-max must be in [0, 1].")
    if not args.A_values or any(not 0.0 <= value <= 1.0 for value in args.A_values):
        raise ValueError("all A-values must be in [0, 1].")
    args.A_values = sorted(set(round(float(value), 10) for value in args.A_values))


def _tail_drift(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 0.0
    split = array.size // 2
    return float(np.mean(array[split:]) - np.mean(array[:split]))


def run_single_simulation(
    model_class: Type[Any],
    topology: str,
    network_kwargs: Mapping[str, Any],
    model_name: str,
    A_env: float,
    run_id: int,
    args: argparse.Namespace,
    model_params: Mapping[str, Any],
) -> Dict[str, Any]:
    seed = int(args.seed_base + run_id * 10000)
    model = model_class(
        num_nodes=int(args.num_nodes),
        seed=seed,
        A_env=float(A_env),
        **dict(network_kwargs),
        **dict(model_params),
    )
    network_info = model.get_network_info()
    samples: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
    start_sample = args.steps - args.steady_window
    started = time.time()

    for step in range(args.steps):
        model.step(synchronize=False)
        if step + 1 <= start_sample:
            continue
        if (step + 1 - start_sample - 1) % args.sample_interval != 0:
            continue
        means = model.get_means()
        for metric in ALL_METRICS:
            samples[metric].append(float(means[metric]))

    cuda.synchronize()
    elapsed = time.time() - started
    if not samples["mean_D"]:
        raise RuntimeError("steady-state sampling produced no observations.")

    record: Dict[str, Any] = {
        "topology": topology,
        "model": model_name,
        "A_env": float(A_env),
        "run_id": int(run_id),
        "seed": seed,
        "num_nodes": int(args.num_nodes),
        "steps": int(args.steps),
        "steady_window": int(args.steady_window),
        "sample_interval": int(args.sample_interval),
        "sample_count": int(len(samples["mean_D"])),
        "w_payoff": float(model_params["w_payoff"]),
        "w_memory": float(model_params["w_memory"]),
        "alpha_conc": float(model_params["alpha_conc"]),
        "memory_affinity_enabled": int(float(model_params["w_memory"]) > 0.0),
        "concentration_gate_enabled": int(float(model_params["alpha_conc"]) > 0.0),
        "regulation_formula": REGULATION_FORMULA,
        "elapsed_seconds": float(elapsed),
    }
    for key, value in network_info.items():
        output_key = "network_type" if key == "network_type" else f"network_{key}"
        record[output_key] = value
    for metric in ALL_METRICS:
        short = metric.removeprefix("mean_")
        record[metric] = float(np.mean(samples[metric]))
        record[f"std_{short}_time"] = float(np.std(samples[metric], ddof=0))
        record[f"tail_drift_{short}"] = _tail_drift(samples[metric])
    return record


def _ci95_half_width(std: float, n: int) -> float:
    if n <= 1 or not np.isfinite(std):
        return 0.0
    return float(student_t.ppf(0.975, df=n - 1) * std / math.sqrt(n))


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["topology", "model", "A_env"]
    grouped = raw.groupby(group_cols, sort=False, dropna=False)
    summary = grouped.size().reset_index(name="runs")

    metadata_cols = [
        column
        for column in raw.columns
        if column.startswith("network_")
        or column in {"num_nodes", "steps", "steady_window", "sample_interval"}
    ]
    for column in metadata_cols:
        values = grouped[column].first().reset_index()
        summary = summary.merge(values, on=group_cols, how="left")

    for metric in ALL_METRICS:
        stats = grouped[metric].agg(["mean", "std"]).reset_index()
        stats["std"] = stats["std"].fillna(0.0)
        stats = stats.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        summary = summary.merge(stats, on=group_cols, how="left")
        summary[f"{metric}_ci95_half_width"] = [
            _ci95_half_width(std, int(n))
            for std, n in zip(summary[f"{metric}_std"], summary["runs"])
        ]

    for metric in ("x", "a", "D", "M"):
        column = f"tail_drift_{metric}"
        drift = (
            raw.assign(_abs_drift=raw[column].abs())
            .groupby(group_cols, sort=False, dropna=False)["_abs_drift"]
            .agg(["mean", "max"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"{column}_abs_mean",
                    "max": f"{column}_abs_max",
                }
            )
        )
        summary = summary.merge(drift, on=group_cols, how="left")

    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["model", "A_env"]).reset_index(drop=True)
    summary["model"] = summary["model"].astype(str)
    return summary


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig: mpl.figure.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(f"{output_base}.png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(f"{output_base}.pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(Path(f"{output_base}.svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {output_base}.[png|pdf|svg]")


def plot_core_ablation(summary: pd.DataFrame, output_base: Path) -> None:
    set_publication_style()
    specs = (
        ("mean_a", r"AI use $\langle a\rangle$"),
        ("mean_x", r"Quality assurance $\langle x\rangle$"),
        ("mean_D", r"Technical debt $\langle D\rangle$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), sharex=True, sharey=True)
    legend_handles = []

    for panel_index, (ax, (metric, title)) in enumerate(zip(axes, specs)):
        for model_name in MODEL_ORDER:
            subset = summary[summary["model"] == model_name].sort_values("A_env")
            x = subset["A_env"].to_numpy(dtype=float)
            y = subset[f"{metric}_mean"].to_numpy(dtype=float)
            ci = subset[f"{metric}_ci95_half_width"].to_numpy(dtype=float)
            line, = ax.plot(
                x,
                y,
                color=MODEL_COLORS[model_name],
                linestyle=(0, (4, 2)),
                marker=MODEL_MARKERS[model_name],
                markerfacecolor=MODEL_COLORS[model_name],
                markeredgecolor="white",
                markeredgewidth=0.35,
                linewidth=1.9 if model_name == "Full Model" else 1.2,
                markersize=3.6 if model_name == "Full Model" else 3.0,
                label=MODEL_LABELS[model_name],
                zorder=5 if model_name == "Full Model" else 3,
            )
            ax.fill_between(
                x,
                np.clip(y - ci, 0.0, 1.0),
                np.clip(y + ci, 0.0, 1.0),
                color=MODEL_COLORS[model_name],
                alpha=0.10 if model_name == "Full Model" else 0.045,
                linewidth=0.0,
                zorder=1,
            )
            if panel_index == 0:
                legend_handles.append(line)

        ax.set_title(title, pad=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks(np.linspace(0.0, 1.0, 5))
        ax.set_xlabel(r"AI environment intensity $A$")
        ax.set_ylabel("Late-time mean" if panel_index == 0 else "")
        ax.grid(True, color="#DEDEDE", linewidth=0.5, alpha=0.72)
        _panel_label(ax, chr(ord("a") + panel_index))

    fig.legend(
        handles=legend_handles,
        labels=[MODEL_LABELS[name] for name in MODEL_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        handlelength=2.6,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.78, wspace=0.26)
    save_figure(fig, output_base)
    plt.close(fig)


def _paths(output_dir: Path, topology: str) -> Dict[str, Path]:
    stem = f"experiment_2_{topology}"
    return {
        "raw": output_dir / f"{stem}_raw.csv",
        "summary": output_dir / f"{stem}_summary.csv",
        "metadata": output_dir / f"{stem}_metadata.json",
        "figure": output_dir / f"fig_exp2_{topology}_concentration_gate_ablation_axD",
    }


def _metadata(
    args: argparse.Namespace,
    topology: str,
    network_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment": "Experiment 2 topology robustness",
        "topology": topology,
        "network_parameters": dict(network_kwargs),
        "num_nodes": int(args.num_nodes),
        "steps": int(args.steps),
        "steady_window": int(args.steady_window),
        "sample_interval": int(args.sample_interval),
        "runs": int(args.runs),
        "seed_base": int(args.seed_base),
        "init_D_max": float(args.init_D_max),
        "A_values": [float(value) for value in args.A_values],
        "models": MODEL_ORDER,
        "regulation_formula": REGULATION_FORMULA,
        "technical_debt_role": (
            "D remains in debt dynamics, payoff, and reliability-memory formation; "
            "D is absent from the concentration gate."
        ),
        "ablation_design": {
            "factor_1": "reliability-memory affinity (w_memory > 0)",
            "factor_2": "concentration gate (alpha_conc > 0)",
            "configurations": get_ablation_configs(),
        },
        "base_model_params": get_base_params(args.init_D_max),
        "uncertainty": "95% Student-t confidence interval across independent seeds",
    }


def _ensure_compatible_metadata(path: Path, requested: Mapping[str, Any]) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Existing raw results have no metadata file: {path}. "
            "Use --overwrite or choose a new output directory."
        )
    existing = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "topology",
        "network_parameters",
        "num_nodes",
        "steps",
        "steady_window",
        "sample_interval",
        "runs",
        "seed_base",
        "init_D_max",
        "A_values",
        "models",
        "regulation_formula",
        "technical_debt_role",
        "ablation_design",
        "base_model_params",
    )
    differences = [key for key in keys if existing.get(key) != requested.get(key)]
    if differences:
        raise RuntimeError(
            "Existing results are incompatible for: "
            + ", ".join(differences)
            + ". Use --overwrite or a different output directory."
        )


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    pd.DataFrame([record]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        float_format="%.10g",
    )


def _load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw result file not found: {path}")
    raw = pd.read_csv(path)
    required = {
        "topology",
        "model",
        "A_env",
        "run_id",
        "memory_affinity_enabled",
        "concentration_gate_enabled",
        "regulation_formula",
        *ALL_METRICS,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Raw results are missing columns: {missing}")
    if raw[list(required)].isna().any().any():
        raise ValueError("Raw results contain missing required values.")
    unknown = sorted(set(raw["model"].astype(str).unique()).difference(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Raw results contain unexpected model groups: {unknown}")
    formulas = set(raw["regulation_formula"].astype(str).unique())
    if formulas != {REGULATION_FORMULA}:
        raise ValueError(
            "Raw results use an incompatible regulation formula. "
            "Rerun with --overwrite or choose a new output directory."
        )
    return raw


def _validate_complete(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    expected = len(MODEL_ORDER) * len(args.A_values) * args.runs
    unique = raw[["model", "A_env", "run_id"]].drop_duplicates()
    if len(unique) != expected:
        raise RuntimeError(
            f"Experiment has {len(unique)}/{expected} completed conditions. "
            "Finish the simulation before plotting."
        )


def _print_convergence(summary: pd.DataFrame, tolerance: float = 0.01) -> None:
    print("\nMaximum absolute half-window drift across cells:")
    for metric in ("x", "a", "D", "M"):
        column = f"tail_drift_{metric}_abs_max"
        maximum = float(summary[column].max())
        status = "OK" if maximum <= tolerance else "CHECK"
        print(f"  {metric:>2s}: {maximum:.5f} [{status}]")


def run_experiment2(
    args: argparse.Namespace,
    model_class: Type[Any],
    topology: str,
    network_kwargs: Mapping[str, Any],
) -> None:
    configure_smoke_test(args)
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(output_dir, topology)
    metadata = _metadata(args, topology, network_kwargs)

    if args.overwrite and args.mode in {"both", "run"}:
        for path in (paths["raw"], paths["summary"], paths["metadata"]):
            if path.exists():
                path.unlink()
        for suffix in (".png", ".pdf", ".svg"):
            figure_path = Path(f"{paths['figure']}{suffix}")
            if figure_path.exists():
                figure_path.unlink()

    if args.mode in {"both", "run"}:
        if paths["raw"].exists():
            _ensure_compatible_metadata(paths["metadata"], metadata)
            existing_raw = _load_raw(paths["raw"])
            completed = {
                (str(row.model), round(float(row.A_env), 10), int(row.run_id))
                for row in existing_raw.itertuples(index=False)
            }
        else:
            completed = set()
            paths["metadata"].write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        configs = get_ablation_configs()
        validate_ablation_configs(configs)
        base_params = get_base_params(args.init_D_max)
        total = len(MODEL_ORDER) * len(args.A_values) * args.runs
        initially_done = len(completed)
        started = time.time()
        print("\n================ Experiment 2 topology robustness ================")
        print(f"Topology         : {topology}")
        print(f"Network params    : {dict(network_kwargs)}")
        print(f"Nodes             : {args.num_nodes}")
        print(f"Steps/window      : {args.steps}/{args.steady_window}")
        print(f"Sample interval   : {args.sample_interval}")
        print(f"Runs per condition: {args.runs}")
        print(f"A values          : {args.A_values}")
        print(f"Total simulations : {total} ({initially_done} already completed)")
        print("==================================================================")

        for run_id in range(args.runs):
            for model_name in MODEL_ORDER:
                model_params = dict(base_params)
                model_params.update(configs[model_name])
                for A_env in args.A_values:
                    key = (model_name, round(float(A_env), 10), run_id)
                    if key in completed:
                        continue
                    record = run_single_simulation(
                        model_class=model_class,
                        topology=topology,
                        network_kwargs=network_kwargs,
                        model_name=model_name,
                        A_env=float(A_env),
                        run_id=run_id,
                        args=args,
                        model_params=model_params,
                    )
                    _append_record(paths["raw"], record)
                    completed.add(key)
                    new_done = len(completed) - initially_done
                    elapsed = time.time() - started
                    eta = (elapsed / max(1, new_done)) * max(0, total - len(completed))
                    print(
                        f"{len(completed):4d}/{total} | {model_name:<27s} | "
                        f"A={A_env:.2f} | run={run_id + 1:02d}/{args.runs:02d} | "
                        f"D={record['mean_D']:.3f} | ETA={eta / 60:.1f} min"
                    )

    raw = _load_raw(paths["raw"])
    _validate_complete(raw, args)
    summary = summarize_results(raw)
    summary.to_csv(paths["summary"], index=False, float_format="%.10g")
    _print_convergence(summary)

    if args.mode in {"both", "plot"}:
        plot_core_ablation(summary, paths["figure"])

    print("\nExperiment 2 topology run complete.")
    print(f"Raw data : {paths['raw']}")
    print(f"Summary  : {paths['summary']}")
    print(f"Metadata : {paths['metadata']}")
