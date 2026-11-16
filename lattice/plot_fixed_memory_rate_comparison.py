"""Plot the fixed-memory-rate robustness check for the full model.

The script compares the established environment-dependent update

    lambda_M(A) = 0.10 + 0.48 A

with a fixed rate, lambda_M = 0.34.  Both inputs must contain seed-level
Experiment 1 outputs generated on the same A grid with the same run IDs.
Simulation is not performed here.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DYNAMIC = (
    SCRIPT_DIR
    / "results_exp1_refined_concentration_gate"
    / "experiment_1_raw.csv"
)
DEFAULT_FIXED = (
    SCRIPT_DIR
    / "results_exp1_fixed_memory_lambda0p34"
    / "experiment_1_raw.csv"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "results_exp1_fixed_memory_lambda0p34"
    / "comparison_figures"
)

METRICS = ("mean_a", "mean_x", "mean_D")
TITLES = {
    "mean_a": "AI use",
    "mean_x": "Quality assurance",
    "mean_D": "Technical debt",
}
Y_LABELS = {
    "mean_a": r"$\langle a\rangle$",
    "mean_x": r"$\langle x\rangle$",
    "mean_D": r"$\langle D\rangle$",
}

CONDITION_ORDER = ("environment-dependent", "fixed")
CONDITION_LABELS = {
    "environment-dependent": r"Environment-dependent $\lambda_M(A)$",
    "fixed": r"Fixed $\lambda_M=0.34$",
}
COLORS = {
    "environment-dependent": "#2474A6",
    "fixed": "#D58A1F",
}
LINESTYLES = {
    "environment-dependent": "-",
    "fixed": (0, (4, 2)),
}
MARKERS = {
    "environment-dependent": "o",
    "fixed": "s",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare environment-dependent and fixed memory-updating rates."
    )
    parser.add_argument("--dynamic-raw", default=str(DEFAULT_DYNAMIC))
    parser.add_argument("--fixed-raw", default=str(DEFAULT_FIXED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--save-tiff", action="store_true")
    return parser.parse_args()


def load_condition(path: Path, condition: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing seed-level result file: {path}")
    frame = pd.read_csv(path)
    required = {"A_env", "run_id", "seed", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path} contains missing values in required columns.")
    frame = frame.copy()
    frame["condition"] = condition
    return frame.sort_values(["A_env", "run_id"]).reset_index(drop=True)


def validate_pair(dynamic: pd.DataFrame, fixed: pd.DataFrame) -> None:
    dynamic_keys = dynamic[["A_env", "run_id", "seed"]].reset_index(drop=True)
    fixed_keys = fixed[["A_env", "run_id", "seed"]].reset_index(drop=True)
    if not dynamic_keys.equals(fixed_keys):
        raise ValueError(
            "The two conditions do not share the same A values, run IDs, and seeds. "
            "A paired robustness comparison would be invalid."
        )
    controlled_columns = (
        "grid_size",
        "steps",
        "steady_window",
        "r_public",
        "w_payoff",
        "w_memory",
        "alpha_conc",
        "init_D_max",
    )
    for column in controlled_columns:
        if column not in dynamic.columns or column not in fixed.columns:
            raise ValueError(f"Cannot verify controlled setting because {column!r} is missing.")
        left = dynamic[column].to_numpy(dtype=float)
        right = fixed[column].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Controlled setting {column!r} differs between the two sweeps."
            )
    if "lambda_M" not in fixed.columns or not np.allclose(
        fixed["lambda_M"].to_numpy(dtype=float), 0.34, rtol=0.0, atol=1e-12
    ):
        raise ValueError("The fixed-rate input does not consistently use lambda_M=0.34.")


def mean_ci95(values: Iterable[float]) -> Tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(array))
    if n == 1:
        return mean, np.nan, 1
    sem = float(np.std(array, ddof=1) / math.sqrt(n))
    ci = float(student_t.ppf(0.975, n - 1) * sem)
    return mean, ci, n


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (condition, A_env), group in frame.groupby(
        ["condition", "A_env"], sort=False
    ):
        record: Dict[str, float | int | str] = {
            "condition": str(condition),
            "A_env": float(A_env),
        }
        for metric in METRICS:
            mean, ci, n = mean_ci95(group[metric].to_numpy(dtype=float))
            record[metric] = mean
            record[f"{metric}_ci95"] = ci
            record[f"{metric}_n"] = n
        high_assurance = (
            (group["mean_x"].to_numpy(dtype=float) > 0.8)
            & (group["mean_D"].to_numpy(dtype=float) < 0.2)
        )
        record["high_assurance_fraction"] = float(np.mean(high_assurance))
        record["high_assurance_n"] = int(high_assurance.size)
        records.append(record)
    result = pd.DataFrame(records)
    result["condition"] = pd.Categorical(
        result["condition"], categories=CONDITION_ORDER, ordered=True
    )
    return result.sort_values(["condition", "A_env"]).reset_index(drop=True)


def paired_differences(dynamic: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    merged = fixed.merge(
        dynamic,
        on=["A_env", "run_id", "seed"],
        suffixes=("_fixed", "_dynamic"),
        validate="one_to_one",
    )
    records = []
    for A_env, group in merged.groupby("A_env", sort=True):
        for metric in METRICS:
            difference = (
                group[f"{metric}_fixed"].to_numpy(dtype=float)
                - group[f"{metric}_dynamic"].to_numpy(dtype=float)
            )
            mean, ci, n = mean_ci95(difference)
            records.append(
                {
                    "A_env": float(A_env),
                    "metric": metric,
                    "fixed_minus_dynamic": mean,
                    "ci95_half_width": ci,
                    "n": n,
                }
            )
    return pd.DataFrame(records)


def save_figure(fig, output_base: Path, save_tiff: bool) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(f"{output_base}.png"), dpi=600, bbox_inches="tight")
    fig.savefig(Path(f"{output_base}.pdf"), bbox_inches="tight")
    fig.savefig(Path(f"{output_base}.svg"), bbox_inches="tight")
    if save_tiff:
        fig.savefig(
            Path(f"{output_base}.tiff"),
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )


def plot_comparison(summary: pd.DataFrame, output_base: Path, save_tiff: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    legend_handles = []
    for panel_index, (ax, metric) in enumerate(zip(axes, METRICS)):
        for condition in CONDITION_ORDER:
            sub = summary[summary["condition"] == condition].sort_values("A_env")
            x = sub["A_env"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            ci = np.nan_to_num(sub[f"{metric}_ci95"].to_numpy(dtype=float), nan=0.0)
            line, = ax.plot(
                x,
                y,
                color=COLORS[condition],
                linestyle=LINESTYLES[condition],
                marker=MARKERS[condition],
                markerfacecolor="white",
                markeredgecolor=COLORS[condition],
                markeredgewidth=0.8,
                markersize=3.2,
                linewidth=1.55,
                label=CONDITION_LABELS[condition],
                zorder=3,
            )
            ax.fill_between(
                x,
                y - ci,
                y + ci,
                color=COLORS[condition],
                alpha=0.11,
                linewidth=0.0,
                zorder=1,
            )
            if panel_index == 0:
                legend_handles.append(line)

        ax.set_title(TITLES[metric], pad=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(r"AI environment intensity $A$")
        ax.set_ylabel(Y_LABELS[metric] if panel_index == 0 else "")
        ax.grid(True, color="#DEDEDE", linewidth=0.5, alpha=0.75)
        ax.text(
            -0.14,
            1.05,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            clip_on=False,
            fontsize=8,
            fontweight="bold",
        )

    fig.legend(
        legend_handles,
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        handlelength=3.1,
        columnspacing=1.7,
    )
    axes[0].set_ylabel("Late-time mean")
    fig.tight_layout(rect=(0.035, 0.03, 1.0, 0.82), w_pad=1.05)
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    dynamic = load_condition(Path(args.dynamic_raw), "environment-dependent")
    fixed = load_condition(Path(args.fixed_raw), "fixed")
    validate_pair(dynamic, fixed)
    combined = pd.concat([dynamic, fixed], ignore_index=True)

    summary = summarize(combined)
    differences = paired_differences(dynamic, fixed)
    summary.to_csv(output_dir / "fixed_memory_rate_summary_ci95.csv", index=False)
    differences.to_csv(output_dir / "fixed_memory_rate_paired_differences.csv", index=False)
    plot_comparison(
        summary,
        output_dir / "fig_fixed_memory_rate_A_sweep_axD",
        args.save_tiff,
    )

    print("Fixed-memory-rate comparison finished.")
    print(f"Summary: {output_dir / 'fixed_memory_rate_summary_ci95.csv'}")
    print(f"Paired differences: {output_dir / 'fixed_memory_rate_paired_differences.csv'}")


if __name__ == "__main__":
    main()
