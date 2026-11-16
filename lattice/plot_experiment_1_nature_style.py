"""Redraw completed Experiment 1 results without rerunning simulations.

The script reads seed-level output from the refined AI-environment sweep,
computes two-sided 95% Student-t confidence intervals across independent
seeds, and exports one publication-ready near-square PNG/PDF/SVG bundle.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent

METRICS = ("mean_a", "mean_x", "mean_D", "mean_M", "mean_payoff")
CORE_METRICS = ("mean_a", "mean_x", "mean_D")
SUPPORT_METRICS = ("mean_M", "mean_payoff")
STATE_KEYS = ("a", "x", "D", "M")

METRIC_TITLES = {
    "mean_a": "AI use",
    "mean_x": "Quality assurance",
    "mean_D": "Technical debt",
    "mean_M": "Reliability memory",
    "mean_payoff": "Average payoff",
}

METRIC_LABELS = {
    "mean_a": r"AI use $\langle a\rangle$",
    "mean_x": r"Quality assurance $\langle x\rangle$",
    "mean_D": r"Technical debt $\langle D\rangle$",
    "mean_M": r"Reliability memory $\langle M\rangle$",
    "mean_payoff": r"Average payoff $\langle\Pi\rangle$",
}

# Metric colours match the final Experiment 4 visual language:
# quality assurance = green, AI use = blue, technical debt = plum.
METRIC_COLORS = {
    "mean_a": "#496E99",
    "mean_x": "#4A8068",
    "mean_D": "#8A4F6B",
    "mean_M": "#8491B4",
    "mean_payoff": "#1F77B4",
}

METRIC_MARKERS = {
    "mean_a": "D",
    "mean_x": "o",
    "mean_D": "^",
    "mean_M": "D",
    "mean_payoff": "o",
}

LINESTYLE = "-"


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
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw completed Experiment 1 outputs without rerunning simulations."
    )
    parser.add_argument(
        "--input-raw",
        default=str(
            SCRIPT_DIR
            / "results_exp1_refined_concentration_gate"
            / "experiment_1_raw.csv"
        ),
        help="Seed-level Experiment 1 raw CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            SCRIPT_DIR
            / "results_exp1_refined_concentration_gate"
            / "refined_figures"
        ),
        help="Directory for revised figures and derived source data.",
    )
    parser.add_argument("--transition-min", type=float, default=0.15)
    parser.add_argument("--transition-max", type=float, default=0.30)
    parser.add_argument(
        "--drift-tolerance",
        type=float,
        default=0.01,
        help="Reference tolerance for maximum absolute tail-window drift.",
    )
    parser.add_argument("--save-tiff", action="store_true")
    return parser.parse_args()


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Experiment 1 raw data not found: {path}")

    raw = pd.read_csv(path)
    required = {
        "A_env",
        "run_id",
        "seed",
        *METRICS,
        *(f"tail_drift_{key}" for key in STATE_KEYS),
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Raw data are missing required columns: {missing}")
    if raw[list(required)].isna().any().any():
        raise ValueError("Experiment 1 raw data contain missing required values.")
    if raw.duplicated(["A_env", "run_id", "seed"]).any():
        raise ValueError("Duplicate A/run/seed records found in Experiment 1 raw data.")
    if not raw["A_env"].between(0.0, 1.0).all():
        raise ValueError("A_env must be within [0, 1].")

    if "target_runs" in raw.columns:
        observed = raw.groupby("A_env", sort=True)["run_id"].nunique()
        target = raw.groupby("A_env", sort=True)["target_runs"].first().astype(int)
        if not observed.astype(int).equals(target):
            raise ValueError("Observed run counts do not match target_runs for every A value.")

    return raw.sort_values(["A_env", "run_id"]).reset_index(drop=True)


def ci95(values: Sequence[float]) -> Tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(array))
    if n == 1:
        return mean, 0.0, 1
    sem = float(np.std(array, ddof=1) / math.sqrt(n))
    half_width = float(student_t.ppf(0.975, n - 1) * sem)
    return mean, half_width, n


def summarize_seed_results(raw: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for A_env, group in raw.groupby("A_env", sort=True):
        record: Dict[str, Any] = {
            "A_env": float(A_env),
            "runs": int(group["run_id"].nunique()),
        }
        for metric in METRICS:
            mean, half_width, n = ci95(group[metric].to_numpy(dtype=float))
            record[metric] = mean
            record[f"{metric}_ci95"] = half_width
            record[f"{metric}_n"] = n
        for key in STATE_KEYS:
            drift = np.abs(group[f"tail_drift_{key}"].to_numpy(dtype=float))
            record[f"tail_drift_{key}_abs_mean"] = float(np.mean(drift))
            record[f"tail_drift_{key}_abs_max"] = float(np.max(drift))
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values("A_env").reset_index(drop=True)


def save_figure(fig, output_base: Path, save_tiff: bool) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(f"{output_base}.svg"), bbox_inches="tight")
    fig.savefig(Path(f"{output_base}.pdf"), bbox_inches="tight")
    fig.savefig(Path(f"{output_base}.png"), dpi=600, bbox_inches="tight")
    if save_tiff:
        fig.savefig(
            Path(f"{output_base}.tiff"),
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    print(f"Saved figure bundle: {output_base}.[png|pdf|svg]")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        clip_on=False,
        fontsize=8,
        fontweight="bold",
    )


def style_A_axis(ax, xlim: Tuple[float, float] = (-0.02, 1.02)) -> None:
    ax.set_xlim(*xlim)
    if xlim[1] > 0.5:
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        ax.set_xticks([0.15, 0.20, 0.25, 0.30])
    ax.set_xlabel(r"AI environment intensity $A$")
    ax.grid(True, color="#DEDEDE", linewidth=0.5, alpha=0.75)


def draw_metric(
    ax,
    summary: pd.DataFrame,
    metric: str,
    transition_span: Tuple[float, float] | None = None,
) -> None:
    x = summary["A_env"].to_numpy(dtype=float)
    y = summary[metric].to_numpy(dtype=float)
    ci = summary[f"{metric}_ci95"].to_numpy(dtype=float)

    if transition_span is not None:
        ax.axvspan(
            transition_span[0],
            transition_span[1],
            color="#EFEFEF",
            linewidth=0.0,
            zorder=0,
        )
    ax.fill_between(
        x,
        y - ci,
        y + ci,
        color=METRIC_COLORS[metric],
        alpha=0.16,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(
        x,
        y,
        color=METRIC_COLORS[metric],
        linestyle=LINESTYLE,
        linewidth=1.7,
        marker=METRIC_MARKERS[metric],
        markersize=3.3,
        markerfacecolor=METRIC_COLORS[metric],
        markeredgecolor="white",
        markeredgewidth=0.4,
        zorder=3,
    )


def plot_core_response(
    summary: pd.DataFrame,
    output_base: Path,
    save_tiff: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True, sharey=True)
    for index, (ax, metric) in enumerate(zip(axes, CORE_METRICS)):
        draw_metric(ax, summary, metric)
        style_A_axis(ax)
        ax.set_title(METRIC_TITLES[metric], pad=4)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("Tail-window mean" if index == 0 else "")
        panel_label(ax, chr(ord("a") + index))
    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.99), w_pad=1.0)
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def largest_usage_jump(summary: pd.DataFrame) -> Tuple[float, float]:
    ordered = summary.sort_values("A_env").reset_index(drop=True)
    differences = np.abs(np.diff(ordered["mean_a"].to_numpy(dtype=float)))
    index = int(np.argmax(differences))
    return float(ordered.loc[index, "A_env"]), float(ordered.loc[index + 1, "A_env"])


def set_zoom_ylim(ax, summary: pd.DataFrame, metric: str) -> None:
    y = summary[metric].to_numpy(dtype=float)
    ci = summary[f"{metric}_ci95"].to_numpy(dtype=float)
    low = float(np.min(y - ci))
    high = float(np.max(y + ci))
    if metric == "mean_a":
        ax.set_ylim(-0.03, 1.03)
    elif metric == "mean_D":
        ax.set_ylim(0.0, max(0.05, high * 1.12))
    else:
        margin = max(0.003, 0.12 * (high - low))
        ax.set_ylim(low - margin, high + margin)


def plot_transition_focus(
    summary: pd.DataFrame,
    transition_min: float,
    transition_max: float,
    output_base: Path,
    save_tiff: bool,
) -> Tuple[float, float]:
    zoom = summary[
        summary["A_env"].between(transition_min, transition_max, inclusive="both")
    ].copy()
    if len(zoom) < 3:
        raise ValueError("Transition interval contains fewer than three A values.")

    jump_interval = largest_usage_jump(zoom)
    padding = max(0.005, 0.03 * (transition_max - transition_min))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True)
    for index, (ax, metric) in enumerate(zip(axes, CORE_METRICS)):
        draw_metric(ax, zoom, metric, transition_span=jump_interval)
        style_A_axis(ax, (transition_min - padding, transition_max + padding))
        ax.set_title(METRIC_TITLES[metric], pad=4)
        ax.set_ylabel(METRIC_LABELS[metric])
        set_zoom_ylim(ax, zoom, metric)
        panel_label(ax, chr(ord("a") + index))
    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.99), w_pad=1.1)
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)
    return jump_interval


def plot_supporting_outcomes(
    summary: pd.DataFrame,
    output_base: Path,
    save_tiff: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True)
    for index, (ax, metric) in enumerate(zip(axes, SUPPORT_METRICS)):
        draw_metric(ax, summary, metric)
        style_A_axis(ax)
        ax.set_title(METRIC_TITLES[metric], pad=4)
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric == "mean_M":
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        else:
            y = summary[metric].to_numpy(dtype=float)
            ci = summary[f"{metric}_ci95"].to_numpy(dtype=float)
            low = float(np.min(y - ci))
            high = float(np.max(y + ci))
            margin = max(0.02, 0.10 * (high - low))
            ax.set_ylim(low - margin, high + margin)
        panel_label(ax, chr(ord("a") + index))
    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.99), w_pad=1.2)
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def plot_axD_combined(
    summary: pd.DataFrame,
    output_base: Path,
    save_tiff: bool,
) -> None:
    """Overlay the three core outcomes in a conventional landscape line plot."""
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    x = summary["A_env"].to_numpy(dtype=float)
    labels = {
        "mean_a": r"AI use $\langle a\rangle$",
        "mean_x": r"Quality assurance $\langle x\rangle$",
        "mean_D": r"Technical debt $\langle D\rangle$",
    }

    for metric in CORE_METRICS:
        y = summary[metric].to_numpy(dtype=float)
        ci = summary[f"{metric}_ci95"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            np.clip(y - ci, 0.0, 1.0),
            np.clip(y + ci, 0.0, 1.0),
            color=METRIC_COLORS[metric],
            alpha=0.12,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=METRIC_COLORS[metric],
            linestyle=LINESTYLE,
            linewidth=2.0,
            marker=METRIC_MARKERS[metric],
            markersize=4.1,
            markerfacecolor="white",
            markeredgecolor=METRIC_COLORS[metric],
            markeredgewidth=0.9,
            label=labels[metric],
            zorder=3,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"AI environment intensity $A$")
    ax.grid(False)
    ax.grid(axis="y", color="#DADADA", linewidth=0.55, alpha=0.72)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Late-time mean")
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        fontsize=6.4,
        handlelength=2.35,
        handletextpad=0.55,
        columnspacing=1.25,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.90), pad=0.7)
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def plot_convergence_diagnostic(
    summary: pd.DataFrame,
    tolerance: float,
    output_base: Path,
    save_tiff: bool,
) -> List[str]:
    fig, ax = plt.subplots(figsize=(7.2, 2.85))
    x = summary["A_env"].to_numpy(dtype=float)
    failures: List[str] = []
    for key, metric in zip(STATE_KEYS, ("mean_a", "mean_x", "mean_D", "mean_M")):
        column = f"tail_drift_{key}_abs_max"
        y = summary[column].to_numpy(dtype=float)
        if float(np.max(y)) > tolerance:
            failures.append(key)
        ax.plot(
            x,
            y,
            color=METRIC_COLORS[metric],
            linestyle=LINESTYLE,
            linewidth=1.35,
            marker=METRIC_MARKERS[metric],
            markersize=3.0,
            markerfacecolor=METRIC_COLORS[metric],
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=rf"$\langle {key}\rangle$",
        )
    ax.axhline(
        tolerance,
        color="#5C5C5C",
        linestyle=(0, (1.5, 1.5)),
        linewidth=1.0,
        label=f"Tolerance ({tolerance:g})",
    )
    style_A_axis(ax)
    ax.set_ylabel("Max absolute tail-window drift")
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.40),
        ncol=5,
        handlelength=2.5,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0.04, 0.19, 1.0, 0.99))
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)
    return failures


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.transition_min < args.transition_max <= 1.0):
        raise ValueError("transition bounds must satisfy 0 <= min < max <= 1.")
    if args.drift_tolerance < 0.0:
        raise ValueError("drift-tolerance must be non-negative.")

    input_path = Path(args.input_raw).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    raw = load_raw_data(input_path)
    summary = summarize_seed_results(raw)
    summary_path = output_dir / "experiment_1_refined_summary_ci95.csv"
    summary.to_csv(summary_path, index=False)

    plot_axD_combined(
        summary,
        output_dir / "fig_exp1_axD_combined",
        args.save_tiff,
    )

    transition = summary[
        summary["A_env"].between(
            args.transition_min,
            args.transition_max,
            inclusive="both",
        )
    ]
    jump_interval = largest_usage_jump(transition)
    drift_failures = [
        key
        for key in STATE_KEYS
        if float(summary[f"tail_drift_{key}_abs_max"].max())
        > args.drift_tolerance
    ]

    runs_by_A = {
        f"{row.A_env:g}": int(row.runs)
        for row in summary[["A_env", "runs"]].itertuples(index=False)
    }
    config = {
        "input_raw": str(input_path),
        "output_dir": str(output_dir),
        "A_values": summary["A_env"].astype(float).tolist(),
        "runs_by_A": runs_by_A,
        "statistics": "mean and two-sided 95% Student-t CI across independent seeds",
        "reported_quantity": "mean over the final simulation window for each seed",
        "transition_focus": [args.transition_min, args.transition_max],
        "largest_adjacent_AI_usage_jump": list(jump_interval),
        "drift_tolerance": args.drift_tolerance,
        "state_metrics_exceeding_drift_tolerance": drift_failures,
        "figure_bundle": "fig_exp1_axD_combined.[png|pdf|svg]",
    }
    config_path = output_dir / "experiment_1_figure_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("\nExperiment 1 refined plotting finished.")
    print(f"Input raw : {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Summary   : {summary_path}")
    print(f"Config    : {config_path}")
    if drift_failures:
        print(
            "Review warning: drift tolerance exceeded for state metrics "
            + ", ".join(drift_failures)
        )


if __name__ == "__main__":
    main()
