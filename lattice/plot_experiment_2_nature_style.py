"""
Redraw the Experiment 2 mechanism ablation with the payoff--Fermi comparator.

The simulation is never rerun by this script. It reads seed-level raw results,
computes means and 95% Student-t confidence intervals across independent seeds,
and creates the following five publication-ready figures.

1. Core ablation trajectories for <a>, <x>, and <D>.
2. Supporting reliability-memory and payoff trajectories.
3. Paired effects relative to Payoff-only at a selected high-A condition.
4. The debt-payoff trajectory across the AI environment.
5. A standalone four-mechanism payoff-versus-environment figure for the manuscript.

Example
-------
    python plot_experiment_2_nature_style.py

Optional TIFF export
--------------------
    python plot_experiment_2_nature_style.py --save-tiff
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
NEW_RAW_PATH = (
    SCRIPT_DIR
    / "results_exp2_five_group_with_fermi"
    / "experiment_2_raw.csv"
)

MODEL_ORDER = [
    "Payoff-only",
    "Payoff+MemoryAffinity",
    "Payoff+ConcentrationGate",
    "Full Model",
    "Payoff-Fermi",
]

MODEL_LABELS = {
    "Payoff-only": "Payoff only",
    "Payoff+MemoryAffinity": "Proxy-coupled diffusion",
    "Payoff+ConcentrationGate": "Concentration gate",
    "Full Model": "Full model",
    "Payoff-Fermi": "Payoff--Fermi",
}

# Same mechanism palette used by the refined Experiment 3 figures.
MODEL_COLORS = {
    "Payoff-only": "#5C5C5C",
    "Payoff+MemoryAffinity": "#6F63C2",
    "Payoff+ConcentrationGate": "#2C9B52",
    "Full Model": "#1F77B4",
    "Payoff-Fermi": "#D97832",
}

MODEL_LINESTYLES = {
    "Payoff-only": (0, (4, 2)),
    "Payoff+MemoryAffinity": (0, (1, 1)),
    "Payoff+ConcentrationGate": (0, (5, 1, 1, 1)),
    "Full Model": (0, (4, 2)),
    "Payoff-Fermi": (0, (2, 1)),
}

MODEL_MARKERS = {
    "Payoff-only": "o",
    "Payoff+MemoryAffinity": "D",
    "Payoff+ConcentrationGate": "s",
    "Full Model": "o",
    "Payoff-Fermi": "^",
}

METRIC_LABELS = {
    "mean_a": r"AI use $\langle a\rangle$",
    "mean_x": r"Quality assurance $\langle x\rangle$",
    "mean_D": r"Technical debt $\langle D\rangle$",
    "mean_M": r"Endogenous reliability proxy $\langle M\rangle$",
    "mean_payoff": r"Mean model payoff $\langle\Pi\rangle$",
}

CORE_METRICS = ("mean_a", "mean_x", "mean_D")
SUPPORT_METRICS = ("mean_M", "mean_payoff")

EFFECT_COLORS = {
    "mean_a": "#6F63C2",
    "mean_x": "#2C9B52",
    "mean_D": "#D95F0E",
}

EFFECT_MARKERS = {"mean_a": "o", "mean_x": "s", "mean_D": "^"}


def default_input_raw() -> Path:
    """Return the canonical concentration-gate Experiment 2 data path."""
    return NEW_RAW_PATH


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
            "lines.linewidth": 1.5,
            "lines.markersize": 3.2,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw completed Experiment 2 ablation results without rerunning simulations."
    )
    parser.add_argument(
        "--input-raw",
        default=str(default_input_raw()),
        help="Seed-level Experiment 2 raw CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            SCRIPT_DIR
            / "results_exp2_five_group_with_fermi"
            / "refined_figures"
        ),
        help="Directory for revised figures and derived source data.",
    )
    parser.add_argument(
        "--high-A",
        type=float,
        default=0.8,
        help="A value used for the paired mechanism-effect forest plot. Default: 0.8.",
    )
    parser.add_argument("--save-tiff", action="store_true")
    return parser.parse_args()


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Experiment 2 raw data not found: {path}")
    raw = pd.read_csv(path)
    required = {
        "model",
        "A_env",
        "run_id",
        "seed",
        "mean_a",
        "mean_x",
        "mean_D",
        "mean_M",
        "mean_payoff",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Raw data are missing required columns: {missing}")

    if raw[list(required)].isna().any().any():
        raise ValueError("Experiment 2 raw data contain missing values in required columns.")
    unknown = sorted(set(raw["model"].unique()).difference(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unexpected model configurations in raw data: {unknown}")

    counts = raw.groupby(["model", "A_env"])["run_id"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Seed counts differ between model/A conditions; paired comparisons are unsafe.")

    raw = raw.copy()
    raw["model"] = pd.Categorical(raw["model"], categories=MODEL_ORDER, ordered=True)
    return raw.sort_values(["model", "A_env", "run_id"]).reset_index(drop=True)


def ci95(values: Sequence[float]) -> Tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(values))
    if n == 1:
        return mean, np.nan, 1
    sd = float(np.std(values, ddof=1))
    critical = float(student_t.ppf(0.975, df=n - 1))
    return mean, critical * sd / math.sqrt(n), n


def summarize_seed_results(raw: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for (model, A_env), group in raw.groupby(["model", "A_env"], observed=True, sort=False):
        record: Dict[str, Any] = {"model": str(model), "A_env": float(A_env)}
        for metric in (*CORE_METRICS, *SUPPORT_METRICS):
            mean, ci, n = ci95(group[metric].to_numpy(dtype=float))
            record[metric] = mean
            record[f"{metric}_ci95"] = ci
            record[f"{metric}_n"] = n
        records.append(record)
    summary = pd.DataFrame(records)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    return summary.sort_values(["model", "A_env"]).reset_index(drop=True)


def calculate_paired_effects(raw: pd.DataFrame, high_A: float) -> Tuple[pd.DataFrame, float]:
    available = np.sort(raw["A_env"].unique().astype(float))
    selected_A = float(available[np.argmin(np.abs(available - high_A))])
    if not math.isclose(selected_A, high_A, abs_tol=1e-12):
        print(f"Requested high-A={high_A:.3f}; using nearest available A={selected_A:.3f}.")

    at_A = raw[np.isclose(raw["A_env"].astype(float), selected_A)].copy()
    baseline = at_A[at_A["model"] == "Payoff-only"]
    baseline = baseline[["run_id", *CORE_METRICS]].copy()

    paired_records: List[Dict[str, Any]] = []
    for model in MODEL_ORDER[1:]:
        model_rows = at_A[at_A["model"] == model][["run_id", *CORE_METRICS]].copy()
        paired = model_rows.merge(
            baseline,
            on="run_id",
            suffixes=("_model", "_baseline"),
            validate="one_to_one",
        )
        for metric in CORE_METRICS:
            effect = (
                paired[f"{metric}_model"].to_numpy(dtype=float)
                - paired[f"{metric}_baseline"].to_numpy(dtype=float)
            )
            mean, ci, n = ci95(effect)
            paired_records.append(
                {
                    "model": model,
                    "A_env": selected_A,
                    "metric": metric,
                    "effect": mean,
                    "ci95": ci,
                    "n": n,
                }
            )
    return pd.DataFrame(paired_records), selected_A


def save_figure(fig, output_base: Path, save_tiff: bool) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    # Append extensions instead of replacing suffixes so names such as A0.8 are preserved.
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


def draw_model_trajectories(
    ax,
    summary: pd.DataFrame,
    metric: str,
    show_labels: bool,
    linestyle_override: Any | None = None,
) -> List[Any]:
    handles = []
    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model].sort_values("A_env")
        x = sub["A_env"].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        ci = sub[f"{metric}_ci95"].to_numpy(dtype=float)
        line, = ax.plot(
            x,
            y,
            color=MODEL_COLORS[model],
            linestyle=(
                linestyle_override
                if linestyle_override is not None
                else MODEL_LINESTYLES[model]
            ),
            marker=MODEL_MARKERS[model],
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.35,
            linewidth=1.9 if model == "Full Model" else 1.25,
            markersize=3.6 if model == "Full Model" else 3.0,
            label=MODEL_LABELS[model] if show_labels else "_nolegend_",
            zorder=5 if model == "Full Model" else 3,
        )
        ax.fill_between(
            x,
            y - ci,
            y + ci,
            color=MODEL_COLORS[model],
            alpha=0.11 if model == "Full Model" else 0.055,
            linewidth=0.0,
            zorder=1,
        )
        handles.append(line)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"AI environment intensity $A$")
    ax.grid(True, color="#DEDEDE", linewidth=0.5, alpha=0.75)
    return handles


def plot_core_ablation(summary: pd.DataFrame, output_base: Path, save_tiff: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), sharex=True, sharey=True)
    titles = ["AI use", "Quality assurance", "Technical debt"]
    handles = None
    for index, (ax, metric, title) in enumerate(zip(axes, CORE_METRICS, titles)):
        lines = draw_model_trajectories(
            ax,
            summary,
            metric,
            show_labels=index == 0,
            linestyle_override=(0, (4, 2)),
        )
        if index == 0:
            handles = lines
        ax.set_title(title, pad=4)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("Late-time mean" if index == 0 else "")
        panel_label(ax, chr(ord("a") + index))

    fig.legend(
        handles=handles,
        labels=[MODEL_LABELS[model] for model in MODEL_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        handlelength=2.7,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0.04, 0.22, 1.0, 0.98), w_pad=1.0)
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def plot_supporting_outcomes(summary: pd.DataFrame, output_base: Path, save_tiff: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    handles = None
    for index, (ax, metric, title) in enumerate(
        zip(axes, SUPPORT_METRICS, ["Reliability proxy", "Model payoff"])
    ):
        lines = draw_model_trajectories(ax, summary, metric, show_labels=index == 0)
        if index == 0:
            handles = lines
        ax.set_title(title, pad=4)
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric == "mean_M":
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        else:
            values = summary[metric].to_numpy(dtype=float)
            ci = np.nan_to_num(summary[f"{metric}_ci95"].to_numpy(dtype=float), nan=0.0)
            low = float(np.min(values - ci))
            high = float(np.max(values + ci))
            margin = max(0.05, 0.08 * (high - low))
            ax.set_ylim(low - margin, high + margin)
        panel_label(ax, chr(ord("a") + index))

    fig.legend(
        handles=handles,
        labels=[MODEL_LABELS[model] for model in MODEL_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        handlelength=2.7,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0.04, 0.22, 1.0, 0.98), w_pad=1.15)
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def plot_payoff_trajectories(
    summary: pd.DataFrame,
    output_base: Path,
    save_tiff: bool,
) -> None:
    """Standalone payoff panel matching the core ablation visual language."""
    fig, ax = plt.subplots(figsize=(5.7, 3.25))
    handles = draw_model_trajectories(
        ax,
        summary,
        "mean_payoff",
        show_labels=True,
        linestyle_override=(0, (4, 2)),
    )
    values = summary["mean_payoff"].to_numpy(dtype=float)
    ci = np.nan_to_num(summary["mean_payoff_ci95"].to_numpy(dtype=float), nan=0.0)
    low = float(np.min(values - ci))
    high = float(np.max(values + ci))
    margin = max(0.05, 0.08 * (high - low))
    ax.set_ylim(low - margin, high + margin)
    ax.set_ylabel(r"Mean model payoff $\langle\Pi\rangle$")
    fig.legend(
        handles=handles,
        labels=[MODEL_LABELS[model] for model in MODEL_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        handlelength=2.7,
        columnspacing=1.35,
    )
    fig.tight_layout(rect=(0.05, 0.03, 1.0, 0.82))
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def plot_high_A_effects(
    effects: pd.DataFrame,
    selected_A: float,
    output_base: Path,
    save_tiff: bool,
) -> None:
    models = MODEL_ORDER[1:]
    y = np.arange(len(models), dtype=float)
    offsets = {"mean_a": -0.19, "mean_x": 0.00, "mean_D": 0.19}
    fig, ax = plt.subplots(figsize=(7.2, 3.25))

    for row in range(len(models)):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color="#F5F5F5", zorder=0)

    for metric in CORE_METRICS:
        means = []
        ci = []
        for model in models:
            row = effects[(effects["model"] == model) & (effects["metric"] == metric)].iloc[0]
            means.append(float(row["effect"]))
            ci.append(float(row["ci95"]))
        ax.errorbar(
            means,
            y + offsets[metric],
            xerr=ci,
            fmt=EFFECT_MARKERS[metric],
            markersize=4.0,
            color=EFFECT_COLORS[metric],
            ecolor=EFFECT_COLORS[metric],
            elinewidth=0.9,
            capsize=2.0,
            markeredgewidth=0.0,
            label={
                "mean_a": r"$\Delta\langle a\rangle$",
                "mean_x": r"$\Delta\langle x\rangle$",
                "mean_D": r"$\Delta\langle D\rangle$",
            }[metric],
            zorder=3,
        )

    ax.axvline(0.0, color="#606060", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[model] for model in models])
    ax.invert_yaxis()
    ax.set_xlabel("Paired effect relative to Payoff only")
    ax.grid(axis="x", color="#DEDEDE", linewidth=0.5, alpha=0.75)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3)
    fig.tight_layout(rect=(0.08, 0.02, 1.0, 0.98))
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def plot_debt_payoff_tradeoff(
    summary: pd.DataFrame,
    output_base: Path,
    save_tiff: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model].sort_values("A_env")
        debt = sub["mean_D"].to_numpy(dtype=float)
        payoff = sub["mean_payoff"].to_numpy(dtype=float)
        ax.plot(
            debt,
            payoff,
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=1.9 if model == "Full Model" else 1.25,
            marker=MODEL_MARKERS[model],
            markersize=3.6 if model == "Full Model" else 3.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=MODEL_LABELS[model],
            zorder=5 if model == "Full Model" else 3,
        )
        ax.scatter(
            debt[0],
            payoff[0],
            s=22,
            facecolors="white",
            edgecolors=MODEL_COLORS[model],
            linewidths=0.9,
            zorder=6,
        )
        ax.scatter(
            debt[-1],
            payoff[-1],
            s=22,
            facecolors=MODEL_COLORS[model],
            edgecolors="white",
            linewidths=0.4,
            zorder=6,
        )

    ax.set_xlabel(r"Technical debt $\langle D\rangle$")
    ax.set_ylabel(r"Average payoff $\langle\Pi\rangle$")
    ax.set_xlim(-0.03, 1.03)
    ax.grid(True, color="#DEDEDE", linewidth=0.5, alpha=0.75)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        handlelength=2.7,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0.04, 0.18, 1.0, 0.99))
    save_figure(fig, output_base, save_tiff)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.high_A <= 1.0):
        raise ValueError("high-A must be in [0, 1].")

    input_path = Path(args.input_raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    raw = load_raw_data(input_path)
    canonical_raw_path = output_dir.parent / "experiment_2_raw.csv"
    if input_path.resolve() != canonical_raw_path.resolve():
        canonical = raw.copy()
        canonical["model"] = canonical["model"].astype(str)
        canonical.to_csv(canonical_raw_path, index=False)
        print(f"Saved canonical five-group raw data: {canonical_raw_path}")
    summary = summarize_seed_results(raw)
    effects, selected_A = calculate_paired_effects(raw, args.high_A)

    summary_path = output_dir / "experiment_2_refined_summary_ci95.csv"
    effects_path = output_dir / f"experiment_2_paired_effects_A{selected_A:.1f}.csv"
    summary.to_csv(summary_path, index=False)
    effects.to_csv(effects_path, index=False)

    plot_core_ablation(
        summary,
        output_dir / "fig_exp2_core_ablation_axD",
        args.save_tiff,
    )
    plot_supporting_outcomes(
        summary,
        output_dir / "fig_exp2_supporting_memory_payoff",
        args.save_tiff,
    )
    plot_payoff_trajectories(
        summary,
        output_dir / "fig_exp2_payoff_vs_A",
        args.save_tiff,
    )
    plot_high_A_effects(
        effects,
        selected_A,
        output_dir / f"fig_exp2_paired_effects_A{selected_A:.1f}",
        args.save_tiff,
    )
    plot_debt_payoff_tradeoff(
        summary,
        output_dir / "fig_exp2_debt_payoff_tradeoff",
        args.save_tiff,
    )

    config = {
        "input_raw": str(input_path),
        "output_dir": str(output_dir),
        "models": MODEL_ORDER,
        "A_values": sorted(raw["A_env"].astype(float).unique().tolist()),
        "runs_per_condition": int(
            raw.groupby(["model", "A_env"], observed=True)["run_id"].nunique().iloc[0]
        ),
        "uncertainty": "95% Student-t confidence interval across independent seeds",
        "high_A_for_paired_effects": selected_A,
        "figure_outcomes": ["mean_a", "mean_x", "mean_D", "mean_M", "mean_payoff"],
    }
    with (output_dir / "experiment_2_figure_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("\nExperiment 2 refined plotting finished.")
    print(f"Input raw : {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Summary   : {summary_path}")
    print(f"Effects   : {effects_path}")


if __name__ == "__main__":
    main()
