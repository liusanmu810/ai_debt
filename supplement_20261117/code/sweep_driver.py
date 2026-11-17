"""
Experiment 1: AI Environment Intensity Sweep
============================================

研究目的
--------
检验外部 AI 环境强度 A 对系统长期演化的影响：

    A ↑ -> AI 使用强度 <a> ↑
        -> 质量保障投入 <x> 被挤出
        -> AI 诱发技术债 <D> 积累
        -> 可靠性记忆 <M> 下降，长期收益 <Pi> 受损

本脚本只负责实验 1，不包含模型定义。
模型定义请放在同目录下的 MODEL.py 中，并保证其中包含：

    AITechnicalDebtMDAE_GPU

注意
----
本实验脚本已适配“删除 M 直接清债项”的新版 MODEL.py。
因此 get_model_params() 中不再传入 mu_M。

也就是说，可靠性记忆 M 仍然通过以下机制发挥作用：

    1. 可靠性记忆更新；
    2. MDAE 记忆亲和度；
    3. 免疫刺激值中的低可靠性抑制。

但 M 不再直接进入技术债清理项。

重要初始化设定
--------------
本实验依赖新版 MODEL.py 的初始化方式：

    x_i(0): 随机异质，但系统总平均严格为 0.5
    a_i(0): 随机异质，但系统总平均严格为 0.5
    D_i(0): 低初始技术债，默认 D_i(0) ~ U(0, 0.02)
    M_i(0): 中性可靠性记忆，M_i(0)=0.5

为保证 A 扫描公平，同一个 run_id 下，不同 A_env 使用相同 seed。
因此同一个 run_id 内，不同 A 的初始状态和随机扰动序列一致，差异主要来自 A_env。

运行方式
--------
默认正式运行：

    python experiment_1_ai_environment_sweep_no_M_cleaning.py

快速测试：

    python experiment_1_ai_environment_sweep_no_M_cleaning.py --grid-size 40 --steps 1000 --steady-window 100 --runs 2

更稳定的正式实验：

    python experiment_1_ai_environment_sweep_no_M_cleaning.py --grid-size 100 --steps 10000 --steady-window 1000 --runs 10

如果希望严格从无技术债开始：

    python experiment_1_ai_environment_sweep_no_M_cleaning.py --init-D-max 0.0

输出文件
--------
results_exp1/experiment_1_raw.csv
results_exp1/experiment_1_summary.csv
results_exp1/experiment_1_main_curves.png
results_exp1/experiment_1_main_curves_no_std.png

Refined design
--------------
The current implementation adds a dense scan in the transition region,
additional independent seeds there, 95% Student-t confidence intervals, and a
tail-window drift diagnostic. New concentration-gate formal runs use
``results_exp1_refined_concentration_gate`` so
the original coarse-scan outputs are preserved.
"""

import os
import time
import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import cuda
from scipy.stats import t as student_t

from MODEL import AITechnicalDebtMDAE_GPU


METRICS = ["mean_x", "mean_a", "mean_D", "mean_M", "mean_payoff"]
STATE_METRICS = ["mean_x", "mean_a", "mean_D", "mean_M"]

# Preserve the original 0.1 grid and resolve the observed A=0.2--0.3 transition.
DEFAULT_A_VALUES = [
    0.0, 0.1,
    0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


# ==============================================================================
# 1. 参数：实验 1 当前候选默认参数
# ==============================================================================


def get_model_params(init_D_max: float = 0.02) -> Dict[str, float]:
    """
    实验 1 使用的默认参数。

    说明：
        - c_x 固定为 1，保持经典公共物品博弈中的贡献成本规范化；
        - r_public < 5，仍然保留公共品社会困境；
        - init_D_max 控制初始技术债上界：D_i(0) ~ U(0, init_D_max)；
        - 已删除 mu_M，因为新版 MODEL.py 中 M 不再直接参与技术债清理。
    """
    return dict(
        # ------------------------------------------------------------------
        # AI 诱发技术债动态参数
        # ------------------------------------------------------------------
        delta_D=0.06,
        beta_a=0.095,
        omega_D=0.42,
        beta_N=0.010,
        mu_x=0.040,
        kappa_D=0.32,
        spillover_include_self=False,

        # ------------------------------------------------------------------
        # 公共物品收益与个体收益参数
        # ------------------------------------------------------------------
        r_public=3.2,
        theta_D=0.55,
        b_ai=2.30,
        chi_x=0.55,
        c_x=1.00,
        c_a=0.21,
        tau_D=0.45,

        # ------------------------------------------------------------------
        # 可靠性记忆参数
        # ------------------------------------------------------------------
        lambda_0=0.10,
        lambda_A=0.48,
        u_D=2.80,

        # ------------------------------------------------------------------
        # MDAE 亲和度参数
        # ------------------------------------------------------------------
        w_payoff=0.88,
        w_memory=0.12,
        K_P=1.20,
        K_M=1.20,

        # ------------------------------------------------------------------
        # 双策略浓度与免疫抑制参数
        # ------------------------------------------------------------------
        delta=0.3,
        lambda_a=1.00,
        concentration_include_self=False,
        alpha_conc=0.18,
        screening_mode=3,  # rho * (1 - M') concentration gate

        # ------------------------------------------------------------------
        # 自适应扰动与惯性策略更新参数
        # ------------------------------------------------------------------
        sigma_min=0.001,
        sigma_max=0.055,
        learning_rate=0.13,

        # ------------------------------------------------------------------
        # 初始状态参数
        # ------------------------------------------------------------------
        init_D_max=float(init_D_max),
    )

# ==============================================================================
# 2. 单次仿真
# ==============================================================================


def run_single_simulation(
    A_env: float,
    run_id: int,
    grid_size: int,
    steps: int,
    steady_window: int,
    seed_base: int,
    model_params: Dict[str, float],
) -> Dict[str, float]:
    """
    对给定 A_env 和随机种子运行一次仿真。

    同一个 run_id 下，不同 A_env 使用相同 seed。
    这样不同 A 的差异主要来自 A_env 本身，而不是初始状态差异。

    返回最后 steady_window 步的时间平均指标。
    """
    seed = seed_base + run_id * 10000

    model = AITechnicalDebtMDAE_GPU(
        grid_size=grid_size,
        seed=seed,
        A_env=A_env,
        **model_params,
    )

    mean_x_list: List[float] = []
    mean_a_list: List[float] = []
    mean_D_list: List[float] = []
    mean_M_list: List[float] = []
    mean_payoff_list: List[float] = []

    start_time = time.time()

    for t in range(steps):
        model.step(synchronize=False)
        current_t = t + 1

        if current_t > steps - steady_window:
            means = model.get_means()
            observables = model.get_observables()

            mean_x_list.append(means["mean_x"])
            mean_a_list.append(means["mean_a"])
            mean_D_list.append(means["mean_D"])
            mean_M_list.append(means["mean_M"])
            mean_payoff_list.append(float(np.mean(observables["payoff"])))

    cuda.synchronize()
    elapsed = time.time() - start_time

    def tail_drift(values: List[float]) -> float:
        """Second-half minus first-half mean within the steady-state window."""
        array = np.asarray(values, dtype=np.float64)
        if array.size < 2:
            return 0.0
        midpoint = array.size // 2
        return float(np.mean(array[midpoint:]) - np.mean(array[:midpoint]))

    result = {
        "A_env": float(A_env),
        "run_id": int(run_id),
        "seed": int(seed),
        "grid_size": int(grid_size),
        "steps": int(steps),
        "steady_window": int(steady_window),
        "r_public": float(model_params["r_public"]),
        "w_payoff": float(model_params["w_payoff"]),
        "w_memory": float(model_params["w_memory"]),
        "alpha_conc": float(model_params["alpha_conc"]),
        "lambda_0": float(model_params["lambda_0"]),
        "lambda_A": float(model_params["lambda_A"]),
        "lambda_M": float(model._lambda_M()),
        "init_D_max": float(model_params["init_D_max"]),

        "mean_x": float(np.mean(mean_x_list)),
        "mean_a": float(np.mean(mean_a_list)),
        "mean_D": float(np.mean(mean_D_list)),
        "mean_M": float(np.mean(mean_M_list)),
        "mean_payoff": float(np.mean(mean_payoff_list)),

        # 这是单次仿真在稳态窗口内的时间波动，不是跨重复实验标准差。
        "std_x_time": float(np.std(mean_x_list)),
        "std_a_time": float(np.std(mean_a_list)),
        "std_D_time": float(np.std(mean_D_list)),
        "std_M_time": float(np.std(mean_M_list)),
        "std_payoff_time": float(np.std(mean_payoff_list)),

        # Convergence diagnostics; these are not independent-replicate effects.
        "tail_drift_x": tail_drift(mean_x_list),
        "tail_drift_a": tail_drift(mean_a_list),
        "tail_drift_D": tail_drift(mean_D_list),
        "tail_drift_M": tail_drift(mean_M_list),
        "tail_drift_payoff": tail_drift(mean_payoff_list),

        "elapsed_seconds": float(elapsed),
    }

    return result


# ==============================================================================
# 3. 汇总与绘图
# ==============================================================================


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """按 A_env 汇总不同随机重复实验。"""
    work_df = raw_df.copy()
    grouped = work_df.groupby("A_env")
    summary = grouped.size().reset_index(name="runs")

    metadata_cols = [
        "grid_size", "steps", "steady_window", "r_public",
        "w_payoff", "w_memory", "alpha_conc",
        "lambda_0", "lambda_A", "lambda_M", "memory_update_mode",
        "init_D_max", "target_runs", "is_transition_point",
    ]
    for col in metadata_cols:
        if col in work_df.columns:
            metadata = grouped[col].first().reset_index()
            summary = summary.merge(metadata, on="A_env", how="left")

    for metric in METRICS:
        temp = grouped[metric].agg(["mean", "std", "count"]).reset_index()
        temp["std"] = temp["std"].fillna(0.0)
        temp["sem"] = temp["std"] / np.sqrt(temp["count"])
        temp["t_critical_95"] = temp["count"].map(
            lambda n: float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
        )
        temp["ci95_half_width"] = temp["t_critical_95"] * temp["sem"]
        temp["ci95_low"] = temp["mean"] - temp["ci95_half_width"]
        temp["ci95_high"] = temp["mean"] + temp["ci95_half_width"]
        temp = temp.rename(columns={
            "mean": f"{metric}_mean",
            "std": f"{metric}_std",
            "sem": f"{metric}_sem",
            "ci95_half_width": f"{metric}_ci95_half_width",
            "ci95_low": f"{metric}_ci95_low",
            "ci95_high": f"{metric}_ci95_high",
        })
        temp = temp.drop(columns=["count", "t_critical_95"])
        summary = summary.merge(temp, on="A_env", how="left")

        drift_col = f"tail_drift_{metric.removeprefix('mean_')}"
        if drift_col in work_df.columns:
            abs_col = f"{drift_col}_abs"
            work_df[abs_col] = work_df[drift_col].abs()
            drift_temp = work_df.groupby("A_env")[abs_col].agg(["mean", "max"]).reset_index()
            drift_temp = drift_temp.rename(columns={
                "mean": f"{drift_col}_abs_mean",
                "max": f"{drift_col}_abs_max",
            })
            summary = summary.merge(drift_temp, on="A_env", how="left")

    # 单次重复时 std 为 NaN，统一填 0。
    summary = summary.fillna(0.0)
    summary = summary.sort_values("A_env").reset_index(drop=True)
    return summary


def plot_main_curves(summary_df: pd.DataFrame, output_path: str, show_ci: bool = True) -> None:
    """Plot seed means with an optional 95% Student-t confidence band."""
    A = summary_df["A_env"].to_numpy(dtype=float)
    plot_specs = [
        ("mean_x", "Quality assurance investment", r"$\langle x \rangle$"),
        ("mean_a", "AI usage intensity", r"$\langle a \rangle$"),
        ("mean_D", "AI-induced technical debt", r"$\langle D \rangle$"),
        ("mean_M", "Reliability memory", r"$\langle M \rangle$"),
        ("mean_payoff", "Mean model payoff", r"$\langle \Pi \rangle$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True)
    axes = axes.ravel()
    color = "#1F77B4"

    for idx, (metric, title, ylabel) in enumerate(plot_specs):
        ax = axes[idx]
        y = summary_df[f"{metric}_mean"].to_numpy(dtype=float)
        ax.plot(A, y, marker="o", markersize=3.0, linewidth=1.4, color=color)

        if show_ci:
            low = summary_df[f"{metric}_ci95_low"].to_numpy(dtype=float)
            high = summary_df[f"{metric}_ci95_high"].to_numpy(dtype=float)
            ax.fill_between(A, low, high, color=color, alpha=0.18, linewidth=0)

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("AI environment intensity A")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=6.5)

    axes[-1].axis("off")
    note = (
        "Shading: 95% Student-t CI across independent seeds; n is reported in source data."
        if show_ci else
        "Lines: means across independent seeds; n is reported in source data."
    )
    fig.text(0.01, 0.01, note, fontsize=6, color="#4D4D4D")
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))

    stem, _ = os.path.splitext(output_path)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def print_summary_and_diagnosis(summary_df: pd.DataFrame, drift_tolerance: float) -> None:
    """打印汇总结果和简单诊断。"""
    print("\n================ Summary ================")
    print(summary_df.to_string(index=False))

    first = summary_df.iloc[0]
    last = summary_df.iloc[-1]

    delta_a = last["mean_a_mean"] - first["mean_a_mean"]
    delta_x = last["mean_x_mean"] - first["mean_x_mean"]
    delta_D = last["mean_D_mean"] - first["mean_D_mean"]
    delta_M = last["mean_M_mean"] - first["mean_M_mean"]
    delta_payoff = last["mean_payoff_mean"] - first["mean_payoff_mean"]

    print("\n================ Quick Diagnosis ================")
    print(f"Delta <a>      : {delta_a:.4f}")
    print(f"Delta <x>      : {delta_x:.4f}")
    print(f"Delta <D>      : {delta_D:.4f}")
    print(f"Delta <M>      : {delta_M:.4f}")
    print(f"Delta <Pi>     : {delta_payoff:.4f}")

    if delta_a > 0 and delta_x < 0 and delta_D > 0:
        print("Main trend: A increases AI usage, reduces QA investment, and increases technical debt.")
    else:
        print("Main trend is not fully clear. Check parameter settings or run more repetitions.")

    max_D = summary_df["mean_D_mean"].max()
    min_D = summary_df["mean_D_mean"].min()
    print(f"Technical debt range: {min_D:.4f} ~ {max_D:.4f}")

    if max_D > 0.85:
        print("Warning: D is close to saturation under high A.")
    elif max_D < 0.10:
        print("Warning: D may be too small to support the debt-accumulation narrative.")
    else:
        print("D is in a visible and interpretable range.")

    ordered = summary_df.sort_values("A_env").reset_index(drop=True)
    if len(ordered) > 1:
        jumps = np.abs(np.diff(ordered["mean_a_mean"].to_numpy(dtype=float)))
        jump_idx = int(np.argmax(jumps))
        print(
            "Largest adjacent <a> change: "
            f"A={ordered.loc[jump_idx, 'A_env']:.3f} -> "
            f"A={ordered.loc[jump_idx + 1, 'A_env']:.3f}, "
            f"|Delta|={jumps[jump_idx]:.4f}"
        )

    print("\nTail-window convergence diagnostic:")
    drift_failures = []
    for metric in STATE_METRICS:
        drift_col = f"tail_drift_{metric.removeprefix('mean_')}_abs_max"
        if drift_col not in summary_df.columns:
            continue
        max_drift = float(summary_df[drift_col].max())
        print(f"  max absolute half-window drift, {metric}: {max_drift:.5f}")
        if max_drift > drift_tolerance:
            drift_failures.append(metric)
    if drift_failures:
        print(
            f"Warning: drift exceeds tolerance {drift_tolerance:g} for "
            f"{', '.join(drift_failures)}. Increase steps or steady-window."
        )
    else:
        print(f"All state-variable drifts are within tolerance {drift_tolerance:g}.")


# ==============================================================================
# 4. 主程序
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: AI environment intensity sweep."
    )

    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--steady-window", type=int, default=500)
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Independent seeds per A outside the transition region.",
    )
    parser.add_argument(
        "--transition-runs",
        type=int,
        default=10,
        help="Independent seeds per A inside [transition-min, transition-max].",
    )
    parser.add_argument("--transition-min", type=float, default=0.15)
    parser.add_argument("--transition-max", type=float, default=0.30)
    parser.add_argument(
        "--drift-tolerance",
        type=float,
        default=0.01,
        help="Warning threshold for absolute half-window drift of x, a, D, or M.",
    )
    parser.add_argument("--seed-base", type=int, default=20260514)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to the established Experiment 1 directory "
            "for environment-dependent memory and to a separate robustness directory "
            "for fixed memory updating."
        ),
    )
    parser.add_argument(
        "--memory-update-mode",
        choices=("environment-dependent", "fixed"),
        default="environment-dependent",
        help=(
            "Use lambda_M(A)=lambda_0+lambda_A*A or hold lambda_M fixed while A varies."
        ),
    )
    parser.add_argument(
        "--fixed-lambda",
        type=float,
        default=0.34,
        help=(
            "Fixed memory-updating rate used when --memory-update-mode=fixed. "
            "The default 0.34 equals the baseline rate at A=0.5."
        ),
    )
    parser.add_argument(
        "--init-D-max",
        type=float,
        default=0.02,
        help="Initial technical debt upper bound. D_i(0) ~ U(0, init_D_max).",
    )

    parser.add_argument(
        "--A-values",
        type=float,
        nargs="+",
        default=DEFAULT_A_VALUES,
        help=(
            "AI environment intensities. Default keeps the 0.1 coarse grid and "
            "adds 0.025-spaced points from A=0.15 to 0.30."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        fixed_tag = f"{args.fixed_lambda:.2f}".replace(".", "p")
        args.output_dir = (
            f"results_exp1_fixed_memory_lambda{fixed_tag}"
            if args.memory_update_mode == "fixed"
            else "results_exp1_refined_concentration_gate"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.steady_window <= 0:
        raise ValueError("steady-window must be positive.")
    if args.steady_window > args.steps:
        raise ValueError("steady-window cannot be larger than steps.")
    if args.runs <= 0:
        raise ValueError("runs must be positive.")
    if args.transition_runs <= 0:
        raise ValueError("transition-runs must be positive.")
    if args.transition_runs < args.runs:
        raise ValueError("transition-runs must be greater than or equal to runs.")
    if not (0.0 <= args.transition_min <= args.transition_max <= 1.0):
        raise ValueError("transition bounds must satisfy 0 <= min <= max <= 1.")
    if args.drift_tolerance < 0.0:
        raise ValueError("drift-tolerance must be non-negative.")
    if args.grid_size <= 0:
        raise ValueError("grid-size must be positive.")
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if not (0.0 <= args.init_D_max <= 1.0):
        raise ValueError("init-D-max must be in [0, 1].")
    if not (0.0 <= args.fixed_lambda <= 1.0):
        raise ValueError("fixed-lambda must be in [0, 1].")
    if any((A < 0.0 or A > 1.0) for A in args.A_values):
        raise ValueError("All A-values must be in [0, 1].")

    args.A_values = sorted(set(float(A) for A in args.A_values))

    model_params = get_model_params(init_D_max=args.init_D_max)
    if args.memory_update_mode == "fixed":
        model_params["lambda_0"] = float(args.fixed_lambda)
        model_params["lambda_A"] = 0.0
    target_runs_by_A = {
        A: (
            args.transition_runs
            if args.transition_min <= A <= args.transition_max
            else args.runs
        )
        for A in args.A_values
    }
    total_simulations = int(sum(target_runs_by_A.values()))

    print("\n================ Experiment 1: AI Environment Intensity Sweep ================")
    print(f"Grid size      : {args.grid_size} x {args.grid_size}")
    print(f"Steps          : {args.steps}")
    print(f"Steady window  : {args.steady_window}")
    print(f"Base runs/A    : {args.runs}")
    print(
        f"Transition runs: {args.transition_runs} for "
        f"A in [{args.transition_min}, {args.transition_max}]"
    )
    print(f"Total sims     : {total_simulations}")
    print(f"Seed base      : {args.seed_base}")
    print(f"Init D max     : {args.init_D_max}")
    print(f"Memory update  : {args.memory_update_mode}")
    if args.memory_update_mode == "fixed":
        print(f"Fixed lambda_M : {args.fixed_lambda}")
    print(f"A values       : {args.A_values}")
    print(f"Output dir     : {args.output_dir}")
    print("=============================================================================\n")

    all_results = []
    total_start = time.time()

    max_runs = max(target_runs_by_A.values())
    for run_id in range(max_runs):
        active_A_values = [A for A in args.A_values if run_id < target_runs_by_A[A]]
        if not active_A_values:
            continue
        print(f"\n---------- Initial condition / seed index {run_id + 1}/{max_runs} ----------")
        for A_env in active_A_values:
            target_runs = target_runs_by_A[A_env]
            print(f"Running A={A_env:.3f}, run={run_id + 1}/{target_runs} ...")

            result = run_single_simulation(
                A_env=A_env,
                run_id=run_id,
                grid_size=args.grid_size,
                steps=args.steps,
                steady_window=args.steady_window,
                seed_base=args.seed_base,
                model_params=model_params,
            )
            result["target_runs"] = int(target_runs)
            result["is_transition_point"] = bool(
                args.transition_min <= A_env <= args.transition_max
            )
            result["memory_update_mode"] = args.memory_update_mode
            all_results.append(result)

            print(
                f"  <x>={result['mean_x']:.4f}, "
                f"<a>={result['mean_a']:.4f}, "
                f"<D>={result['mean_D']:.4f}, "
                f"<M>={result['mean_M']:.4f}, "
                f"<Pi>={result['mean_payoff']:.4f}, "
                f"time={result['elapsed_seconds']:.2f}s"
            )

    total_elapsed = time.time() - total_start

    raw_df = pd.DataFrame(all_results)
    summary_df = summarize_results(raw_df)

    raw_path = os.path.join(args.output_dir, "experiment_1_raw.csv")
    summary_path = os.path.join(args.output_dir, "experiment_1_summary.csv")
    config_path = os.path.join(args.output_dir, "experiment_1_config.json")
    fig_path = os.path.join(args.output_dir, "experiment_1_main_curves_ci95.png")
    fig_mean_only_path = os.path.join(args.output_dir, "experiment_1_main_curves_mean_only.png")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": "Experiment 1 refined AI-environment sweep",
                "regulation_formula": "Stim = Aff * exp[-alpha_conc * rho * (1 - M_prime)]",
                "arguments": vars(args),
                "model_params": model_params,
                "target_runs_by_A": {str(A): n for A, n in target_runs_by_A.items()},
                "statistics": {
                    "center": "mean across independent seeds",
                    "interval": "two-sided 95% Student-t confidence interval",
                    "convergence": "absolute difference between steady-window half means",
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    plot_main_curves(summary_df, fig_path, show_ci=True)
    plot_main_curves(summary_df, fig_mean_only_path, show_ci=False)

    print("\n================ Experiment Finished ================")
    print(f"Total elapsed time : {total_elapsed:.2f}s")
    print(f"Raw results        : {raw_path}")
    print(f"Summary results    : {summary_path}")
    print(f"Configuration      : {config_path}")
    print(f"CI figure bundle   : {os.path.splitext(fig_path)[0]}.[png|pdf|svg]")
    print(f"Mean figure bundle : {os.path.splitext(fig_mean_only_path)[0]}.[png|pdf|svg]")
    print("=====================================================")

    print_summary_and_diagnosis(summary_df, drift_tolerance=args.drift_tolerance)


if __name__ == "__main__":
    main()
