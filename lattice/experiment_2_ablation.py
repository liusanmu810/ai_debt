"""
Experiment 2: Ablation Study with A-scan and Clean Line Plots
==============================================================

研究目的
--------
在实验 1 已确认主线后，本实验通过机制消融检验：

    1. 可靠性记忆 Reliability Memory
    2. 浓度抑制 Concentration Suppression

是否能够在不同 AI 环境强度 A 下延缓质量保障崩塌、抑制技术债扩散、维持系统收益。

核心比较
--------
比较现有 4 组机制配置，并增加 1 组传统收益费米更新基准：

1. Payoff-only
   - 只根据收益学习
   - w_memory = 0
   - alpha_conc = 0

2. Payoff + Memory
   - 收益亲和度 + 可靠性记忆
   - alpha_conc = 0

3. Payoff + Concentration
   - 收益亲和度 + 浓度抑制
   - w_memory = 0

4. Full Model
   - 收益亲和度 + 可靠性记忆 + 浓度抑制

5. Payoff-Fermi
   - 随机选择一个邻居
   - 按收益差的传统费米概率决定是否模仿
   - 不使用可靠性代理或浓度门控进行候选选择

重要初始化设定
--------------
本实验依赖新版 MODEL.py 的初始化方式：

    x_i(0): 随机异质，但系统总平均严格为 0.5
    a_i(0): 随机异质，但系统总平均严格为 0.5
    D_i(0): 低初始技术债，默认 D_i(0) ~ U(0, 0.02)
    M_i(0): 中性可靠性记忆，M_i(0)=0.5

为保证机制消融和 A 扫描公平，同一个 run_id 下，所有模型配置和所有 A_env
使用相同 seed，从而获得配对的初始状态。由于确定性更新和费米更新消耗随机数的方式不同，
不能把后续随机扰动逐项视为完全相同；独立实验单位仍是每个完整模拟实现。

绘图说明
--------
正文主图默认不画标准差，只画均值折线，避免误差棒影响视觉效果。
如果需要附录图，可以使用 --plot-std 生成 mean ± std 阴影图。

运行方式
--------
默认正式运行：

    python experiment_2_ablation_scanA_lines_no_std.py

快速测试：

    python experiment_2_ablation_scanA_lines_no_std.py --grid-size 40 --steps 1000 --steady-window 100 --runs 2

正式稳定实验：

    python experiment_2_ablation_scanA_lines_no_std.py --grid-size 100 --steps 10000 --steady-window 1000 --runs 10

如果希望严格从无技术债开始：

    python experiment_2_ablation_scanA_lines_no_std.py --init-D-max 0.0

生成附录标准差阴影图：

    python experiment_2_ablation_scanA_lines_no_std.py --plot-std

仅补跑费米对照并合并已有四组结果：

    python "Experiment 2 Ablation Memory Concentration.py" --models Payoff-Fermi \
        --merge-raw results_exp2_four_group_concentration_gate/experiment_2_raw.csv

输出文件
--------
results_exp2_scanA/experiment_2_raw.csv
results_exp2_scanA/experiment_2_summary.csv
results_exp2_scanA/experiment_2_ablation_line_curves_no_std.png
results_exp2_scanA/experiment_2_debt_payoff_focus_no_std.png
results_exp2_scanA/experiment_2_debt_payoff_tradeoff.png

如果使用 --plot-std，还会输出：
results_exp2_scanA/experiment_2_ablation_line_curves_with_std.png

Fermi-augmented concentration-gate revision
--------------------------------------------
The formal model uses reliability-conditioned concentration regulation,
``rho * (1 - M')``. The ablation isolates memory affinity and this
concentration gate with a 2 x 2 mechanism design. Technical debt remains in
the debt dynamics, payoff, and reliability-memory update, but not in the gate.
The additional Payoff-Fermi group uses traditional pairwise payoff comparison
with K_F = 0.10. New runs are written to ``results_exp2_five_group_with_fermi`` so
earlier outputs remain untouched.
"""

import os
import time
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import cuda

from MODEL import AITechnicalDebtMDAE_GPU


# ==============================================================================
# 1. 基准参数：沿用实验 1 当前参数
# ==============================================================================


def get_base_params(init_D_max: float = 0.02) -> Dict[str, float]:
    """
    实验 1 当前参数。

    注意：
        - c_x 固定为 1，保持经典公共物品博弈中贡献成本规范化；
        - init_D_max 控制初始技术债上界：D_i(0) ~ U(0, init_D_max)。
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
        # 候选更新规则
        # ------------------------------------------------------------------
        selection_rule=0,  # 0: deterministic argmax; 1: payoff Fermi
        fermi_K=0.10,

        # ------------------------------------------------------------------
        # 自适应扰动与惯性策略更新参数
        # ------------------------------------------------------------------
        sigma_min=0.001,
        sigma_max=0.055,
        learning_rate=0.13,

        # ------------------------------------------------------------------
        # 初始状态参数
        # ------------------------------------------------------------------
        # D_i(0) ~ U(0, init_D_max)
        # x_i(0), a_i(0), M_i(0) 由 MODEL.py 按统一规则初始化。
        init_D_max=float(init_D_max),
    )


# ==============================================================================
# 2. 消融配置
# ==============================================================================


def get_ablation_configs() -> Dict[str, Dict[str, float]]:
    """
    四组机制配置。

    每个配置只覆盖 base_params 中需要修改的参数。
    """
    return {
        "Payoff-only": dict(
            w_payoff=1.00,
            w_memory=0.00,
            alpha_conc=0.00,
        ),
        "Payoff+MemoryAffinity": dict(
            w_payoff=0.88,
            w_memory=0.12,
            alpha_conc=0.00,
        ),
        "Payoff+ConcentrationGate": dict(
            w_payoff=1.00,
            w_memory=0.00,
            alpha_conc=0.18,
        ),
        "Full Model": dict(
            w_payoff=0.88,
            w_memory=0.12,
            alpha_conc=0.18,
        ),
        "Payoff-Fermi": dict(
            w_payoff=1.00,
            w_memory=0.00,
            alpha_conc=0.00,
            selection_rule=1,
        ),
    }


MODEL_ORDER = [
    "Payoff-only",
    "Payoff+MemoryAffinity",
    "Payoff+ConcentrationGate",
    "Full Model",
    "Payoff-Fermi",
]


# ==============================================================================
# 3. 参数合并与单次仿真
# ==============================================================================


def merge_params(base_params: Dict[str, float], override_params: Dict[str, float]) -> Dict[str, float]:
    params = dict(base_params)
    params.update(override_params)
    return params


def run_single_simulation(
    model_name: str,
    A_env: float,
    run_id: int,
    grid_size: int,
    steps: int,
    steady_window: int,
    seed_base: int,
    model_params: Dict[str, float],
) -> Dict[str, float]:
    """
    运行一个模型组别、一个 A 值、一个随机重复。

    同一个 run_id 下，所有模型配置和所有 A_env 使用相同 seed，以配对初始状态。
    不同更新规则会以不同方式消耗随机数，因此并不声称逐轮随机扰动完全相同。
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

    return {
        "model": model_name,
        "A_env": float(A_env),
        "run_id": int(run_id),
        "seed": int(seed),
        "grid_size": int(grid_size),
        "steps": int(steps),
        "steady_window": int(steady_window),
        "w_payoff": float(model_params["w_payoff"]),
        "w_memory": float(model_params["w_memory"]),
        "alpha_conc": float(model_params["alpha_conc"]),
        "selection_rule": int(model_params["selection_rule"]),
        "fermi_K": float(model_params["fermi_K"]),

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

        "elapsed_seconds": float(elapsed),
    }


# ==============================================================================
# 4. 汇总
# ==============================================================================


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """按 model 和 A_env 汇总多次独立重复。"""
    metrics = ["mean_x", "mean_a", "mean_D", "mean_M", "mean_payoff"]

    grouped = raw_df.groupby(["model", "A_env"])
    summary = grouped.size().reset_index(name="runs")

    metadata_cols = [
        "w_payoff",
        "w_memory",
        "alpha_conc",
        "selection_rule",
        "fermi_K",
    ]
    for col in metadata_cols:
        if col in raw_df.columns:
            metadata = grouped[col].first().reset_index()
            summary = summary.merge(metadata, on=["model", "A_env"], how="left")

    for metric in metrics:
        temp = grouped[metric].agg(["mean", "std"]).reset_index()
        temp = temp.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
            }
        )
        summary = summary.merge(temp, on=["model", "A_env"], how="left")

    summary = summary.fillna(0.0)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["model", "A_env"]).reset_index(drop=True)
    summary["model"] = summary["model"].astype(str)
    return summary


# ==============================================================================
# 5. 绘图函数
# ==============================================================================


def plot_ablation_line_curves_no_std(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    正文主图：只画均值折线，不画标准差。
    """
    metrics = [
        ("mean_x", "Quality assurance investment", r"$\langle x \rangle$"),
        ("mean_a", "AI usage intensity", r"$\langle a \rangle$"),
        ("mean_D", "AI-induced technical debt", r"$\langle D \rangle$"),
        ("mean_M", "Reliability memory", r"$\langle M \rangle$"),
        ("mean_payoff", "Average payoff", r"$\langle \Pi \rangle$"),
    ]

    plt.figure(figsize=(15, 9))

    for idx, (metric, title, ylabel) in enumerate(metrics, start=1):
        ax = plt.subplot(2, 3, idx)

        for model_name in MODEL_ORDER:
            subset = summary_df[summary_df["model"] == model_name].sort_values("A_env")
            if subset.empty:
                continue

            ax.plot(
                subset["A_env"].values,
                subset[f"{metric}_mean"].values,
                marker="o",
                linewidth=2.0,
                label=model_name,
            )

        ax.set_title(title)
        ax.set_xlabel("AI environment intensity A")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if idx == 1:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_ablation_line_curves_with_std(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    附录图：mean ± std 阴影带。
    使用阴影代替 error bar，视觉上更干净。
    """
    metrics = [
        ("mean_x", "Quality assurance investment", r"$\langle x \rangle$"),
        ("mean_a", "AI usage intensity", r"$\langle a \rangle$"),
        ("mean_D", "AI-induced technical debt", r"$\langle D \rangle$"),
        ("mean_M", "Reliability memory", r"$\langle M \rangle$"),
        ("mean_payoff", "Average payoff", r"$\langle \Pi \rangle$"),
    ]

    plt.figure(figsize=(15, 9))

    for idx, (metric, title, ylabel) in enumerate(metrics, start=1):
        ax = plt.subplot(2, 3, idx)

        for model_name in MODEL_ORDER:
            subset = summary_df[summary_df["model"] == model_name].sort_values("A_env")
            if subset.empty:
                continue

            A = subset["A_env"].values
            mean = subset[f"{metric}_mean"].values
            std = subset[f"{metric}_std"].values

            line = ax.plot(A, mean, marker="o", linewidth=2.0, label=model_name)[0]
            color = line.get_color()
            ax.fill_between(A, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("AI environment intensity A")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if idx == 1:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_debt_payoff_focus_no_std(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    正文辅助图：只突出技术债和收益。
    """
    metrics = [
        ("mean_D", "AI-induced technical debt", r"$\langle D \rangle$"),
        ("mean_payoff", "Average payoff", r"$\langle \Pi \rangle$"),
    ]

    plt.figure(figsize=(12, 5))

    for idx, (metric, title, ylabel) in enumerate(metrics, start=1):
        ax = plt.subplot(1, 2, idx)

        for model_name in MODEL_ORDER:
            subset = summary_df[summary_df["model"] == model_name].sort_values("A_env")
            if subset.empty:
                continue

            ax.plot(
                subset["A_env"].values,
                subset[f"{metric}_mean"].values,
                marker="o",
                linewidth=2.0,
                label=model_name,
            )

        ax.set_title(title)
        ax.set_xlabel("AI environment intensity A")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if idx == 1:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_debt_payoff_tradeoff(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    技术债-收益权衡图。
    """
    plt.figure(figsize=(8, 6))

    for model_name in MODEL_ORDER:
        subset = summary_df[summary_df["model"] == model_name].sort_values("A_env")
        if subset.empty:
            continue

        plt.plot(
            subset["mean_D_mean"].values,
            subset["mean_payoff_mean"].values,
            marker="o",
            linewidth=2.0,
            label=model_name,
        )

    plt.xlabel(r"Mean technical debt $\langle D \rangle$")
    plt.ylabel(r"Mean payoff $\langle \Pi \rangle$")
    plt.title("Debt-payoff tradeoff under different mechanisms")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ==============================================================================
# 6. 打印诊断
# ==============================================================================


def print_summary_and_diagnosis(summary_df: pd.DataFrame) -> None:
    print("\n================ Experiment 2 Summary ================")
    print(summary_df.to_string(index=False))

    print("\n================ Quick Diagnosis ================")

    for model_name in MODEL_ORDER:
        subset = summary_df[summary_df["model"] == model_name].sort_values("A_env")
        if subset.empty:
            continue

        first = subset.iloc[0]
        last = subset.iloc[-1]

        print(f"\nModel: {model_name}")
        print(f"  Delta <a>  : {last['mean_a_mean'] - first['mean_a_mean']:.4f}")
        print(f"  Delta <x>  : {last['mean_x_mean'] - first['mean_x_mean']:.4f}")
        print(f"  Delta <D>  : {last['mean_D_mean'] - first['mean_D_mean']:.4f}")
        print(f"  Delta <M>  : {last['mean_M_mean'] - first['mean_M_mean']:.4f}")
        print(f"  Delta <Pi> : {last['mean_payoff_mean'] - first['mean_payoff_mean']:.4f}")

    high_A = summary_df[summary_df["A_env"] == summary_df["A_env"].max()].copy()
    if not high_A.empty:
        min_debt_row = high_A.loc[high_A["mean_D_mean"].idxmin()]
        max_payoff_row = high_A.loc[high_A["mean_payoff_mean"].idxmax()]
        print("\nAt highest A:")
        print(f"  Lowest debt   : {min_debt_row['model']} (<D>={min_debt_row['mean_D_mean']:.4f})")
        print(f"  Highest payoff: {max_payoff_row['model']} (<Pi>={max_payoff_row['mean_payoff_mean']:.4f})")

    print("===================================================")


# ==============================================================================
# 7. 主程序
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: ablation study with A scan and clean line plots."
    )

    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--steady-window", type=int, default=500)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260514)
    parser.add_argument(
        "--fermi-K",
        type=float,
        default=0.10,
        help="Decision-noise scale K_F for the payoff-based Fermi comparator.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Model groups to run. Use --models Payoff-Fermi to add only the new comparator.",
    )
    parser.add_argument(
        "--merge-raw",
        type=str,
        default=None,
        help="Optional existing Experiment 2 raw CSV to merge with newly simulated groups.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_exp2_five_group_with_fermi",
    )
    parser.add_argument("--plot-std", action="store_true", help="Also generate mean ± std shaded appendix figure.")
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
        default=[round(x, 2) for x in np.linspace(0.0, 1.0, 11)],
        help="AI environment intensity values. Default: 0.0 0.1 ... 1.0",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.steady_window <= 0:
        raise ValueError("steady-window must be positive.")
    if args.steady_window > args.steps:
        raise ValueError("steady-window cannot be larger than steps.")
    if not (0.0 <= args.init_D_max <= 1.0):
        raise ValueError("init-D-max must be in [0, 1].")
    if args.fermi_K <= 0.0:
        raise ValueError("fermi-K must be positive.")

    base_params = get_base_params(init_D_max=args.init_D_max)
    base_params["fermi_K"] = float(args.fermi_K)
    ablation_configs = get_ablation_configs()

    print("\n================ Experiment 2: Ablation with A-scan ================")
    print(f"Grid size      : {args.grid_size} x {args.grid_size}")
    print(f"Steps          : {args.steps}")
    print(f"Steady window  : {args.steady_window}")
    print(f"Runs per group : {args.runs}")
    print(f"Seed base      : {args.seed_base}")
    print(f"Init D max     : {args.init_D_max}")
    print(f"A values       : {args.A_values}")
    print(f"Models         : {args.models}")
    print(f"Fermi K        : {args.fermi_K}")
    print(f"Merge raw      : {args.merge_raw}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Plot std       : {args.plot_std}")
    print("===================================================================\n")

    all_results = []
    total_start = time.time()

    for run_id in range(args.runs):
        print(f"\n---------- Initial condition / run {run_id + 1}/{args.runs} ----------")
        for model_name in args.models:
            override_params = ablation_configs[model_name]
            params = merge_params(base_params, override_params)

            for A_env in args.A_values:
                print(f"Running model={model_name}, A={A_env:.2f}, run={run_id + 1}/{args.runs} ...")

                result = run_single_simulation(
                    model_name=model_name,
                    A_env=A_env,
                    run_id=run_id,
                    grid_size=args.grid_size,
                    steps=args.steps,
                    steady_window=args.steady_window,
                    seed_base=args.seed_base,
                    model_params=params,
                )
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
    if args.merge_raw is not None:
        existing_raw = pd.read_csv(args.merge_raw)
        raw_df = pd.concat([existing_raw, raw_df], ignore_index=True, sort=False)
        raw_df = raw_df.drop_duplicates(
            subset=["model", "A_env", "run_id"],
            keep="last",
        ).reset_index(drop=True)
    summary_df = summarize_results(raw_df)

    raw_path = os.path.join(args.output_dir, "experiment_2_raw.csv")
    summary_path = os.path.join(args.output_dir, "experiment_2_summary.csv")
    line_no_std_path = os.path.join(args.output_dir, "experiment_2_ablation_line_curves_no_std.png")
    focus_path = os.path.join(args.output_dir, "experiment_2_debt_payoff_focus_no_std.png")
    tradeoff_path = os.path.join(args.output_dir, "experiment_2_debt_payoff_tradeoff.png")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_ablation_line_curves_no_std(summary_df, line_no_std_path)
    plot_debt_payoff_focus_no_std(summary_df, focus_path)
    plot_debt_payoff_tradeoff(summary_df, tradeoff_path)

    std_path = None
    if args.plot_std:
        std_path = os.path.join(args.output_dir, "experiment_2_ablation_line_curves_with_std.png")
        plot_ablation_line_curves_with_std(summary_df, std_path)

    print("\n================ Experiment Finished ================")
    print(f"Total elapsed time : {total_elapsed:.2f}s")
    print(f"Raw results        : {raw_path}")
    print(f"Summary results    : {summary_path}")
    print(f"Main line figure   : {line_no_std_path}")
    print(f"Debt/payoff focus  : {focus_path}")
    print(f"Tradeoff figure    : {tradeoff_path}")
    if std_path is not None:
        print(f"Std appendix figure: {std_path}")
    print("=====================================================")

    print_summary_and_diagnosis(summary_df)


if __name__ == "__main__":
    main()
