"""
Experiment 3 Supplement: Long-run Temporal Dynamics
===================================================

研究目的
--------
实验 3 的空间斑图展示了：

    a_i 扩散 -> x_i 退化或维持 -> D_i 积累或被抑制

本脚本用于补充实验 3 的长期宏观时间序列，量化 Payoff-only 与 Full Model
在相同初始状态、相同随机种子下的长期演化差异。

展示指标
--------
1. <a>(t): 平均 AI 使用强度
2. <x>(t): 平均质量保障投入
3. <D>(t): 平均 AI 诱发技术债

M 和债务陷阱比例仅作为内部诊断字段记录，不进入实验 3 图片或运行摘要。

实验设计
--------
- 默认 A = 0.4，与实验 3 正文斑图一致；
- 默认运行 100000 步，用于检查长期稳定性；
- 比较两个模型：
      1. Payoff-only
      2. Full Model

- 使用与实验 3 空间斑图完全相同的种子逻辑：
      common_seed = seed + int(round(A_env * 1000))

横坐标设计
----------
绘图时使用：

    x_axis = log10(t + 1)

但横轴标签仍显示真实演化时间 t。

默认横轴刻度：
    0, 10, 100, 1k, 10k, 50k, 100k

输出文件
--------
results_exp3_temporal_core_concentration_gate_A0.8/
    experiment_3_temporal_dynamics.csv
    fig_exp3_temporal_dynamics_A0.8.png/pdf

运行方式
--------
默认运行：

    python experiment_3_temporal_dynamics_100k.py

正式运行并保存 PDF：

    python experiment_3_temporal_dynamics_100k.py --save-pdf

快速测试：

    python experiment_3_temporal_dynamics_100k.py --grid-size 50 --steps 1000 --record-interval 10

如果希望和斑图完全对应，请保证以下参数一致：

    --grid-size
    --A-env
    --seed
    --init-D-max
"""

import os
import time
import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import cuda

from MODEL import AITechnicalDebtMDAE_GPU


# ==============================================================================
# 1. 基准参数：与实验 3 斑图保持一致
# ==============================================================================

def get_base_params(init_D_max: float = 0.02) -> Dict[str, float]:
    """
    与实验 1 当前配置、实验 3 空间斑图一致的参数。
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
        # 初始技术债参数
        # ------------------------------------------------------------------
        init_D_max=float(init_D_max),
    )


# ==============================================================================
# 2. 两个核心对照模型：与实验 3 斑图保持一致
# ==============================================================================

def get_model_configs() -> Dict[str, Dict[str, float]]:
    """
    两个核心对照组：

    1. Payoff-only:
       只根据收益学习，关闭记忆亲和度、浓度抑制和记忆清债。

    2. Full Model:
       完整 MDAE 机制。
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
    }


CORE_MODEL_ORDER = ["Payoff-only", "Full Model"]
FOUR_MODEL_ORDER = [
    "Payoff-only",
    "Payoff+MemoryAffinity",
    "Payoff+ConcentrationGate",
    "Full Model",
]

COLOR_MAP = {
    "Payoff-only": "#5C5C5C",
    "Payoff+MemoryAffinity": "#6F63C2",
    "Payoff+ConcentrationGate": "#2C9B52",
    "Full Model": "#1F77B4",
}

MODEL_LABELS = {
    "Payoff-only": "Payoff only",
    "Payoff+MemoryAffinity": "Memory affinity",
    "Payoff+ConcentrationGate": "Concentration gate",
    "Full Model": "Full model",
}

LINE_STYLES = {
    "Payoff-only": (0, (4, 2)),
    "Payoff+MemoryAffinity": (0, (1, 1)),
    "Payoff+ConcentrationGate": (0, (5, 1, 1, 1)),
    "Full Model": (0, (4, 2)),
}


def get_model_order(model_set: str) -> List[str]:
    if model_set == "core":
        return list(CORE_MODEL_ORDER)
    if model_set == "four":
        return list(FOUR_MODEL_ORDER)
    raise ValueError(f"Unknown model set: {model_set}")


# ==============================================================================
# 3. 绘图风格
# ==============================================================================

def set_publication_style() -> None:
    """设置论文风格。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.0,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_publication_figure(fig, output_base_path: str, save_tiff: bool = False) -> None:
    """Save an editable vector bundle plus a high-resolution preview."""
    fig.savefig(f"{output_base_path}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base_path}.svg", bbox_inches="tight")
    if save_tiff:
        fig.savefig(f"{output_base_path}.tiff", dpi=600, bbox_inches="tight")
    print(f"Saved figure bundle: {output_base_path}.[png|pdf|svg]")


# ==============================================================================
# 4. 工具函数：与实验 3 斑图保持一致
# ==============================================================================

def merge_params(
    base_params: Dict[str, float],
    override_params: Dict[str, float],
) -> Dict[str, float]:
    params = dict(base_params)
    params.update(override_params)
    return params


def format_A(A_env: float) -> str:
    return f"A{A_env:.1f}"


def random_array_with_exact_mean_half(
    grid_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    生成 [0,1] 随机矩阵，并保证系统总平均严格为 0.5。

    方法：
        成对生成 u 和 1-u；
        如果格点总数为奇数，最后补一个 0.5。
    """
    shape = (grid_size, grid_size)
    total_size = grid_size * grid_size
    half_size = total_size // 2

    first_half = rng.random(half_size, dtype=np.float64)

    values = np.empty(total_size, dtype=np.float64)
    values[:half_size] = first_half
    values[half_size:2 * half_size] = 1.0 - first_half

    if total_size % 2 == 1:
        values[-1] = 0.5

    rng.shuffle(values)
    return np.ascontiguousarray(values.reshape(shape))


def create_initial_state(
    grid_size: int,
    seed: int,
    init_D_max: float = 0.02,
) -> Dict[str, np.ndarray]:
    """
    创建公共初始状态。

    与实验 3 空间斑图一致：

    x_i(0), a_i(0):
        随机异质，但系统总平均严格为 0.5；

    D_i(0):
        默认 D_i(0) ~ U(0, 0.02)；

    M_i(0):
        中性可靠性记忆 M_i(0)=0.5。
    """
    if not (0.0 <= init_D_max <= 1.0):
        raise ValueError("init_D_max must be in [0, 1].")

    rng = np.random.default_rng(seed)
    shape = (grid_size, grid_size)

    init_x = random_array_with_exact_mean_half(grid_size, rng)
    init_a = random_array_with_exact_mean_half(grid_size, rng)

    if init_D_max == 0.0:
        init_D = np.zeros(shape, dtype=np.float64)
    else:
        init_D = rng.random(shape, dtype=np.float64) * init_D_max

    init_M = np.full(shape, 0.5, dtype=np.float64)

    return {
        "x": np.ascontiguousarray(init_x),
        "a": np.ascontiguousarray(init_a),
        "D": np.ascontiguousarray(init_D),
        "M": np.ascontiguousarray(init_M),
    }


def compute_debt_trap_fraction(
    state: Dict[str, np.ndarray],
    a_threshold: float = 0.8,
    x_threshold: float = 0.4,
    D_threshold: float = 0.3,
) -> float:
    """
    计算债务陷阱比例。

    债务陷阱定义：
        a_i > 0.8,
        x_i < 0.4,
        D_i > 0.3.
    """
    trap_mask = (
        (state["a"] > a_threshold)
        & (state["x"] < x_threshold)
        & (state["D"] > D_threshold)
    )
    return float(np.mean(trap_mask))


def compute_macro_record(
    state: Dict[str, np.ndarray],
    model_name: str,
    A_env: float,
    time_step: int,
) -> Dict[str, float]:
    """从当前状态计算宏观指标。"""
    return {
        "model": model_name,
        "A_env": float(A_env),
        "time": int(time_step),
        "mean_a": float(np.mean(state["a"])),
        "mean_x": float(np.mean(state["x"])),
        "mean_D": float(np.mean(state["D"])),
        "mean_M": float(np.mean(state["M"])),
        "trap_fraction": compute_debt_trap_fraction(state),
    }


def build_record_times(
    steps: int,
    record_interval: int,
    extra_times: List[int],
) -> List[int]:
    """
    构造记录时间点。

    默认每 record_interval 步记录一次，
    同时强制包含 0、steps 和额外时间点。
    """
    if record_interval <= 0:
        raise ValueError("record_interval must be positive.")

    times = set(range(0, steps + 1, record_interval))
    times.add(0)
    times.add(steps)

    for t in extra_times:
        if 0 <= t <= steps:
            times.add(int(t))

    return sorted(times)


# ==============================================================================
# 5. 单模型仿真
# ==============================================================================

def run_temporal_simulation(
    model_name: str,
    A_env: float,
    grid_size: int,
    steps: int,
    record_times: List[int],
    seed: int,
    init_state: Dict[str, np.ndarray],
    model_params: Dict[str, float],
) -> Tuple[List[Dict[str, float]], float]:
    """
    运行单个模型，并记录宏观时间序列。
    """
    model = AITechnicalDebtMDAE_GPU(
        grid_size=grid_size,
        seed=seed,
        A_env=A_env,
        init_x=init_state["x"],
        init_a=init_state["a"],
        init_D=init_state["D"],
        init_M=init_state["M"],
        **model_params,
    )

    record_set = set(record_times)
    records: List[Dict[str, float]] = []

    def record(t: int) -> None:
        state = model.get_state()
        record_data = compute_macro_record(
            state=state,
            model_name=model_name,
            A_env=A_env,
            time_step=t,
        )
        record_data.update({
            "seed": int(seed),
            "grid_size": int(grid_size),
            "w_payoff": float(model_params["w_payoff"]),
            "w_memory": float(model_params["w_memory"]),
            "alpha_conc": float(model_params["alpha_conc"]),
        })
        records.append(record_data)

    if 0 in record_set:
        record(0)

    start_time = time.time()

    progress_times = {
        1,
        2,
        5,
        10,
        100,
        1000,
        5000,
        10000,
        20000,
        50000,
        100000,
        steps,
    }

    for t in range(1, steps + 1):
        model.step(synchronize=False)

        if t in record_set:
            cuda.synchronize()
            record(t)

            if t in progress_times:
                latest = records[-1]
                print(
                    f"  recorded: model={model_name}, t={t}, "
                    f"<a>={latest['mean_a']:.4f}, "
                    f"<x>={latest['mean_x']:.4f}, "
                    f"<D>={latest['mean_D']:.4f}"
                )

    cuda.synchronize()
    elapsed = time.time() - start_time

    return records, elapsed


# ==============================================================================
# 6. 绘图函数：100000 步横坐标版本
# ==============================================================================

def time_to_axis_x(t):
    """
    将真实时间 t 映射到绘图坐标。

    使用 log10(t + 1)，好处是：
        1. 可以显示 t = 0；
        2. 早期动态不会被压缩；
        3. 可以手动控制刻度，避免长期仿真下标签重叠。
    """
    return np.log10(np.asarray(t, dtype=np.float64) + 1.0)


def _format_time_tick_label(t: int) -> str:
    """
    格式化横轴刻度标签。
    """
    if t >= 1000 and t % 1000 == 0:
        return f"{t // 1000}k"
    return str(t)


def _get_time_ticks_for_plot(steps: int) -> List[int]:
    """
    根据总步数自动选择横轴刻度。

    对 100000 步仿真，显示：
        0, 10, 100, 1k, 10k, 50k, 100k
    """
    if steps <= 100:
        base_ticks = [0, 10, steps]
    elif steps <= 1000:
        base_ticks = [0, 10, 100, steps]
    elif steps <= 10000:
        base_ticks = [0, 10, 100, 1000, steps]
    elif steps <= 30000:
        base_ticks = [0, 10, 100, 1000, 5000, steps]
    elif steps <= 100000:
        base_ticks = [0, 10, 100, 1000, 10000, 50000, steps]
    else:
        base_ticks = [0, 10, 100, 1000, 10000, 100000, steps]

    ticks = []
    for t in base_ticks:
        if 0 <= t <= steps and int(t) not in ticks:
            ticks.append(int(t))

    return ticks


def _apply_time_axis_style(ax, steps: int) -> None:
    """
    设置重新设计后的时间轴。

    横坐标实际绘制的是 log10(t + 1)，
    但标签仍然显示真实演化时间 t。
    """
    ticks = _get_time_ticks_for_plot(steps)
    tick_positions = time_to_axis_x(ticks)
    tick_labels = [_format_time_tick_label(t) for t in ticks]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    x_min = float(time_to_axis_x(0))
    x_max = float(time_to_axis_x(steps))
    span = x_max - x_min

    # 右侧留出少量空间，防止最后一个刻度标签贴边
    ax.set_xlim(x_min, x_max + 0.025 * span)

    ax.tick_params(axis="x", labelrotation=0)


def plot_temporal_dynamics(
    df: pd.DataFrame,
    A_env: float,
    steps: int,
    model_order: List[str],
    output_base_path: str,
    save_tiff: bool = False,
) -> None:
    """
    绘制宏观时间序列图：

        <a>(t), <x>(t), <D>(t)
    """
    metrics = [
        ("mean_a", r"AI use $\langle a\rangle$"),
        ("mean_x", r"Quality assurance $\langle x\rangle$"),
        ("mean_D", r"Technical debt $\langle D\rangle$"),
    ]

    # A slightly taller manuscript aspect keeps the three trajectories readable
    # while avoiding a short, width-bound strip on a portrait results page.
    fig_height = 3.50 if len(model_order) <= 3 else 3.75
    fig, axes = plt.subplots(1, 3, figsize=(7.2, fig_height), sharex=True)
    axes = np.asarray(axes).ravel()
    panel_labels = ["a", "b", "c"]
    legend_handles = []

    for panel_idx, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        for model_name in model_order:
            sub = df[df["model"] == model_name].sort_values("time")
            color = COLOR_MAP[model_name]

            time_values = sub["time"].values
            plot_x = time_to_axis_x(time_values)

            line, = ax.plot(
                plot_x,
                sub[metric].to_numpy(dtype=float),
                color=color,
                linestyle=LINE_STYLES[model_name],
                label=MODEL_LABELS[model_name],
                linewidth=1.8 if model_name == "Full Model" else 1.35,
                zorder=4 if model_name == "Full Model" else 2,
            )
            if panel_idx == 0:
                legend_handles.append(line)

        ax.set_xlabel(r"Evolution time $t$")
        ax.set_ylabel(ylabel)

        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        _apply_time_axis_style(ax, steps)

        ax.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
        ax.text(
            -0.14,
            1.05,
            panel_labels[panel_idx],
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            clip_on=False,
            ha="left",
        )

    fig.legend(
        handles=legend_handles,
        labels=[MODEL_LABELS[name] for name in model_order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=len(model_order) if len(model_order) <= 3 else 3,
        handlelength=2.8,
        columnspacing=1.3,
    )
    # fig.text(
    #     0.08,
    #     0.012,
    #     "Representative realization with one shared initial state and noise seed.",
    #     fontsize=5.8,
    #     color="#4D4D4D",
    #     ha="left",
    # )
    fig.tight_layout(rect=(0.04, 0.20, 1.0, 0.98), w_pad=1.0)
    save_publication_figure(fig, output_base_path, save_tiff=save_tiff)
    plt.close(fig)


# ==============================================================================
# 7. 主程序
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3 long-run temporal dynamics for Experiment 2-aligned model controls."
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Lattice size L. Default: 200.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Total simulation steps. Default: 50000.",
    )
    parser.add_argument(
        "--A-env",
        type=float,
        default=0.6,
        help="AI environment intensity. Default: 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260514,
        help="Random seed. Default is the same as Experiment 3 spatial patterns.",
    )
    parser.add_argument(
        "--init-D-max",
        type=float,
        default=0.02,
        help="Initial technical debt upper bound. Default: 0.02.",
    )
    parser.add_argument(
        "--record-interval",
        type=int,
        default=100,
        help="Record macro statistics every N steps. Default: 100.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_exp3_temporal_core_concentration_gate_A0.8",
        help="Output directory for the refined temporal experiment.",
    )
    parser.add_argument(
        "--model-set",
        choices=["core", "four"],
        default="core",
        help="core: main-text baseline/full comparison; four: supplementary trajectories.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Retained for compatibility; PDF and SVG are now always saved.",
    )
    parser.add_argument(
        "--save-tiff",
        action="store_true",
        help="Also save a 600-dpi TIFF figure.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_publication_style()

    if not (0.0 <= args.A_env <= 1.0):
        raise ValueError("A-env must be in [0, 1].")
    if not (0.0 <= args.init_D_max <= 1.0):
        raise ValueError("init-D-max must be in [0, 1].")
    if args.grid_size <= 0:
        raise ValueError("grid-size must be positive.")
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.record_interval <= 0:
        raise ValueError("record-interval must be positive.")

    # 强制包含实验 3 斑图和长期检查的关键时间点
    # 强制记录早期细节：
    # t = 0, 1, 2, ..., 10 每一步都记录；
    # 但横坐标刻度仍然只显示 0 和 10，不显示 1-9。
    early_dense_times = list(range(0, 11))

    extra_times = early_dense_times + [
        100,
        1000,
        5000,
        10000,
        20000,
        50000,
        args.steps,
    ]

    record_times = build_record_times(
        steps=args.steps,
        record_interval=args.record_interval,
        extra_times=extra_times,
    )

    base_params = get_base_params(init_D_max=args.init_D_max)
    model_configs = get_model_configs()
    model_order = get_model_order(args.model_set)

    print("\n================ Experiment 3: Long-run Temporal Dynamics ================")
    print(f"Grid size       : {args.grid_size} x {args.grid_size}")
    print(f"A_env           : {args.A_env}")
    print(f"Steps           : {args.steps}")
    print(f"Init D max      : {args.init_D_max}")
    print(f"Record interval : {args.record_interval}")
    print(f"Record points   : {len(record_times)}")
    print(f"Model set       : {args.model_set}")
    print(f"Models          : {model_order}")
    print(f"Output dir      : {args.output_dir}")
    print("Seed rule       : seed + int(round(A_env * 1000))")
    print("Time axis       : log10(t + 1), labels show real t")
    print("========================================================================\n")

    # 与实验 3 斑图完全一致的初始状态种子
    common_seed = args.seed + int(round(args.A_env * 1000))

    init_state = create_initial_state(
        grid_size=args.grid_size,
        seed=common_seed,
        init_D_max=args.init_D_max,
    )

    print("Initial state means:")
    print(f"  <a(0)> = {np.mean(init_state['a']):.6f}")
    print(f"  <x(0)> = {np.mean(init_state['x']):.6f}")
    print(f"  <D(0)> = {np.mean(init_state['D']):.6f}")
    print(f"  <M(0)> = {np.mean(init_state['M']):.6f}\n")

    all_records: List[Dict[str, float]] = []

    for model_name in model_order:
        print(f"Running model={model_name}, A={args.A_env:.2f} ...")

        params = merge_params(base_params, model_configs[model_name])

        # 与实验 3 斑图完全一致的模型随机种子
        model_seed = common_seed

        records, elapsed = run_temporal_simulation(
            model_name=model_name,
            A_env=args.A_env,
            grid_size=args.grid_size,
            steps=args.steps,
            record_times=record_times,
            seed=model_seed,
            init_state=init_state,
            model_params=params,
        )

        all_records.extend(records)

        final_record = records[-1]
        print(
            f"  final t={final_record['time']}: "
            f"<a>={final_record['mean_a']:.4f}, "
            f"<x>={final_record['mean_x']:.4f}, "
            f"<D>={final_record['mean_D']:.4f}, "
            f"elapsed={elapsed:.2f}s\n"
        )

    df = pd.DataFrame(all_records)
    df = df.sort_values(["model", "time"]).reset_index(drop=True)

    csv_path = os.path.join(args.output_dir, "experiment_3_temporal_dynamics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    config_path = os.path.join(args.output_dir, "experiment_3_temporal_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": "Experiment 3 refined temporal dynamics",
                "regulation_formula": "Stim = Aff * exp[-alpha_conc * rho * (1 - M_prime)]",
                "arguments": vars(args),
                "model_order": model_order,
                "base_params": base_params,
                "model_configs": {name: model_configs[name] for name in model_order},
                "figure_note": "Representative trajectory with a shared initial state and seed.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved config: {config_path}")

    # 输出时间序列主图
    output_base = os.path.join(
        args.output_dir,
        f"fig_exp3_temporal_dynamics_{format_A(args.A_env)}",
    )
    plot_temporal_dynamics(
        df=df,
        A_env=args.A_env,
        steps=args.steps,
        model_order=model_order,
        output_base_path=output_base,
        save_tiff=args.save_tiff,
    )

    print("\n================ Experiment 3 Long-run Temporal Dynamics Finished ================")
    print(f"Output dir: {args.output_dir}")
    print("Generated:")
    print(f"  {csv_path}")
    print(f"  fig_exp3_temporal_dynamics_{format_A(args.A_env)}.[png|pdf|svg]")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
