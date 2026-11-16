"""
Experiment 3: Core Spatial Patterns of AI-induced Technical Debt
=================================================================

研究目的
--------
实验 1 已经说明：

    A ↑ -> <a> ↑ -> <x> ↓ -> <D> ↑

实验 2 已经说明：

    Full Model 相比消融模型能够延缓质量保障崩塌与技术债扩散。

实验 3 的任务不是再次做全面消融，而是用空间斑图解释主线机制：

    AI 使用强度 a_i 如何在空间中扩散？
    质量保障投入 x_i 是否随之退化？
    技术债 D_i 是否在空间中累积？
    Full Model 是否能相比 Payoff-only 抑制这种债务扩散？

实验设计
--------
- 正文默认使用临界 AI 环境：

      A = 0.4

  原因：
      A = 0.4 附近更容易观察到 Payoff-only 与 Full Model 的空间分化。
      Payoff-only 倾向于形成低质量保障和高技术债区域；
      Full Model 倾向于维持高质量保障并抑制技术债扩散。

- 补充材料可使用高 AI 环境：

      A = 0.8

- 只比较两个模型：

      1. Payoff-only
      2. Full Model

- 展示三个核心变量：

      1. a_i(t): AI usage intensity
      2. x_i(t): quality assurance investment
      3. D_i(t): AI-induced technical debt

- 默认时间点：

      t = 0, 10, 100, 1000, 10000

图像结构
--------
每张图：
    - 2 行：Payoff-only, Full Model
    - 5 列：t = 0, 10, 100, 1000, 10000

输出三张正文主图：
    1. fig_exp3_core_a_A0.4.png
    2. fig_exp3_core_x_A0.4.png
    3. fig_exp3_core_D_A0.4.png

重要初始化设定
--------------
两个模型使用完全相同的初始状态和随机扰动种子。
因此两组斑图差异主要来自机制参数，而不是随机性差异。

初始状态为：
    x_i(0): 随机异质，但系统总平均严格为 0.5
    a_i(0): 随机异质，但系统总平均严格为 0.5
    D_i(0): 低初始技术债，默认 D_i(0) ~ U(0, 0.02)
    M_i(0): 中性可靠性记忆，M_i(0)=0.5

运行方式
--------
默认正式运行：

    python experiment_3_spatial_core_patterns.py

快速测试：

    python experiment_3_spatial_core_patterns.py --grid-size 50 --steps 1000 --snapshot-times 0 10 100 1000

正式运行并保存 PDF：

    python experiment_3_spatial_core_patterns.py --grid-size 200 --steps 10000 --snapshot-times 0 10 100 1000 10000 --save-pdf

补充材料高 AI 环境：

    python experiment_3_spatial_core_patterns.py --A-env 0.8 --output-dir results_exp3_spatial_core_concentration_gate_A0.8 --save-pdf

严格从无技术债开始：

    python experiment_3_spatial_core_patterns.py --init-D-max 0.0

输出目录
--------
results_exp3_core/
    experiment_3_core_summary.csv
    snapshots_A0.4_Payoff_only.npz
    snapshots_A0.4_Full_Model.npz
    fig_exp3_core_a_A0.4.png/pdf
    fig_exp3_core_x_A0.4.png/pdf
    fig_exp3_core_D_A0.4.png/pdf
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
from matplotlib.colors import LinearSegmentedColormap
from numba import cuda

from MODEL import AITechnicalDebtMDAE_GPU


# ==============================================================================
# 1. 基准参数：实验 1 / 实验 2 当前配置
# ==============================================================================

def get_base_params(init_D_max: float = 0.02) -> Dict[str, float]:
    """
    实验 1 与实验 2 当前使用的参数。

    注意：
        - c_x 固定为 1，保持经典公共物品博弈贡献成本规范化；
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
# 2. 两个核心对照模型
# ==============================================================================

def get_model_configs() -> Dict[str, Dict[str, float]]:
    """
    实验 3 只保留两个核心对照组：

        1. Payoff-only:
           只根据收益学习。
           关闭可靠性记忆亲和度、浓度抑制和记忆清债项。

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

MODEL_LABELS = {
    "Payoff-only": "Payoff only",
    "Payoff+MemoryAffinity": "Memory affinity",
    "Payoff+ConcentrationGate": "Concentration gate",
    "Full Model": "Full model",
}

MODEL_COLORS = {
    "Payoff-only": "#5C5C5C",
    "Payoff+MemoryAffinity": "#6F63C2",
    "Payoff+ConcentrationGate": "#2C9B52",
    "Full Model": "#1F77B4",
}


def get_model_order(model_set: str) -> List[str]:
    if model_set == "core":
        return list(CORE_MODEL_ORDER)
    if model_set == "four":
        return list(FOUR_MODEL_ORDER)
    raise ValueError(f"Unknown model set: {model_set}")

# 实验 3 的空间机制链条：
# a_i 扩散 -> x_i 退化或维持 -> D_i 积累或被抑制
VARIABLE_ORDER = ["a", "x", "D"]

VARIABLE_LABELS = {
    "a": r"AI usage intensity $a_i$",
    "x": r"Quality assurance $x_i$",
    "D": r"Technical debt $D_i$",
}


# ==============================================================================
# 3. 论文风格与高对比配色
# ==============================================================================

def set_publication_style() -> None:
    """设置适合论文正文的简洁风格。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_high_contrast_colormaps() -> Dict[str, LinearSegmentedColormap]:
    """
    三个核心变量使用不同色系：

    a: 红色系，表示 AI 使用强度；
    x: 青绿-蓝色系，表示质量保障投入；
    D: 黄-橙-红-紫色系，表示技术债风险。
    """
    return {
        "a": LinearSegmentedColormap.from_list(
            "ai_usage_purple",
            [
                "#FCFBFD",
                "#EFEDF5",
                "#DADAEB",
                "#9E9AC8",
                "#756BB1",
                "#54278F",
            ],
        ),
        "x": LinearSegmentedColormap.from_list(
            "qa_teal",
            [
                "#F7FCFD",
                "#E5F5F9",
                "#99D8C9",
                "#41AE76",
                "#238B45",
                "#005824",
            ],
        ),
        "D": LinearSegmentedColormap.from_list(
            "debt_orange_red",
            [
                "#FFF7EC",
                "#FEE8C8",
                "#FDBB84",
                "#FC8D59",
                "#D7301F",
                "#7F0000",
            ],
        ),
    }


def save_publication_figure(fig, output_base_path: str, save_tiff: bool = False) -> None:
    """Save an editable vector bundle plus a high-resolution preview."""
    fig.savefig(f"{output_base_path}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base_path}.svg", bbox_inches="tight")
    if save_tiff:
        fig.savefig(f"{output_base_path}.tiff", dpi=600, bbox_inches="tight")
    print(f"Saved figure bundle: {output_base_path}.[png|pdf|svg]")


# ==============================================================================
# 4. 工具函数
# ==============================================================================

def merge_params(
    base_params: Dict[str, float],
    override_params: Dict[str, float],
) -> Dict[str, float]:
    params = dict(base_params)
    params.update(override_params)
    return params


def sanitize_model_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("+", "plus")


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

    两个模型使用相同初始状态，保证空间对比公平。

    x_i(0), a_i(0):
        随机异质，但系统总平均严格为 0.5；

    D_i(0):
        低初始技术债，默认 D_i(0) ~ U(0, 0.02)；

    M_i(0):
        中性可靠性记忆，固定为 0.5。
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


def compute_state_means(state: Dict[str, np.ndarray]) -> Dict[str, float]:
    """计算当前状态的系统平均值。"""
    return {
        "mean_x": float(np.mean(state["x"])),
        "mean_a": float(np.mean(state["a"])),
        "mean_D": float(np.mean(state["D"])),
        "mean_M": float(np.mean(state["M"])),
    }


def compute_debt_trap_fraction(
    state: Dict[str, np.ndarray],
    a_threshold: float = 0.8,
    x_threshold: float = 0.4,
    D_threshold: float = 0.3,
) -> float:
    """
    计算债务陷阱比例。

    债务陷阱定义为：
        a_i > 0.8, x_i < 0.4, D_i > 0.3

    该指标不是绘图必需，但有助于后续写论文和扩展实验。
    """
    trap = (
        (state["a"] > a_threshold)
        & (state["x"] < x_threshold)
        & (state["D"] > D_threshold)
    )
    return float(np.mean(trap))


# ==============================================================================
# 5. 仿真并记录空间快照
# ==============================================================================

def run_snapshot_simulation(
    model_name: str,
    A_env: float,
    grid_size: int,
    steps: int,
    snapshot_times: List[int],
    seed: int,
    init_state: Dict[str, np.ndarray],
    model_params: Dict[str, float],
) -> Tuple[Dict[str, Dict[int, np.ndarray]], List[Dict[str, float]], float]:
    """
    运行单个模型，并在指定时间点记录 a、x 和 D 的空间快照。
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

    snapshot_set = set(snapshot_times)
    snapshots = {var: {} for var in VARIABLE_ORDER}
    summary_records: List[Dict[str, float]] = []

    def record_snapshot(t: int) -> None:
        state = model.get_state()

        for var in VARIABLE_ORDER:
            snapshots[var][t] = np.asarray(state[var], dtype=np.float64)

        means = compute_state_means(state)
        trap_fraction = compute_debt_trap_fraction(state)

        summary_records.append({
            "model": model_name,
            "A_env": float(A_env),
            "time": int(t),
            "seed": int(seed),
            "grid_size": int(grid_size),
            "w_payoff": float(model_params["w_payoff"]),
            "w_memory": float(model_params["w_memory"]),
            "alpha_conc": float(model_params["alpha_conc"]),
            **means,
            "trap_fraction": trap_fraction,
        })

    if 0 in snapshot_set:
        record_snapshot(0)

    start_time = time.time()

    for t in range(1, steps + 1):
        model.step(synchronize=False)

        if t in snapshot_set:
            cuda.synchronize()
            record_snapshot(t)
            print(f"  snapshot recorded: model={model_name}, A={A_env:.2f}, t={t}")

    cuda.synchronize()
    elapsed = time.time() - start_time

    return snapshots, summary_records, elapsed


def save_snapshots_npz(
    snapshots: Dict[str, Dict[int, np.ndarray]],
    output_path: str,
    A_env: float,
    model_name: str,
    snapshot_times: List[int],
    seed: int,
    grid_size: int,
    model_params: Dict[str, float],
) -> None:
    """
    保存空间快照为 npz。
    """
    arrays = {}

    for var, time_dict in snapshots.items():
        for t, arr in time_dict.items():
            arrays[f"{var}_t{t}"] = arr

    arrays["A_env"] = np.array(A_env, dtype=np.float64)
    arrays["model_name"] = np.array(model_name)
    arrays["snapshot_times"] = np.asarray(snapshot_times, dtype=np.int64)
    arrays["variables"] = np.asarray(VARIABLE_ORDER)
    arrays["seed"] = np.array(seed, dtype=np.int64)
    arrays["grid_size"] = np.array(grid_size, dtype=np.int64)
    for key in ("w_payoff", "w_memory", "alpha_conc"):
        arrays[key] = np.array(model_params[key], dtype=np.float64)

    np.savez_compressed(output_path, **arrays)
    print(f"Saved snapshots: {output_path}")


# ==============================================================================
# 6. 绘图：单变量，2 行 × T 列
# ==============================================================================

def plot_variable_snapshots(
    all_snapshots: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    variable: str,
    A_env: float,
    snapshot_times: List[int],
    model_order: List[str],
    output_base_path: str,
    save_tiff: bool = False,
) -> None:
    """
    绘制一个变量的空间演化斑图。

    图像结构：
        行：Payoff-only, Full Model
        列：指定 snapshot_times
    """
    cmaps = make_high_contrast_colormaps()
    cmap = cmaps[variable]

    n_rows = len(model_order)
    n_cols = len(snapshot_times)
    fig_height = max(2.8, 0.96 * n_rows + 0.75)
    fig = plt.figure(figsize=(7.2, fig_height))
    grid = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.055],
        wspace=0.045,
        hspace=0.10,
        left=0.13,
        right=0.94,
        bottom=0.08,
        top=0.86,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)

    im = None

    for i, model_name in enumerate(model_order):
        for j, t in enumerate(snapshot_times):
            ax = fig.add_subplot(grid[i, j])
            axes[i, j] = ax

            if t not in all_snapshots[model_name][variable]:
                raise KeyError(
                    f"Missing snapshot: model={model_name}, variable={variable}, time={t}"
                )

            arr = all_snapshots[model_name][variable][t]

            im = ax.imshow(
                arr,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(f"t = {t:,}", pad=4, fontsize=7.5)

            if j == 0:
                ax.set_ylabel(
                    MODEL_LABELS.get(model_name, model_name),
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                    fontsize=7,
                    fontweight="bold" if model_name == "Full Model" else "normal",
                    color=MODEL_COLORS[model_name],
                )

            # 细边框，增强斑图边界感
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_edgecolor("#B0B0B0")

    colorbar_axis = fig.add_subplot(grid[:, -1])
    cbar = fig.colorbar(
        im,
        cax=colorbar_axis,
    )
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.outline.set_linewidth(0.5)

    # fig.text(
    #     0.13,
    #     0.025,
    #     "Representative snapshot; all models share the same initial state and noise seed.",
    #     fontsize=5.8,
    #     color="#4D4D4D",
    #     ha="left",
    # )

    save_publication_figure(fig, output_base_path, save_tiff=save_tiff)
    plt.close(fig)


def plot_final_state_overview(
    all_snapshots: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    A_env: float,
    final_time: int,
    model_order: List[str],
    output_base_path: str,
    save_tiff: bool = False,
) -> None:
    """Create the main image plate: models by variables at the final time."""
    cmaps = make_high_contrast_colormaps()
    n_rows = len(model_order)
    n_cols = len(VARIABLE_ORDER)
    fig_height = max(3.0, 1.25 * n_rows + 1.0)
    fig = plt.figure(figsize=(7.2, fig_height))
    grid = fig.add_gridspec(
        n_rows + 1,
        n_cols,
        height_ratios=[1.0] * n_rows + [0.065],
        hspace=0.18,
        wspace=0.10,
        left=0.13,
        right=0.97,
        bottom=0.13,
        top=0.85,
    )

    images = {}
    for row, model_name in enumerate(model_order):
        for col, variable in enumerate(VARIABLE_ORDER):
            ax = fig.add_subplot(grid[row, col])
            arr = all_snapshots[model_name][variable][final_time]
            images[variable] = ax.imshow(
                arr,
                cmap=cmaps[variable],
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_edgecolor("#B0B0B0")

            if row == 0:
                ax.set_title(VARIABLE_LABELS[variable], fontsize=7.5, pad=4)
            if col == 0:
                ax.set_ylabel(
                    MODEL_LABELS.get(model_name, model_name),
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                    fontsize=7,
                    fontweight="bold" if model_name == "Full Model" else "normal",
                    color=MODEL_COLORS[model_name],
                )

            ax.text(
                0.03,
                0.96,
                f"mean = {np.mean(arr):.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.8,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            )

    for col, variable in enumerate(VARIABLE_ORDER):
        colorbar_axis = fig.add_subplot(grid[-1, col])
        cbar = fig.colorbar(images[variable], cax=colorbar_axis, orientation="horizontal")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=6, length=2, pad=1)
        cbar.outline.set_linewidth(0.5)

    save_publication_figure(fig, output_base_path, save_tiff=save_tiff)
    plt.close(fig)


# ==============================================================================
# 7. 主程序
# ==============================================================================

def plot_spatial_evolution_overview(
    all_snapshots: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    A_env: float,
    snapshot_times: List[int],
    model_order: List[str],
    output_base_path: str,
    save_tiff: bool = False,
) -> None:
    """Create one manuscript plate showing the full spatial evolution of a, x, and D.

    Each variable forms a two-row block (payoff only, full model), while columns
    advance through common snapshot times. Fixed [0, 1] scales make cross-model
    and cross-time comparisons visually valid.
    """
    del A_env  # Kept in the signature for consistency with the other plotters.
    cmaps = make_high_contrast_colormaps()
    n_models = len(model_order)
    n_times = len(snapshot_times)
    n_rows = len(VARIABLE_ORDER) * n_models

    fig_height = max(6.6, 1.03 * n_rows + 0.65)
    fig = plt.figure(figsize=(7.2, fig_height))
    grid = fig.add_gridspec(
        n_rows,
        n_times + 1,
        width_ratios=[1.0] * n_times + [0.045],
        wspace=0.045,
        hspace=0.075,
        left=0.165,
        right=0.955,
        bottom=0.025,
        top=0.955,
    )

    axes = np.empty((n_rows, n_times), dtype=object)
    block_images: Dict[str, mpl.image.AxesImage] = {}

    for variable_index, variable in enumerate(VARIABLE_ORDER):
        row_start = variable_index * n_models
        for model_index, model_name in enumerate(model_order):
            row = row_start + model_index
            for time_index, t in enumerate(snapshot_times):
                if t not in all_snapshots[model_name][variable]:
                    raise KeyError(
                        f"Missing snapshot: model={model_name}, variable={variable}, time={t}"
                    )

                ax = fig.add_subplot(grid[row, time_index])
                axes[row, time_index] = ax
                image = ax.imshow(
                    all_snapshots[model_name][variable][t],
                    cmap=cmaps[variable],
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                    rasterized=True,
                )
                block_images[variable] = image
                ax.set_xticks([])
                ax.set_yticks([])

                for spine in ax.spines.values():
                    spine.set_linewidth(0.42)
                    spine.set_edgecolor("#B7B7B7")

                if row == 0:
                    ax.set_title(f"$t={t:,}$", fontsize=7.2, pad=3.5)

                if time_index == 0:
                    ax.set_ylabel(
                        MODEL_LABELS.get(model_name, model_name),
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=7,
                        fontsize=6.6,
                        fontweight="bold" if model_name == "Full Model" else "normal",
                        color=MODEL_COLORS[model_name],
                    )

        colorbar_axis = fig.add_subplot(grid[row_start:row_start + n_models, -1])
        cbar = fig.colorbar(block_images[variable], cax=colorbar_axis)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=7, length=2, pad=2)
        box = cbar.ax.get_position()
        inset = 6.0 / (fig.get_figheight() * 72.0)
        cbar.ax.set_position([box.x0, box.y0 + inset, box.width, box.height - 2 * inset])
        cbar.outline.set_linewidth(0.45)

    fig.canvas.draw()
    panel_letters = ["a", "b", "c"]
    for variable_index, variable in enumerate(VARIABLE_ORDER):
        row_start = variable_index * n_models
        block_top = axes[row_start, 0].get_position().y1
        block_bottom = axes[row_start + n_models - 1, 0].get_position().y0
        block_center = 0.5 * (block_top + block_bottom)
        fig.text(
            0.018,
            block_top,
            panel_letters[variable_index],
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
        )
        fig.text(
            0.043,
            block_center,
            VARIABLE_LABELS[variable],
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.8,
        )

    save_publication_figure(fig, output_base_path, save_tiff=save_tiff)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: core spatial patterns for Experiment 2-aligned model controls."
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=200,
        help="Lattice size L. The system size is L x L. Default: 200.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total simulation steps. Default: 20000.",
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
        help="Random seed. Default: 20260514.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_exp3_spatial_core_concentration_gate_A0.8",
        help="Output directory for the refined spatial experiment.",
    )
    parser.add_argument(
        "--model-set",
        choices=["core", "four"],
        default="core",
        help="core: main-text baseline/full comparison; four: supplementary mechanism plate.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Retained for compatibility; PDF and SVG are now always saved.",
    )
    parser.add_argument(
        "--save-tiff",
        action="store_true",
        help="Also save 600-dpi TIFF figures.",
    )
    parser.add_argument(
        "--init-D-max",
        type=float,
        default=0.02,
        help="Initial technical debt upper bound. D_i(0) ~ U(0, init_D_max).",
    )
    parser.add_argument(
        "--snapshot-times",
        type=int,
        nargs="+",
        default=[0, 100, 1000, 20000,100000],
        help="Snapshot times. Default: 0 10 100 1000 20000.",
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

    snapshot_times = sorted(set(args.snapshot_times))

    if min(snapshot_times) < 0:
        raise ValueError("snapshot-times must be non-negative.")
    if max(snapshot_times) > args.steps:
        raise ValueError("Maximum snapshot time cannot exceed total steps.")

    base_params = get_base_params(init_D_max=args.init_D_max)
    model_configs = get_model_configs()
    model_order = get_model_order(args.model_set)

    print("\n================ Experiment 3: Core Spatial Patterns ================")
    print(f"Grid size      : {args.grid_size} x {args.grid_size}")
    print(f"A_env          : {args.A_env}")
    print(f"Steps          : {args.steps}")
    print(f"Init D max     : {args.init_D_max}")
    print(f"Snapshot times : {snapshot_times}")
    print(f"Model set      : {args.model_set}")
    print(f"Models         : {model_order}")
    print(f"Variables      : {VARIABLE_ORDER}")
    print(f"Output dir     : {args.output_dir}")
    print("====================================================================\n")

    # 两个模型使用相同初始状态。
    init_seed = args.seed + int(round(args.A_env * 1000))
    init_state = create_initial_state(
        grid_size=args.grid_size,
        seed=init_seed,
        init_D_max=args.init_D_max,
    )

    print("Initial state means:")
    print(f"  <a(0)> = {np.mean(init_state['a']):.6f}")
    print(f"  <x(0)> = {np.mean(init_state['x']):.6f}")
    print(f"  <D(0)> = {np.mean(init_state['D']):.6f}")
    print(f"  <M(0)> = {np.mean(init_state['M']):.6f}\n")

    all_snapshots: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    all_summary_records: List[Dict[str, float]] = []

    for model_name in model_order:
        print(f"Running model={model_name}, A={args.A_env:.2f} ...")

        params = merge_params(base_params, model_configs[model_name])

        # 两个模型使用完全相同的随机扰动种子。
        # 初始状态也完全相同，因此空间差异主要来自机制参数本身。
        model_seed = args.seed + int(round(args.A_env * 1000))

        snapshots, summary_records, elapsed = run_snapshot_simulation(
            model_name=model_name,
            A_env=args.A_env,
            grid_size=args.grid_size,
            steps=args.steps,
            snapshot_times=snapshot_times,
            seed=model_seed,
            init_state=init_state,
            model_params=params,
        )

        all_snapshots[model_name] = snapshots
        all_summary_records.extend(summary_records)

        npz_path = os.path.join(
            args.output_dir,
            f"snapshots_{format_A(args.A_env)}_{sanitize_model_name(model_name)}.npz",
        )
        save_snapshots_npz(
            snapshots=snapshots,
            output_path=npz_path,
            A_env=args.A_env,
            model_name=model_name,
            snapshot_times=snapshot_times,
            seed=model_seed,
            grid_size=args.grid_size,
            model_params=params,
        )

        final_time = max(snapshot_times)
        final_records = [r for r in summary_records if r["time"] == final_time]
        if final_records:
            final_record = final_records[0]
            print(
                f"  final t={final_time}: "
                f"<a>={final_record['mean_a']:.4f}, "
                f"<x>={final_record['mean_x']:.4f}, "
                f"<D>={final_record['mean_D']:.4f}, "
                f"elapsed={elapsed:.2f}s"
            )
        print()

    # 绘制 a、x、D 三张核心正文图
    overview_base = os.path.join(
        args.output_dir,
        f"fig_exp3_{args.model_set}_spatial_evolution_{format_A(args.A_env)}",
    )
    plot_spatial_evolution_overview(
        all_snapshots=all_snapshots,
        A_env=args.A_env,
        snapshot_times=snapshot_times,
        model_order=model_order,
        output_base_path=overview_base,
        save_tiff=args.save_tiff,
    )

    for variable in VARIABLE_ORDER:
        output_base_path = os.path.join(
            args.output_dir,
            f"fig_exp3_{args.model_set}_{variable}_{format_A(args.A_env)}",
        )

        plot_variable_snapshots(
            all_snapshots=all_snapshots,
            variable=variable,
            A_env=args.A_env,
            snapshot_times=snapshot_times,
            model_order=model_order,
            output_base_path=output_base_path,
            save_tiff=args.save_tiff,
        )

    # 保存汇总数据
    summary_df = pd.DataFrame(all_summary_records)
    summary_path = os.path.join(
        args.output_dir,
        f"experiment_3_{args.model_set}_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    config_path = os.path.join(args.output_dir, "experiment_3_spatial_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": "Experiment 3 refined spatial snapshots",
                "regulation_formula": "Stim = Aff * exp[-alpha_conc * rho * (1 - M_prime)]",
                "arguments": vars(args),
                "model_order": model_order,
                "base_params": base_params,
                "model_configs": {name: model_configs[name] for name in model_order},
                "figure_note": "Representative realization with a shared initial state and seed.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved config : {config_path}")

    print("\n================ Experiment 3 Finished ================")
    print(f"Output dir: {args.output_dir}")
    print("Generated figures:")
    print(f"  fig_exp3_{args.model_set}_spatial_evolution_{format_A(args.A_env)}.[png|pdf|svg]")
    for variable in VARIABLE_ORDER:
        print(f"  fig_exp3_{args.model_set}_{variable}_{format_A(args.A_env)}.[png|pdf|svg]")
    print("=======================================================")


if __name__ == "__main__":
    main()
