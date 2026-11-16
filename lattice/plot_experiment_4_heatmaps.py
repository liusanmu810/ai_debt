"""
Experiment 4: Plot Heatmaps from NPZ Data (All panels as individual figures)
============================================================================

用途
----
读取实验 4 生成的三组 50 x 50 .npz 热图数据，并绘制单独热图：

    1. (A, w_M)          -> <x>, <a>, <D>
    2. (A, alpha_conc)   -> <x>, <a>, <D>
    3. (w_M, alpha_conc) -> <x>, <a>, <D>, fixed A = 1.0

每组参数扫描输出一张包含三个指标的组合图，同时保留 9 张单指标图。

关键设置
--------
1. <x> 和 <a> 的色带固定为 [0, 1]；
2. 三个归一化状态量的色带均固定为 [0, 1]，便于不同参数面比较；
3. <x>、<a>、<D> 均使用高对比度 turbo 彩虹色带；
4. 保留 zoom + gaussian_filter + bicubic 的平滑绘图方式；
5. 平滑只影响展示，不修改原始 .npz 数据。

数据来源
--------
默认读取目录：

    results_exp4_heatmaps_current_formula_50x50/

需要存在以下文件：

    experiment_4_A_wM_heatmap_data.npz
    experiment_4_A_alpha_heatmap_data.npz
    experiment_4_wM_alpha_heatmap_data.npz

输出目录
--------
默认输出到：

    results_exp4_heatmaps_current_formula_50x50_figures/

输出文件示例
-----------
    fig_exp4_A_wM_x.png/pdf
    fig_exp4_A_wM_a.png/pdf
    fig_exp4_A_wM_D.png/pdf

    fig_exp4_A_alpha_x.png/pdf
    fig_exp4_A_alpha_a.png/pdf
    fig_exp4_A_alpha_D.png/pdf

    fig_exp4_wM_alpha_x.png/pdf
    fig_exp4_wM_alpha_a.png/pdf
    fig_exp4_wM_alpha_D.png/pdf

    fig_exp4_A_wM_axd.png/pdf
    fig_exp4_A_alpha_axd.png/pdf
    fig_exp4_alpha_wM_axd.png/pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter, zoom


# ==============================================================================
# 0. 全局绘图设置
# ==============================================================================

plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 0.8


# ==============================================================================
# 1. 配置区
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "results_exp4_heatmaps_current_formula_50x50")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_exp4_heatmaps_current_formula_50x50_figures")

SAVE_PDF = True
DPI = 600

# 平滑参数
SMOOTH = True
ZOOM_FACTOR = 4
SMOOTHING_SIGMA = 4.0

# 色带设置
FIX_X_A_COLOR_RANGE_01 = True
USE_D_DATA_MAX = False
MIN_D_VMAX = 1e-6


HEATMAP_FILES = {
    "A_wM": "experiment_4_A_wM_heatmap_data.npz",
    "A_alpha": "experiment_4_A_alpha_heatmap_data.npz",
    "wM_alpha": "experiment_4_wM_alpha_heatmap_data.npz",
}


HEATMAP_TITLES = {
    "A_wM": r"$(A, w_M)$",
    "A_alpha": r"$(A, \alpha)$",
    "wM_alpha": r"$(w_M, \alpha)$ at $A=1.0$",
}


AXIS_LABELS = {
    "A_env": r"$A$",
    "r_public": r"$r$",
    "w_memory": r"$w_M$",
    "alpha_conc": r"$\alpha$",
}


# 使用饱和度较高、亮度变化较平滑的现代彩虹色带。
# 相比传统 jet，turbo 能减少不自然的颜色断层，同时保留强对比度。
RAINBOW_CMAP = plt.get_cmap("turbo")


METRIC_SPECS = [
    ("mean_x", r"$\langle x \rangle$", RAINBOW_CMAP, "x"),
    ("mean_a", r"$\langle a \rangle$", RAINBOW_CMAP, "a"),
    ("mean_D", r"$\langle D \rangle$", RAINBOW_CMAP, "D"),
]


# ==============================================================================
# 2. 数据读取与图像平滑
# ==============================================================================

def _npz_scalar_to_str(value) -> str:
    """兼容 np.array('xxx') 形式的字符串读取。"""
    arr = np.asarray(value)
    return str(arr.item()) if arr.shape == () else str(arr)


def load_heatmap_npz(filepath: str) -> dict:
    """读取单个实验 4 npz 文件。"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到数据文件: {filepath}")

    data = np.load(filepath, allow_pickle=False)

    required_keys = [
        "axis_x",
        "axis_y",
        "axis_x_name",
        "axis_y_name",
        "mean_x",
        "mean_a",
        "mean_D",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"{filepath} 缺少字段: {key}")

    loaded = {
        "axis_x": data["axis_x"].astype(np.float64),
        "axis_y": data["axis_y"].astype(np.float64),
        "axis_x_name": _npz_scalar_to_str(data["axis_x_name"]),
        "axis_y_name": _npz_scalar_to_str(data["axis_y_name"]),
        "mean_x": data["mean_x"].astype(np.float64),
        "mean_a": data["mean_a"].astype(np.float64),
        "mean_D": data["mean_D"].astype(np.float64),
    }

    if "fixed_A_env" in data:
        loaded["fixed_A_env"] = float(np.asarray(data["fixed_A_env"]).item())
    if "runs" in data:
        loaded["runs"] = int(np.asarray(data["runs"]).item())
    if "steps" in data:
        loaded["steps"] = int(np.asarray(data["steps"]).item())
    if "heatmap_name" in data:
        loaded["heatmap_name"] = _npz_scalar_to_str(data["heatmap_name"])

    return loaded


def smooth_matrix(
    matrix: np.ndarray,
    zoom_factor: int = 4,
    smoothing_sigma: float = 2.0,
    clip_max: float | None = None,
) -> np.ndarray:
    """
    对二维热图矩阵进行上采样和平滑。
    注意：只影响展示，不会改变原始数据。
    """
    arr = np.asarray(matrix, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    if clip_max is not None:
        arr = np.clip(arr, 0.0, clip_max)

    if not SMOOTH:
        return arr

    adapted_sigma = smoothing_sigma * (zoom_factor / 4.0)
    upsampled = zoom(arr, zoom=zoom_factor, order=3)
    smoothed = gaussian_filter(upsampled, sigma=adapted_sigma)

    if clip_max is not None:
        smoothed = np.clip(smoothed, 0.0, clip_max)

    return smoothed


def get_extent(axis_x: np.ndarray, axis_y: np.ndarray):
    """imshow 使用的物理坐标范围。"""
    return [float(axis_x.min()), float(axis_x.max()), float(axis_y.min()), float(axis_y.max())]


# ==============================================================================
# 3. 绘图辅助函数
# ==============================================================================

def _choose_locator_step(value_range: float) -> float:
    """根据坐标轴范围自动选择主刻度间距。"""
    if value_range <= 1.05:
        return 0.2
    if value_range <= 2.5:
        return 0.5
    if value_range <= 5.5:
        return 1.0
    return 2.0


def setup_axis_ticks(ax, axis_x_name: str, axis_y_name: str, axis_x: np.ndarray, axis_y: np.ndarray) -> None:
    """根据不同参数轴设置刻度。"""
    x_range = float(axis_x.max() - axis_x.min())
    y_range = float(axis_y.max() - axis_y.min())

    x_major = _choose_locator_step(x_range)
    y_major = _choose_locator_step(y_range)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_major / 2.0))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_major / 2.0))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax.tick_params(axis="both", which="major", labelsize=11, length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", labelsize=9, length=2.5, width=0.6)


def get_color_limits(metric_key: str, matrix: np.ndarray):
    """
    根据指标类型返回色带范围。
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)

    if metric_key in {"mean_x", "mean_a"}:
        if FIX_X_A_COLOR_RANGE_01:
            return 0.0, 1.0, [0.0, 0.5, 1.0]

        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        if np.isclose(vmax, vmin):
            vmax = vmin + 1e-6
        return vmin, vmax, [vmin, (vmin + vmax) / 2.0, vmax]

    if metric_key == "mean_D":
        vmin = 0.0
        if USE_D_DATA_MAX:
            vmax = float(np.nanmax(matrix))
            vmax = max(vmax, MIN_D_VMAX)
        else:
            vmax = 1.0
        return vmin, vmax, [0.0, vmax / 2.0, vmax]

    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    if np.isclose(vmax, vmin):
        vmax = vmin + 1e-6
    return vmin, vmax, [vmin, (vmin + vmax) / 2.0, vmax]


def format_colorbar(cbar, ticks) -> None:
    """统一设置 colorbar 样式。"""
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=10)


# ==============================================================================
# 4. 单图绘制函数
# ==============================================================================

def plot_single_heatmap(
    heatmap_key: str,
    data: dict,
    metric_key: str,
    metric_label: str,
    cmap: str,
    short_metric_name: str,
    output_dir: str,
) -> None:
    """
    为单个指标绘制单独热图。
    """
    axis_x = data["axis_x"]
    axis_y = data["axis_y"]
    axis_x_name = data["axis_x_name"]
    axis_y_name = data["axis_y_name"]
    matrix = data[metric_key]

    extent = get_extent(axis_x, axis_y)
    vmin, vmax, cbar_ticks = get_color_limits(metric_key, matrix)

    # 针对不同指标决定展示时是否裁剪
    clip_max = vmax if metric_key == "mean_D" else 1.0
    image_matrix = smooth_matrix(
        matrix,
        zoom_factor=ZOOM_FACTOR,
        smoothing_sigma=SMOOTHING_SIGMA,
        clip_max=clip_max,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4), constrained_layout=True)

    im = ax.imshow(
        image_matrix,
        origin="lower",
        cmap=cmap,
        extent=extent,
        aspect="auto",
        interpolation="bicubic",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_box_aspect(1)

    group_title = HEATMAP_TITLES.get(heatmap_key, heatmap_key)
    ax.set_title(f"{group_title}: {metric_label}", fontsize=16, pad=8)
    ax.set_xlabel(AXIS_LABELS.get(axis_x_name, axis_x_name), fontsize=14)
    ax.set_ylabel(AXIS_LABELS.get(axis_y_name, axis_y_name), fontsize=14)

    setup_axis_ticks(ax, axis_x_name, axis_y_name, axis_x, axis_y)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    format_colorbar(cbar, cbar_ticks)

    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, f"fig_exp4_{heatmap_key}_{short_metric_name}.png")
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
    print(f"保存单图: {png_path}")

    if SAVE_PDF:
        pdf_path = os.path.join(output_dir, f"fig_exp4_{heatmap_key}_{short_metric_name}.pdf")
        plt.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
        print(f"保存 PDF: {pdf_path}")

    plt.close(fig)


def plot_combined_heatmaps(heatmap_key: str, data: dict, output_dir: str) -> None:
    """Draw the three state variables as one manuscript-ready heatmap plate."""
    axis_x = data["axis_x"]
    axis_y = data["axis_y"]
    axis_x_name = data["axis_x_name"]
    axis_y_name = data["axis_y_name"]
    extent = get_extent(axis_x, axis_y)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), constrained_layout=True)

    for panel_index, (ax, spec) in enumerate(zip(axes, METRIC_SPECS)):
        metric_key, metric_label, cmap, _ = spec
        matrix = data[metric_key]
        vmin, vmax, cbar_ticks = get_color_limits(metric_key, matrix)
        clip_max = vmax if metric_key == "mean_D" else 1.0
        image_matrix = smooth_matrix(
            matrix,
            zoom_factor=ZOOM_FACTOR,
            smoothing_sigma=SMOOTHING_SIGMA,
            clip_max=clip_max,
        )
        im = ax.imshow(
            image_matrix,
            origin="lower",
            cmap=cmap,
            extent=extent,
            aspect="auto",
            interpolation="bicubic",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_box_aspect(1)
        ax.set_title(metric_label, fontsize=22, pad=7)
        ax.set_xlabel(AXIS_LABELS.get(axis_x_name, axis_x_name), fontsize=20)
        ax.set_ylabel(AXIS_LABELS.get(axis_y_name, axis_y_name), fontsize=20)
        setup_axis_ticks(ax, axis_x_name, axis_y_name, axis_x, axis_y)
        ax.text(
            -0.14,
            1.05,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            va="bottom",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        format_colorbar(cbar, cbar_ticks)
        ax.tick_params(axis="both", which="major", labelsize=16)
        cbar.ax.tick_params(labelsize=16, pad=3)

    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.08, wspace=0.10)
    os.makedirs(output_dir, exist_ok=True)
    output_key = "alpha_wM" if heatmap_key == "wM_alpha" else heatmap_key
    stem = os.path.join(output_dir, f"fig_exp4_{output_key}_axd")
    fig.savefig(stem + ".png", dpi=DPI, bbox_inches="tight")
    print(f"保存组合图: {stem}.png")
    if SAVE_PDF:
        fig.savefig(stem + ".pdf", dpi=DPI, bbox_inches="tight")
        print(f"保存组合 PDF: {stem}.pdf")
    plt.close(fig)


# ==============================================================================
# 5. 主程序
# ==============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n========== Experiment 4 heatmap plotting ==========")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"平滑设置: SMOOTH={SMOOTH}, ZOOM_FACTOR={ZOOM_FACTOR}, SIGMA={SMOOTHING_SIGMA}")
    print("色带设置: <x>, <a>, <D> 均使用 turbo 彩虹色带和 [0,1] 范围")
    print("输出形式: 每组输出一张三面板组合图，并保留单指标图")
    print("=================================================\n")

    for heatmap_key, filename in HEATMAP_FILES.items():
        filepath = os.path.join(INPUT_DIR, filename)
        print(f"\n-> 正在处理 {heatmap_key}: {filepath}")

        try:
            data = load_heatmap_npz(filepath)
            print(
                f"   数据读取成功: "
                f"x-axis={data['axis_x_name']}({len(data['axis_x'])}), "
                f"y-axis={data['axis_y_name']}({len(data['axis_y'])})"
            )
            print(
                f"   raw ranges: "
                f"<x>=[{np.nanmin(data['mean_x']):.4f}, {np.nanmax(data['mean_x']):.4f}], "
                f"<a>=[{np.nanmin(data['mean_a']):.4f}, {np.nanmax(data['mean_a']):.4f}], "
                f"<D>=[{np.nanmin(data['mean_D']):.4f}, {np.nanmax(data['mean_D']):.4f}]"
            )

            for metric_key, metric_label, cmap, short_metric_name in METRIC_SPECS:
                if metric_key == "mean_D":
                    print(
                        f"   {heatmap_key}: <D> color scale = "
                        f"[0.0000, {max(float(np.nanmax(data['mean_D'])), MIN_D_VMAX):.4f}]"
                    )

                plot_single_heatmap(
                    heatmap_key=heatmap_key,
                    data=data,
                    metric_key=metric_key,
                    metric_label=metric_label,
                    cmap=cmap,
                    short_metric_name=short_metric_name,
                    output_dir=OUTPUT_DIR,
                )

            plot_combined_heatmaps(
                heatmap_key=heatmap_key,
                data=data,
                output_dir=OUTPUT_DIR,
            )

        except Exception as e:
            print(f"处理 {heatmap_key} 时出错: {e}")

    print("\n========== 所有热图绘制完成 ==========")


if __name__ == "__main__":
    main()
