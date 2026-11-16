
"""
Shared utilities for Experiment 4 heatmap data generation.

Put this file in the same directory as MODEL.py and the four separated
experiment scripts.
"""

import os
import time
from typing import Callable, Dict, List, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import numpy as np
import numba
from numba import cuda

from MODEL import AITechnicalDebtMDAE_GPU


def ensure_cuda_environment() -> None:
    """Fail early when the local CUDA/Numba stack cannot run this experiment."""
    version_parts = tuple(int(part) for part in numba.__version__.split(".")[:2])
    if version_parts < (0, 63):
        raise RuntimeError(
            "Experiment 4 requires Numba >= 0.63. "
            f"Current Numba version: {numba.__version__}. "
            "Use a Python environment with the required Numba and CUDA support."
        )
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Use a Python environment with the required Numba and CUDA support."
        )


def get_base_params(init_D_max: float = 0.02) -> Dict[str, float]:
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

        # Keep the manuscript's deterministic candidate-selection rule.
        selection_rule=0,
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


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fallback_progress(completed: int, total: int, desc: str, start_time: float) -> None:
    percent = 100.0 * completed / total if total > 0 else 100.0
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    print(
        f"\r{desc}: {completed}/{total} "
        f"({percent:6.2f}%) | elapsed {format_seconds(elapsed)} "
        f"| ETA {format_seconds(remaining)}",
        end="",
        flush=True,
    )


def sample_x_a_D_means(model: AITechnicalDebtMDAE_GPU) -> Tuple[float, float, float]:
    arrays = model.device_arrays()
    mean_x = float(np.mean(arrays["x"].copy_to_host()))
    mean_a = float(np.mean(arrays["a"].copy_to_host()))
    mean_D = float(np.mean(arrays["D"].copy_to_host()))
    return mean_x, mean_a, mean_D


def run_single_simulation(
    A_env: float,
    run_id: int,
    grid_size: int,
    steps: int,
    steady_window: int,
    sample_interval: int,
    seed_base: int,
    model_params: Dict[str, float],
) -> Dict[str, float]:
    seed = seed_base + run_id * 10000
    sample_interval = max(1, int(sample_interval))

    model = AITechnicalDebtMDAE_GPU(
        grid_size=grid_size,
        seed=seed,
        A_env=A_env,
        **model_params,
    )

    mean_x_list: List[float] = []
    mean_a_list: List[float] = []
    mean_D_list: List[float] = []

    steady_start = steps - steady_window
    start_time = time.time()

    for t in range(steps):
        model.step(synchronize=False)
        current_t = t + 1

        if current_t > steady_start:
            offset = current_t - steady_start
            should_sample = (offset % sample_interval == 0) or (current_t == steps)

            if should_sample:
                mean_x, mean_a, mean_D = sample_x_a_D_means(model)
                mean_x_list.append(mean_x)
                mean_a_list.append(mean_a)
                mean_D_list.append(mean_D)

    cuda.synchronize()
    elapsed = time.time() - start_time

    if len(mean_x_list) == 0:
        raise RuntimeError("No late-time samples were collected.")

    return {
        "mean_x": float(np.mean(mean_x_list)),
        "mean_a": float(np.mean(mean_a_list)),
        "mean_D": float(np.mean(mean_D_list)),
        "num_samples": int(len(mean_x_list)),
        "elapsed_seconds": float(elapsed),
        "seed": int(seed),
    }


def run_heatmap_grid(
    heatmap_name: str,
    axis_x_name: str,
    axis_x_values: np.ndarray,
    axis_y_name: str,
    axis_y_values: np.ndarray,
    runs: int,
    grid_size: int,
    steps: int,
    steady_window: int,
    sample_interval: int,
    seed_base: int,
    output_path: str,
    build_point_fn: Callable[[float, float], Tuple[float, Dict[str, float]]],
    extra_metadata: Optional[Dict[str, object]] = None,
    checkpoint_every: int = 25,
) -> None:
    ensure_cuda_environment()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"Completed heatmap already exists; skipping: {output_path}")
        return

    n_x = len(axis_x_values)
    n_y = len(axis_y_values)
    total_jobs = runs * n_y * n_x

    shape = (runs, n_y, n_x)
    raw_x = np.full(shape, np.nan, dtype=np.float64)
    raw_a = np.full(shape, np.nan, dtype=np.float64)
    raw_D = np.full(shape, np.nan, dtype=np.float64)
    elapsed = np.full(shape, np.nan, dtype=np.float64)
    num_samples = np.zeros((runs, n_y, n_x), dtype=np.int64)
    completed_mask = np.zeros(shape, dtype=np.bool_)
    checkpoint_path = output_path + ".checkpoint.npz"

    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        saved_config = (
            int(checkpoint["runs"]),
            int(checkpoint["grid_size"]),
            int(checkpoint["steps"]),
            int(checkpoint["steady_window"]),
            int(checkpoint["sample_interval"]),
            int(checkpoint["seed_base"]),
        )
        requested_config = (
            int(runs),
            int(grid_size),
            int(steps),
            int(steady_window),
            int(sample_interval),
            int(seed_base),
        )
        if saved_config != requested_config:
            raise ValueError(
                f"Checkpoint configuration {saved_config} does not match "
                f"requested configuration {requested_config}."
            )
        saved_shape = tuple(checkpoint["raw_x"].shape)
        if saved_shape != shape:
            raise ValueError(
                f"Checkpoint shape {saved_shape} does not match requested shape {shape}."
            )
        if not np.array_equal(checkpoint["axis_x"], axis_x_values):
            raise ValueError("Checkpoint x-axis does not match the requested grid.")
        if not np.array_equal(checkpoint["axis_y"], axis_y_values):
            raise ValueError("Checkpoint y-axis does not match the requested grid.")
        raw_x = checkpoint["raw_x"].copy()
        raw_a = checkpoint["raw_a"].copy()
        raw_D = checkpoint["raw_D"].copy()
        elapsed = checkpoint["elapsed_seconds"].copy()
        num_samples = checkpoint["num_samples"].copy()
        completed_mask = checkpoint["completed_mask"].astype(np.bool_)

    def save_checkpoint() -> None:
        temporary_path = checkpoint_path + ".tmp.npz"
        np.savez_compressed(
            temporary_path,
            axis_x=axis_x_values.astype(np.float64),
            axis_y=axis_y_values.astype(np.float64),
            runs=np.array(runs, dtype=np.int64),
            grid_size=np.array(grid_size, dtype=np.int64),
            steps=np.array(steps, dtype=np.int64),
            steady_window=np.array(steady_window, dtype=np.int64),
            sample_interval=np.array(sample_interval, dtype=np.int64),
            seed_base=np.array(seed_base, dtype=np.int64),
            raw_x=raw_x,
            raw_a=raw_a,
            raw_D=raw_D,
            elapsed_seconds=elapsed,
            num_samples=num_samples,
            completed_mask=completed_mask,
        )
        os.replace(temporary_path, checkpoint_path)

    print(f"\n================ Running heatmap: {heatmap_name} ================")
    print(f"axis_x ({axis_x_name}): {axis_x_values[0]:.4f} -> {axis_x_values[-1]:.4f}, n={n_x}")
    print(f"axis_y ({axis_y_name}): {axis_y_values[0]:.4f} -> {axis_y_values[-1]:.4f}, n={n_y}")
    print(
        f"runs={runs}, total simulations={total_jobs}, "
        f"grid={grid_size}x{grid_size}, steps={steps}, "
        f"steady_window={steady_window}, sample_interval={sample_interval}"
    )
    print(f"output={output_path}")
    print("=================================================================")

    total_start = time.time()
    completed = int(np.count_nonzero(completed_mask))

    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=total_jobs,
            initial=completed,
            desc=heatmap_name,
            unit="sim",
            dynamic_ncols=True,
            mininterval=1.0,
        )
    else:
        fallback_progress(completed, total_jobs, heatmap_name, total_start)

    try:
        for run_id in range(runs):
            for iy, y_value in enumerate(axis_y_values):
                for ix, x_value in enumerate(axis_x_values):
                    if completed_mask[run_id, iy, ix]:
                        continue
                    A_env, params = build_point_fn(float(x_value), float(y_value))

                    result = run_single_simulation(
                        A_env=A_env,
                        run_id=run_id,
                        grid_size=grid_size,
                        steps=steps,
                        steady_window=steady_window,
                        sample_interval=sample_interval,
                        seed_base=seed_base,
                        model_params=params,
                    )

                    raw_x[run_id, iy, ix] = result["mean_x"]
                    raw_a[run_id, iy, ix] = result["mean_a"]
                    raw_D[run_id, iy, ix] = result["mean_D"]
                    elapsed[run_id, iy, ix] = result["elapsed_seconds"]
                    num_samples[run_id, iy, ix] = result["num_samples"]
                    completed_mask[run_id, iy, ix] = True

                    completed += 1

                    if checkpoint_every > 0 and completed % checkpoint_every == 0:
                        save_checkpoint()

                    if progress is not None:
                        progress.set_postfix(
                            run=f"{run_id + 1}/{runs}",
                            **{
                                axis_x_name: f"{float(x_value):.3f}",
                                axis_y_name: f"{float(y_value):.3f}",
                                "x": f"{result['mean_x']:.3f}",
                                "a": f"{result['mean_a']:.3f}",
                                "D": f"{result['mean_D']:.3f}",
                            },
                        )
                        progress.update(1)
                    else:
                        refresh_every = max(1, total_jobs // 100)
                        if completed == total_jobs or completed % refresh_every == 0:
                            fallback_progress(completed, total_jobs, heatmap_name, total_start)
    finally:
        if completed < total_jobs and completed > 0:
            save_checkpoint()
        if progress is not None:
            progress.close()
        else:
            print()

    total_elapsed = time.time() - total_start

    mean_x = np.mean(raw_x, axis=0)
    mean_a = np.mean(raw_a, axis=0)
    mean_D = np.mean(raw_D, axis=0)

    std_x = np.std(raw_x, axis=0)
    std_a = np.std(raw_a, axis=0)
    std_D = np.std(raw_D, axis=0)

    save_kwargs = dict(
        heatmap_name=np.array(heatmap_name),
        axis_x_name=np.array(axis_x_name),
        axis_y_name=np.array(axis_y_name),
        axis_x=axis_x_values.astype(np.float64),
        axis_y=axis_y_values.astype(np.float64),
        runs=np.array(runs, dtype=np.int64),
        grid_size=np.array(grid_size, dtype=np.int64),
        steps=np.array(steps, dtype=np.int64),
        steady_window=np.array(steady_window, dtype=np.int64),
        sample_interval=np.array(sample_interval, dtype=np.int64),
        seed_base=np.array(seed_base, dtype=np.int64),
        raw_x=raw_x,
        raw_a=raw_a,
        raw_D=raw_D,
        mean_x=mean_x,
        mean_a=mean_a,
        mean_D=mean_D,
        std_x=std_x,
        std_a=std_a,
        std_D=std_D,
        elapsed_seconds=elapsed,
        num_samples=num_samples,
        total_elapsed_seconds=np.array(total_elapsed, dtype=np.float64),
    )

    if extra_metadata is not None:
        for key, value in extra_metadata.items():
            save_kwargs[key] = np.array(value)

    np.savez_compressed(output_path, **save_kwargs)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"Saved heatmap data: {output_path}")
    print(f"Elapsed for {heatmap_name}: {format_seconds(total_elapsed)}")
    print("===============================================================")
