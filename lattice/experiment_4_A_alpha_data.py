
"""
Experiment 4c: (A, alpha_conc) Heatmap Data Generation

This script generates one Experiment 4 heatmap data file only.
It is PyCharm-friendly: edit the CONFIG section and click Run.
"""

import os
import numpy as np

from exp4_common import get_base_params, run_heatmap_grid


# ==============================================================================
# CONFIG
# ==============================================================================

GRID_SIZE = 50
STEPS = 10000
STEADY_WINDOW = 500

# 1 = sample every step; 10 = faster and recommended for large heatmaps.
SAMPLE_INTERVAL = 10

RUNS = 10
SEED_BASE = 20260514
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_exp4_heatmaps_current_formula_50x50")

INIT_D_MAX = 0.02
HEATMAP_RESOLUTION = 50

A_VALUES = np.linspace(0.0, 1.0, HEATMAP_RESOLUTION, dtype=np.float64)
ALPHA_VALUES = np.linspace(0.0, 0.5, HEATMAP_RESOLUTION, dtype=np.float64)

# ==============================================================================


def build_point(x_value: float, y_value: float):
    base_params = get_base_params(init_D_max=INIT_D_MAX)
    A_env = float(x_value)
    alpha_conc = float(y_value)

    params = dict(base_params)
    params["alpha_conc"] = alpha_conc

    return A_env, params


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "experiment_4_A_alpha_heatmap_data.npz")
    metadata = {f"base_{key}": value for key, value in get_base_params(INIT_D_MAX).items()}
    metadata.update(
        {
            "init_D_max": INIT_D_MAX,
            "concentration_formula": "rho_j * (1 - M_prime_j)",
            "debt_in_concentration_gate": False,
            "candidate_selection": "deterministic_argmax",
        }
    )

    run_heatmap_grid(
        heatmap_name="A_alpha",
        axis_x_name="A_env",
        axis_x_values=A_VALUES,
        axis_y_name="alpha_conc",
        axis_y_values=ALPHA_VALUES,
        runs=RUNS,
        grid_size=GRID_SIZE,
        steps=STEPS,
        steady_window=STEADY_WINDOW,
        sample_interval=SAMPLE_INTERVAL,
        seed_base=SEED_BASE,
        output_path=output_path,
        build_point_fn=build_point,
        extra_metadata=metadata,
    )


if __name__ == "__main__":
    main()
