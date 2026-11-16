"""Run the four-group Experiment 2 ablation on a scale-free network.

Formal defaults use 50,000 evolution steps and match the square-grid 2 x 2
ablation while using a
Barabasi--Albert network with N=10,000 and m=2 (mean degree about four).
The concentration gate is rho * (1 - M'); technical debt is absent from it.
"""

from __future__ import annotations

from pathlib import Path

from experiment2_core import make_parser, run_experiment2
from scale_free_model import AITechnicalDebtMDAE_ScaleFreeGPU


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = make_parser(
        "Experiment 2 topology robustness on a scale-free network.",
        SCRIPT_DIR / "results_exp2_scale_free_concentration_gate",
    )
    parser.add_argument(
        "--scale-free-m",
        type=int,
        default=2,
        help="Number of preferential-attachment edges per new node.",
    )
    args = parser.parse_args()
    if args.scale_free_m <= 0:
        parser.error("--scale-free-m must be positive.")
    if args.scale_free_m >= args.num_nodes:
        parser.error("--scale-free-m must be smaller than --num-nodes.")

    run_experiment2(
        args=args,
        model_class=AITechnicalDebtMDAE_ScaleFreeGPU,
        topology="scale_free",
        network_kwargs={"scale_free_m": int(args.scale_free_m)},
    )


if __name__ == "__main__":
    main()
