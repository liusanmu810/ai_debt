"""Run the four-group Experiment 2 ablation on a small-world network.

Formal defaults use 50,000 evolution steps and match the square-grid 2 x 2
ablation while using a
Watts--Strogatz network with N=10,000, k=4, and rewiring probability p=0.1.
The concentration gate is rho * (1 - M'); technical debt is absent from it.
"""

from __future__ import annotations

from pathlib import Path

from experiment2_core import make_parser, run_experiment2
from small_world_model import AITechnicalDebtMDAE_SmallWorldGPU


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = make_parser(
        "Experiment 2 topology robustness on a Watts--Strogatz network.",
        SCRIPT_DIR / "results_exp2_small_world_concentration_gate",
    )
    parser.add_argument(
        "--small-world-k",
        type=int,
        default=4,
        help="Even ring-lattice degree before rewiring.",
    )
    parser.add_argument(
        "--rewiring-probability",
        type=float,
        default=0.1,
        help="Watts--Strogatz edge-rewiring probability.",
    )
    args = parser.parse_args()
    if args.small_world_k < 2 or args.small_world_k % 2 != 0:
        parser.error("--small-world-k must be an even integer of at least 2.")
    if args.small_world_k >= args.num_nodes:
        parser.error("--small-world-k must be smaller than --num-nodes.")
    if not 0.0 <= args.rewiring_probability <= 1.0:
        parser.error("--rewiring-probability must be in [0, 1].")

    run_experiment2(
        args=args,
        model_class=AITechnicalDebtMDAE_SmallWorldGPU,
        topology="small_world",
        network_kwargs={
            "small_world_k": int(args.small_world_k),
            "small_world_p": float(args.rewiring_probability),
        },
    )


if __name__ == "__main__":
    main()
