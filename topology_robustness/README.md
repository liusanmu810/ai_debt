# Alternative network topologies

`scale_free_model.py` and `small_world_model.py` implement the network-specific CUDA models. `experiment2_core.py` supplies the shared experiment, statistical-summary, and plotting workflow.

Use `experiment2_scale_free.py` or `experiment2_small_world.py` as entry points. Both support `--help` and `--mode plot`. The repository README gives the manuscript-scale commands; `configs/` records the saved network settings.
