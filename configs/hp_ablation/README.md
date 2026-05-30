# Hyperparameter ablation configs (App. F, Tab. 6)

Defaults from the paper: $\kappa = 0.0125$, $\tau_b = 3.0$, $K_n =$ default
candidate size, $\lambda = 0.22$.

Each JSON below overrides one hyperparameter while inheriting everything else
from `configs/sampling/canonical_25pct.json`.

| Knob | CLI flag | Sweep values |
| --- | --- | --- |
| $\kappa$ (anchor budget ratio) | `--ratio` | 0.005, 0.01, 0.0125, 0.02, 0.025 |
| $\tau_b$ (entropy temperature) | `--budget_allocation_entropy_temperature` | 0.5, 1.0, 3.0, 5.0, 10.0 |
| $K_n$ (refine candidate size) | `--kmedoid_refine_candidate_size` | 4, 8, 16, 32 |
| $\lambda$ (separation reg.) | `--kmedoid_refine_separation_reg` | 0.0, 0.1, 0.22, 0.5 |
