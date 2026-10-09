# ACO parameters

All optimizer parameters live in one `params` dictionary consumed by
`aco.optimize`. The reported cases build it with `aco.auto_params(grid)`,
which scales the search budget to the lattice, and then apply the driver's
`ACO_OVERRIDES`. `aco.default_params()` holds the generic fallbacks for
direct use of the library.

```python
params = aco.auto_params(grid, overrides=ACO_OVERRIDES)
```

## Budget scaling (auto_params)

With `h = mean(nx, ny, nz) − 1` (the number of hops in a direct route, i.e.
decisions per ant per route) and `rclip(v, lo, hi) = round then clamp`:

| Entry | Value | Role |
| --- | --- | --- |
| `q0_iters` | `rclip(5h, 40, 120)` | exploration window $T_q$: iterations over which `q0_anneal` ramps |
| `n_ants` | `rclip(2.8h, 12, 60)` | ants per iteration |
| `patience` | `q0_iters` | early stop after this many iterations without progress — the ramp always ends before the run can stop |
| `stagnation_reset` | `q0_iters // 2` | stale iterations that trigger a pheromone kick; every restart has half a ramp to re-converge before the patience can trigger |
| `max_iter` | `4 * q0_iters` | safety cap, normally not reached |

The dynamics parameters below are size-independent.

## Transition rule

| Entry | Default (auto) | Role |
| --- | --- | --- |
| `alpha` | 1.0 | exponent of the pheromone field τ in the transition rule |
| `q0_anneal` | (0.2, 0.9) | exploitation probability ramped linearly from start to end over `q0_iters`; fixing `q0_iters` decouples the exploration schedule from `max_iter` |
| `q0` | 0.7 | fixed exploitation probability, used only when `q0_anneal` is unset |
| `q0_rewind_on_kick` | False | restart the q0 ramp after a stagnation kick so the re-diversified pheromone is explored at low q0 |

## Construction constraints

| Entry | Default | Role |
| --- | --- | --- |
| `max_turn_deg` | None | turn-angle limit per node (cheap curvature pre-filter); the drivers derive it from the lattice pitch and `R_min` via `geometry.max_turn_from_pitch` |
| `start_dir`, `start_max_turn_deg` | None | prescribed exit direction at the start node and its cone half-angle (None = `max_turn_deg`, else 90°) |
| `end_dir`, `end_max_turn_deg` | None | prescribed arrival (travel) direction at the target node and its cone |
| `R_min` | 0.0 | hard minimum bend radius checked on the as-built spline (0 = off) |
| `slack_balance` | 0.0 (0.5 reported) | ξ ∈ [0,1]: ration the detour budget `Lmax − d(A,B)` proportionally along the route, using the geodesic distance field, so ants keep the same turning capability near B as near A; 0 = classic first-come-first-served bound (bit-identical to the legacy rule) |
| `bidirectional` | 0.0 (0.5 reported) | fraction of each iteration's ants constructed from B toward A. Completed routes are canonicalized to the A→B orientation before the geometry checks, the objective and the cache, so the excitation/support roles and J are unchanged; pheromone lives on undirected edges, so both populations reinforce one field. Exit/arrival cones are mapped automatically for the reversed construction |

## Pheromone update

| Entry | Default (auto) | Role |
| --- | --- | --- |
| `rho` | 0.15 | evaporation rate |
| `tau0` | 1.0 | initial pheromone level |
| `tau_bounds` | (0.05, 8.0) | MAX–MIN Ant System clamping (τ_min, τ_max) |
| `Q_tau` | 1.0 | deposit scale (constant: J only ranks the routes) |
| `deposit_mode` | `"iteration_best"` | which ants deposit: `"all"`, `"iteration_best"` or `"best_so_far"` |
| `elitist_weight` | 1.0 | best-so-far reinforcement added each iteration (0 disables) |
| `global_evaporation` | True | evaporate on every admissible edge, not only on edges that already carry a deposit |

## Stagnation restart and stopping

| Entry | Default (auto) | Role |
| --- | --- | --- |
| `improve_tol` | 1e-2 | relative gain on the incumbent that counts as progress for the restart and the patience stop (ε_J) |
| `stagnation_reset` | `q0_iters // 2` | stale iterations before a kick is considered |
| `kick_concentration` | 3.0 | concentration gate κ_c: the kick fires only when the iteration's median objective is below κ_c × best, i.e. the colony has actually concentrated |
| `stagnation_smooth` | 1.0 | kick intensity: 1.0 = full reset of τ to τ_max, < 1 = gentle pull toward τ_max |
| `patience` | `q0_iters` | stop after this many iterations without progress |
| `max_iter` | `4 * q0_iters` | iteration cap |

## Execution

| Entry | Default | Role |
| --- | --- | --- |
| `n_eval_workers` | None (`"auto"` in the drivers) | parallel FEM evaluations of one iteration's routes (int, or `"auto"` = one worker per core). Identical results to serial; only wall-clock drops. Automatically disabled inside the multi-start pool to avoid nested pools; silently serial on platforms without fork |
| `store_paths_every` | 100 | snapshot all ant paths every N iterations (for the route-cloud plots) |
| `print_every` | 10 | verbose progress line every N iterations |

Seed-level parallelism is separate: the driver's `N_WORKERS` (fork-based pool)
or `run_campaign.py` (process-spawn based, Windows-safe) run the multi-start
seeds concurrently. Prefer seed-level parallelism when several seeds run, and
`n_eval_workers` for a single seed, to avoid oversubscribing the machine.

## Tuning notes

- **Premature stagnation of the objective**: lower `stagnation_reset` (react
  sooner) or `stagnation_smooth` (gentler kicks); lowering the start of
  `q0_anneal` or raising `tau_bounds[0]` also fights early convergence.
- **Route shape**: `slack_balance` and `bidirectional` control where along the
  route the ants can spend their detour slack; with `bidirectional` consider a
  larger `n_ants` (e.g. 2× the auto value) to keep the per-direction
  population size.
- Any entry can be pinned per case through the driver's `ACO_OVERRIDES`.
