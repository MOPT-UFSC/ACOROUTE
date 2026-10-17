# ACO parameters

All optimizer parameters live in one `params` dictionary consumed by
`aco.optimize`. The reported cases build it with `aco.auto_params(grid)`,
which scales the search budget to the lattice, and then apply the driver's
`ACO_OVERRIDES`. `aco.default_params()` holds the generic fallbacks for
direct use of the library.

```python
params = aco.auto_params(grid, overrides=ACO_OVERRIDES)
```

Some keys keep historical names (`stagnation_reset`, `kick_concentration`,
`q0_rewind_on_kick`). They are retained so that existing configurations and
logs stay valid. In the paper the corresponding mechanism is called the
*stagnation restart*, and the names used there are given in each table.

## Budget scaling (auto_params)

With `h = mean(nx, ny, nz) − 1` (the mean number of hops per axis, written
$h_g$ in the paper) and `rclip(v, lo, hi) = round to the nearest integer, then
clamp to [lo, hi]`:

| Entry | Value | Paper | Role |
| --- | --- | --- | --- |
| `q0_iters` | `rclip(5h, 40, 120)` | $T_q$ | exploration window: iterations over which the exploitation probability follows its linear schedule |
| `n_ants` | `rclip(2.8h, 12, 60)` | $n_{\mathrm{ant}}$ | ants per iteration |
| `patience` | `q0_iters` | $n_{\mathrm{patience}}$ | stop after this many iterations without progress. The exploration window always ends before the run can stop |
| `stagnation_reset` | `q0_iters // 2` | $n_{\mathrm{stall}}$ | stale iterations before a stagnation restart is considered. Every restart has half a window to re-converge before the patience can stop the run |
| `max_iter` | `4 * q0_iters` | $T$ | iteration cap, normally not reached |

The dynamics parameters below are size-independent.

## Transition rule

| Entry | Default (auto) | Role |
| --- | --- | --- |
| `alpha` | 1.0 | exponent $\alpha$ of the pheromone field τ in the transition rule |
| `q0_anneal` | (0.2, 0.9) | exploitation probability $(q_0^{\mathrm{s}}, q_0^{\mathrm{e}})$, varied linearly from start to end over `q0_iters` and held at the end value afterwards. Fixing `q0_iters` decouples the exploration schedule from `max_iter` |
| `q0` | 0.7 | fixed exploitation probability, used only when `q0_anneal` is unset |
| `q0_rewind_on_kick` | False | restart the $q_0$ schedule after a stagnation restart so that the re-diversified pheromone is explored at low $q_0$. Not used in the reported runs |

## Construction constraints

| Entry | Default | Role |
| --- | --- | --- |
| `max_turn_deg` | None | turn-angle limit $\theta_{\max}$ per node (curvature pre-filter). The drivers derive it from the lattice pitch and `R_min` via `geometry.max_turn_from_pitch` |
| `start_dir`, `start_max_turn_deg` | None | prescribed exit direction at the start node and its cone half-angle (None = `max_turn_deg`, else 90°) |
| `end_dir`, `end_max_turn_deg` | None | prescribed arrival (travel) direction at the target node and its cone |
| `R_min` | 0.0 | hard minimum bend radius $R_{\min}$ checked on the as-built spline (0 = off) |
| `max_ops` | `50 * len(grid["coords"])` | construction-operation budget per ant (moves plus backtracks). An ant that exhausts it fails without a route |
| `slack_balance` | 0.0 (0.5 reported) | balanced detour budget $\xi \in [0,1]$: the slack `Lmax − d(A,B)` is rationed along the route in proportion to the geodesic distance already covered, so ants keep the same turning capability near $B$ as near $A$. 0 = classic first-come-first-served bound |
| `bidirectional` | 0.0 (0.5 reported) | fraction of each iteration's ants constructed from $B$ toward $A$ ($n_{\mathrm{rev}}=\operatorname{round}(n_{\mathrm{ant}}/2)$ when 0.5). Completed routes are canonicalized to the $A\to B$ orientation before the geometry checks, the objective and the cache, so the excitation/support roles and $J$ are unchanged. Pheromone lives on undirected edges, so both populations reinforce one field. Exit/arrival cones are mapped automatically for the reversed construction |

Warm-start corridors are passed to `aco.optimize` through the `seed_paths`
argument (one or more lattice routes, built in `run_staircase.py` with
`grid.diverse_seed_paths`). The pheromone of their edges is raised to
`min(6 * tau0, tau_bounds[1])` before the first iteration. The reported runs
use one corridor in the staircase example and none elsewhere.

## Pheromone update

| Entry | Default (auto) | Role |
| --- | --- | --- |
| `rho` | 0.15 | evaporation rate $\rho$ |
| `tau0` | 1.0 | initial pheromone level $\tau_0$ |
| `tau_bounds` | (0.05, 8.0) | MAX–MIN Ant System clamping $(\tau_{\min}, \tau_{\max})$ |
| `Q_tau` | 1.0 | deposit scale $Q_\tau$ (constant: $J$ only ranks the routes) |
| `deposit_mode` | `"iteration_best"` | which ants deposit: `"all"`, `"iteration_best"` or `"best_so_far"` |
| `elitist_weight` | 1.0 | weight $w_e$ of the best-so-far reinforcement added each iteration (0 disables) |
| `global_evaporation` | True | evaporate on every admissible edge, not only on edges that already carry a deposit |

## Stagnation restart and stopping

A restart is triggered when four conditions hold at the end of an iteration:
(i) the incumbent has not improved by more than `improve_tol` for
`stagnation_reset` iterations; (ii) at least `stagnation_reset` iterations
have elapsed since the previous restart or since the start; (iii) the
exploration window has ended (`it >= q0_iters`); (iv) the median objective of
the feasible routes of the iteration is below `kick_concentration` times the
incumbent, so the colony has concentrated. The restart resets τ toward
$\tau_{\max}$ on every admissible edge; the incumbent is kept.

| Entry | Default (auto) | Paper | Role |
| --- | --- | --- | --- |
| `improve_tol` | 1e-2 | $\varepsilon_J$ | relative gain on the incumbent that counts as progress for the restart and for the patience stop |
| `stagnation_reset` | `q0_iters // 2` | $n_{\mathrm{stall}}$ | stale iterations before a restart is considered |
| `kick_concentration` | 3.0 | $\beta_c$ | concentration condition: the restart fires only when the iteration's median objective is below $\beta_c \times$ best |
| `stagnation_smooth` | 1.0 | — | restart intensity: 1.0 = full reset of τ to $\tau_{\max}$, < 1 = partial pull toward $\tau_{\max}$ |
| `patience` | `q0_iters` | $n_{\mathrm{patience}}$ | stop after this many iterations without progress |
| `max_iter` | `4 * q0_iters` | $T$ | iteration cap |

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
  sooner) or `stagnation_smooth` (partial restarts). Lowering the start of
  `q0_anneal` or raising `tau_bounds[0]` also fights early convergence.
- **Route shape**: `slack_balance` and `bidirectional` control where along the
  route the ants can spend their detour slack. With `bidirectional` consider a
  larger `n_ants` (e.g. 2× the auto value) to keep the per-direction
  population size.
- Any entry can be pinned per case through the driver's `ACO_OVERRIDES`.
