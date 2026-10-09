# Code ↔ paper cross-reference

Mapping between the symbols of the manuscript and the names used in the code.
"Default / reported" is the value used in the numerical examples of the paper;
the driver scripts (`run_spline.py`, `run_staircase.py`, `run_sphere.py`) pass
case-specific choices through their configuration blocks and `ACO_OVERRIDES`.

## Problem definition and constraints

| Paper | Code | Default / reported |
| --- | --- | --- |
| $J(\gamma)$ band-integrated active input power | `fem_beam3d.band_integrated_power` via the driver's `geom_fn` | bands per case |
| target bands and weights | `BANDS`, `BAND_WEIGHTS` | per case |
| Rayleigh damping $(a_M, a_K)$ | `RAY` | (0, 2e-7) |
| prescribed displacement phasors at $A$ | `PRESCRIBED_DISP`, `PRESCRIBED_MASK` | (1e-6, 1e-6, 1e-6), all prescribed |
| $L_{\max}=\chi_L L_{AB}$ developed-length budget | `LMAX_FACTOR` (or `LMAX_ABS`) → `Lmax` | 1.7 (3.0 staircase, 4 sphere) |
| $R_{\min}$ minimum bend radius | `R_MIN`, `params["R_min"]` | 8 mm |
| $\theta_{\max}$ turn-angle pre-filter | `MAX_TURN`, `params["max_turn_deg"]` | `geo.max_turn_from_pitch(h, R_MIN)` |
| exit / arrival direction cones at $A$ / $B$ | `EXIT_DIR_A`, `ARRIVE_DIR_B`, `EXIT_CONE_DEG`, `ARRIVE_CONE_DEG` (`params["start_dir"]`, `["end_dir"]`, `["start_max_turn_deg"]`, `["end_max_turn_deg"]`) | per case |
| spline-level end-tangent clamp | `TANGENT_CLAMP` → `geometry.END_TANGENTS` / `fit_spline(..., end_tangents=...)` | on where cones are prescribed |
| $\delta_\Omega$ containment margin | `DOMAIN_TOL` | half the smallest lattice pitch |
| keep-out region | `EXCLUDE_CENTER`, `ENFORCE_KEEPOUT` | per case |
| self-clearance bound | `SELF_CLEARANCE` | `None` (measured and reported, not enforced) |

## Spline fit and finite elements

| Paper | Code | Default / reported |
| --- | --- | --- |
| $s_{\mathrm{fit}}$ absolute smoothing bound | `SMOOTH` | $10^{-4}$ (case value in each driver) |
| $\varepsilon_{\mathrm{fit}}$ per-waypoint alternative ($s=N_{\mathrm{wp}}\varepsilon^2$) | `SMOOTH_RMS` → `geometry.DEFAULT_SMOOTH_RMS` | `None` (off; reported configuration) |
| beam element size (in-loop / reported) | `ELEM` / `ELEM_REPORT` | 5 mm / mesh-convergence pick |
| in-loop harmonic solver | `HARMONIC_SOLVER` | `"modal"` (`"direct"` = reference banded solver) |
| modal cut-off (multiple of the band's $f_{\mathrm{hi}}$) | `MODAL_FCUT` | 3.0 (`None` = complete basis, exact) |

## ACO search

| Paper | Code | Default / reported |
| --- | --- | --- |
| $\xi$ balanced detour budget | `params["slack_balance"]` | 0.5 (0 = classic straight-line bound) |
| bidirectional fraction | `params["bidirectional"]` | 0.5 (0 = all ants from $A$) |
| $d^{\mathcal{G}}$ geodesic distance field | `grid.distances_from` | computed when $\xi>0$; a second field for B-started ants |
| $\alpha$ pheromone exponent | `params["alpha"]` | 1.0 |
| $\rho$ evaporation | `params["rho"]` | 0.15 |
| $\tau_0$, $(\tau_{\min},\tau_{\max})$ | `params["tau0"]`, `params["tau_bounds"]` | 1, (0.05, 8) |
| $Q_\tau$ deposit scale, $w_e$ elitist weight | `params["Q_tau"]`, `params["elitist_weight"]` | 1, 1 |
| $q_0^{\mathrm{s}}, q_0^{\mathrm{e}}$ exploitation ramp | `params["q0_anneal"]` | (0.2, 0.9) |
| $T_q$ exploration window | `params["q0_iters"]` | $\mathrm{rclip}(5h, 40, 120)$, $h$ = mean hops per axis |
| $T$, $n_{\mathrm{patience}}$, $n_{\mathrm{stall}}$ | `params["max_iter"]`, `["patience"]`, `["stagnation_reset"]` | $4T_q$, $T_q$, $\lfloor T_q/2\rfloor$ |
| $\varepsilon_J$ progress tolerance | `params["improve_tol"]` | $10^{-2}$ |
| $\kappa_c$ concentration gate for the restart | `params["kick_concentration"]` | 3 |
| $\delta$ restart intensity | `params["stagnation_smooth"]` | 1 (full reset) |
| $n_{\mathrm{ant}}$ ants per iteration | `params["n_ants"]` | $\mathrm{rclip}(2.8h, 12, 60)$ |
| multi-start seeds $N_s$ | `SEEDS` | [3, 5, 7, 11, 13] |

`aco.auto_params(grid)` derives the budget entries from the lattice; see
[`ACO_PARAMETERS.md`](ACO_PARAMETERS.md) for the role and tuning of each entry.
