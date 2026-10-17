# Code ↔ paper cross-reference

Mapping between the symbols of the manuscript *Ant colony optimization of
small-bore pipe routes for reduced active input power* and the names used in
the code. "Reported" is the value used in the numerical examples of the paper.
The driver scripts (`run_spline.py`, `run_staircase.py`, `run_sphere.py`) pass
case-specific choices through their configuration blocks and `ACO_OVERRIDES`;
the case-by-case settings are listed in the README.

Some dictionary keys keep historical names (for example `kick_concentration`
and `stagnation_reset`). They are retained so that existing configurations and
logs stay valid; the paper vocabulary for each one is given below.

## Problem definition and constraints

| Paper | Code | Reported |
| --- | --- | --- |
| $J(\gamma)$ band-integrated active input power, in W·Hz | `fem_beam3d.band_integrated_power` via the driver's `objective_spline` | per case |
| target bands and weights $w_k$ | `BANDS`, `BAND_WEIGHTS` | per case; (5, 1) in the two-band case |
| stiffness-proportional damping $a_K$ (Rayleigh pair $(a_M, a_K)$) | `RAY` | (0, 2e-7) |
| prescribed displacement phasors at $A$ | `PRESCRIBED_DISP`, `PRESCRIBED_MASK` | (1e-6, 1e-6, 1e-6) m, all three translations prescribed, rotations blocked |
| pipe section: outer radius, wall thickness | `OUTER_RADIUS`, `THICKNESS` | 1.575 mm, 0.5 mm |
| material $E$, $\nu$, $\rho$ | `E`, `nu`, `rho` | 210 GPa, 0.3, 7850 kg/m³ |
| $L_{\max}=\mu_L L_{AB}$ developed-length budget | `LMAX_FACTOR` (or `LMAX_ABS`) → `Lmax` | $\mu_L$ = 1.7 (box cases), 3 (staircase), 4 (ball) |
| $R_{\min}$ minimum bend radius | `R_MIN`, `params["R_min"]` | 8 mm |
| $\theta_{\max}$ turn-angle pre-filter (Eq. for the S-jog) | `MAX_TURN`, `params["max_turn_deg"]` | `geo.max_turn_from_pitch(h_lat, R_MIN)` |
| $h_{\mathrm{lat}}$ lattice pitch | `BOX_SIZE`, `NODES` → `grid["coords"]` | per case |
| exit / arrival direction cones at $A$ / $B$ | `EXIT_DIR_A`, `ARRIVE_DIR_B`, `EXIT_CONE_DEG`, `ARRIVE_CONE_DEG` (`params["start_dir"]`, `["end_dir"]`, `["start_max_turn_deg"]`, `["end_max_turn_deg"]`) | vertical exit at $A$ and vertical arrival at $B$ in every case, half-angle 45° |
| spline-level end-tangent pinning | `TANGENT_CLAMP` → `geometry.END_TANGENTS` / `fit_spline(..., end_tangents=...)` | on where cones are prescribed |
| $\delta_\Omega$ containment margin | `DOMAIN_TOL` | half the smallest lattice pitch |
| keep-out region | `EXCLUDE_CENTER`, `ENFORCE_KEEPOUT` (box and ball); baffle boxes and the segment filter in `run_staircase.py` | per case |
| self-clearance bound | `SELF_CLEARANCE` | `OD` (enforced) in the staircase; `None` elsewhere (measured and reported, not enforced) |
| construction-operation budget per ant (50 × retained nodes) | `params["max_ops"]` | default from `len(grid["coords"])` |

## Spline fit and finite elements

| Paper | Code | Reported |
| --- | --- | --- |
| $s_{\mathrm{fit}}$ absolute smoothing bound | `SMOOTH` | $10^{-4}$ m² |
| $\varepsilon_{\mathrm{fit}}$ per-waypoint alternative ($s=N_{\mathrm{wp}}\varepsilon_{\mathrm{fit}}^2$) | `SMOOTH_RMS` → `geometry.DEFAULT_SMOOTH_RMS` | `None` (off) |
| element length $L_e$ inside the loop | `ELEM` | 5 mm |
| element length of the reported spectra and reactions | `ELEM_REPORT` | `None`: chosen by the mesh-convergence check printed by the driver |
| in-loop harmonic solver | `HARMONIC_SOLVER` | `"modal"` (`"direct"` = banded reference solver) |
| modal cut-off $f_{\mathrm{cut}}=3f_2$ with residual flexibility | `MODAL_FCUT` | 3.0 (`None` = complete basis) |
| frequency grid: $n_b$ base points, $\eta_B$ window half-width, $m_B$ points per bandwidth | `fem_beam3d._adaptive_freq_grid(n_base, span, pts_per_bw)` | 64, 4, 4 |

## ACO search

| Paper | Code | Reported |
| --- | --- | --- |
| $\xi$ balanced detour budget | `params["slack_balance"]` | 0.5 (0 = classic straight-line bound) |
| bidirectional construction, $n_{\mathrm{rev}}=\operatorname{round}(n_{\mathrm{ant}}/2)$ ants started at $B$ | `params["bidirectional"]` | 0.5 (0 = all ants from $A$) |
| $d^{\mathcal{G}}$ geodesic distance field | `grid.distances_from` | computed when $\xi>0$; a second field for the ants started at $B$ |
| $\alpha$ pheromone exponent | `params["alpha"]` | 1 |
| $\rho$ evaporation | `params["rho"]` | 0.15 |
| $\tau_0$, $(\tau_{\min},\tau_{\max})$ | `params["tau0"]`, `params["tau_bounds"]` | 1, (0.05, 8) |
| $Q_\tau$ deposit scale, $w_e$ best-so-far weight | `params["Q_tau"]`, `params["elitist_weight"]` | 1, 1 |
| warm-start corridors, pheromone raised to $\min(6\tau_0,\tau_{\max})$ | `seed_paths=` argument of `aco.optimize`, built by `grid.diverse_seed_paths` (`N_SEED_ROUTES`) | staircase only, one corridor |
| $q_0^{\mathrm{s}}, q_0^{\mathrm{e}}$ linear schedule of the exploitation probability | `params["q0_anneal"]` | (0.2, 0.9) |
| $T_q$ exploration window | `params["q0_iters"]` | $\mathrm{rclip}(5h_g, 40, 120)$ |
| $n_{\mathrm{ant}}$ ants per iteration | `params["n_ants"]` | $\mathrm{rclip}(2.8h_g, 12, 60)$ |
| $n_{\mathrm{patience}}$, $n_{\mathrm{stall}}$, $T$ | `params["patience"]`, `["stagnation_reset"]`, `["max_iter"]` | $T_q$, $\lfloor T_q/2\rfloor$, $4T_q$ |
| $\varepsilon_J$ progress tolerance | `params["improve_tol"]` | $10^{-2}$ |
| $\beta_c$ concentration condition of the stagnation restart | `params["kick_concentration"]` | 3 |
| restart intensity (reset of $\tau$ to $\tau_{\max}$) | `params["stagnation_smooth"]` | 1 (full reset) |
| $N_s$ multi-start seeds | `SEEDS` | [3, 5, 7, 11, 13] |

Here $h_g$ is the mean number of hops per axis, `mean(NODES) - 1`, and
`rclip(v, lo, hi)` rounds `v` to the nearest integer and clamps it to
`[lo, hi]`. `aco.auto_params(grid)` derives the budget entries from the
lattice; see [`ACO_PARAMETERS.md`](ACO_PARAMETERS.md) for the role and tuning
of each entry.

## Scripts cited in the paper

| Paper statement | Script |
| --- | --- |
| validation of the construction rules (turn-angle profiles, geodesic field, bidirectional construction) | `validate_construction.py` |
| segment filter against the baffles, implemented in the driver rather than in `grid.build_neighbours` | `run_staircase.py` |
| count of monotone routes and admissible fraction | `count_monotone.py` |
| random search at equal finite-element budget | `baseline_search.py` |
