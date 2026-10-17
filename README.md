<p align="center">
  <img src="ACOROUTE_FIG.png"
       alt="ACOROUTE — ant colony optimization for pipe routing"
       width="900">
</p>

**Ant colony optimization of small-bore pipe routes for reduced active input power.**

This repository implements the research code associated with the manuscript:

> **Ant colony optimization of small-bore pipe routes for reduced active input power**
> Olavo M. Silva, Eduardo L. Cardoso, Miguel M. Neves, and Arcanjo Lenzi.
> Submitted to the *Journal of Sound and Vibration*.

The method constructs routes on a three-dimensional lattice, converts each completed route into a smoothing B-spline with pinned endpoints and end tangents, checks the admissibility conditions (developed length, minimum bend radius, containment, keep-out and, when active, self-clearance), meshes the centreline with 3D Timoshenko beam elements, and minimizes the active input power integrated over one or more frequency bands. The three filters that act during the construction (turn angle, length lower bound, keep-out) are described in the paper as admissibility filters; the complete checks are applied to the fitted spline.

The optimizer combines the constructive principles of Ant System, the exploration schedule of Ant Colony System and the iteration-best reinforcement, bounded pheromone trail and stagnation restart of MAX–MIN Ant System, on a single pheromone field over the undirected lattice edges, with mechanisms tailored to the geometry of the problem:

- **balanced detour budget** (`slack_balance`, paper symbol ξ): the slack between the straight A–B distance and the length budget is rationed along the route via the geodesic distance field, so ants keep the same turning capability near B as near A;
- **bidirectional construction** (`bidirectional`): half of each iteration's ants build their route from B toward A; completed routes are canonicalized to the A→B orientation before evaluation, so the excitation/support roles and the objective are unchanged;
- **iteration-best deposit with best-so-far elitism**, a linear schedule of the exploitation probability over an exploration window, optional warm-start corridors, and a stagnation restart that fires only after the colony has concentrated.

The objective is non-additive and is available only after a complete geometric and finite-element evaluation of the route. Inside the loop it is computed on an eigenfrequency-guided frequency grid, with modal superposition and residual flexibility in the reported runs, or with the direct banded solver used as reference.

## Start here

| Goal | File |
|---|---|
| Run the main box-domain example | [`run_spline.py`](run_spline.py) |
| Run the non-convex staircase example | [`run_staircase.py`](run_staircase.py) |
| Run the spherical domain with keep-out and end cones | [`run_sphere.py`](run_sphere.py) |
| Run the multi-start seeds of any case in parallel | [`run_campaign.py`](run_campaign.py) |
| Match code names to paper symbols | [`docs/CODE_TO_PAPER.md`](docs/CODE_TO_PAPER.md) |
| Understand and tune every ACO parameter | [`docs/ACO_PARAMETERS.md`](docs/ACO_PARAMETERS.md) |

## Requirements

- Python 3.10 or newer (the paper's results were produced with Python 3.12)
- NumPy
- SciPy
- Matplotlib

Tested with NumPy 2.4, SciPy 1.17 and Matplotlib 3.10; earlier releases providing the same routines work. Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

Run the primary example:

```bash
python run_spline.py
```

Run the same case with its multi-start seeds in parallel processes (works on Windows, no fork required):

```bash
python run_campaign.py run_spline.py
```

For a headless or batch run, set:

```python
SHOW = False
```

in the selected driver script.

The driver scripts are intentionally procedural: the editable problem definition is grouped at the top of each file. The most frequently changed entries are:

```python
BOX_SIZE, NODES, CONNECT            # domain and lattice resolution
EXCLUDE_CENTER, ENFORCE_KEEPOUT     # optional keep-out obstacle
A_NODE, B_NODE                      # endpoints (lattice indices)
EXIT_DIR_A, ARRIVE_DIR_B,
EXIT_CONE_DEG, ARRIVE_CONE_DEG,
TANGENT_CLAMP                       # prescribed exit/arrival directions
LMAX_FACTOR or LMAX_ABS             # developed-length budget
OUTER_RADIUS, THICKNESS             # pipe cross section
E, nu, rho                          # material
BANDS, BAND_WEIGHTS, RAY            # target bands and damping
EXCITATION, PRESCRIBED_DISP,
PRESCRIBED_MASK, DRIVE              # excitation at A
MAX_TURN, R_MIN                     # manufacturability
SMOOTH, SMOOTH_RMS, ELEM            # spline fit and beam mesh
HARMONIC_SOLVER, MODAL_FCUT         # in-loop harmonic solver
SEEDS, N_WORKERS                    # multi-start protocol
ACO_OVERRIDES                       # optimizer tuning
```

See [`docs/CODE_TO_PAPER.md`](docs/CODE_TO_PAPER.md) for the corresponding paper notation and [`docs/ACO_PARAMETERS.md`](docs/ACO_PARAMETERS.md) for the optimizer settings.

## Numerical examples of the paper

Each example of the manuscript is one of the three drivers with the settings below. Every case uses the 26-connected lattice, `R_MIN = 0.008`, `SMOOTH = 1e-4`, `HARMONIC_SOLVER = "modal"` with `MODAL_FCUT = 3.0`, `SEEDS = [3, 5, 7, 11, 13]` and `ACO_OVERRIDES = {"bidirectional": 0.5, "slack_balance": 0.5}` plus `"n_eval_workers": "auto"`. The ACO budgets follow from the lattice through `aco.auto_params`.

| Paper case | Driver | `NODES` | `BANDS` [Hz] | Keep-out | `LMAX_FACTOR` | Other |
|---|---|---|---|---|---|---|
| Representative, free cube | `run_spline.py` | (15, 15, 15) | [(500, 1500)] | `EXCLUDE_CENTER = False` | 1.7 | as shipped |
| Free cube, low band | `run_spline.py` | (15, 15, 15) | [(1000, 3000)] | `False` | 1.7 | |
| Free cube, mid band | `run_spline.py` | (20, 20, 20) | [(5000, 7000)] | `False` | 1.7 | |
| Free cube, high band | `run_spline.py` | (23, 23, 23) | [(6000, 10000)] | `False` | 1.7 | |
| Free cube, two bands | `run_spline.py` | (21, 21, 21) | [(1000, 2000), (6000, 8000)] | `False` | 1.7 | `BAND_WEIGHTS = (5.0, 1.0)` |
| Central box | `run_spline.py` | (15, 15, 15) | [(1000, 3000)] | `EXCLUDE_CENTER = 0.5` | 1.7 | |
| Staircase | `run_staircase.py` | (23, 12, 23) | [(6000, 10000)] | two baffles, as shipped | 3.0 | `SELF_CLEARANCE = OD`, one warm-start corridor |
| Ball | `run_sphere.py` | (21, 21, 21), masked | [(4000, 8000)] | `EXCLUDE_CENTER = 0.4` | 4 | end cones at the ball surface, as shipped |

The drivers are shipped with the representative, staircase and ball settings. The box-domain variants are obtained by editing the entries listed above in `run_spline.py`; the same list is repeated in a comment at the top of that file.

## Prescribed complex motion

`PRESCRIBED_DISP` accepts real or complex harmonic amplitudes. With the convention
**u**(t) = Re{**û** e^(iωt)}, this example imposes a 90-degree phase difference between the `x` and `y` translations:

```python
PRESCRIBED_DISP = (
    1.0e-6,
    1.0e-6j,
    0.0,
)
PRESCRIBED_MASK = (True, True, True)
```

A zero value with a `True` mask entry is held at zero. A `False` mask entry leaves that translation free to respond.

## Outputs

Depending on the selected driver and options, the scripts save files such as:

- `spline_aco_log.txt`: iteration-level diagnostics;
- `fig_cost.png`: computational-cost history;
- `fig_radius_profile.png`: radius of curvature along the best centreline;
- `fig_spline_manufacturable.png`: optimized geometry and beam mesh;
- `fig_spline_convergence.png`: best-so-far and iteration histories;
- `fig_all_splines.png`: evaluated route cloud and best route;
- `fig_spline_manufacturable_frf.png`: input-power spectrum;
- `fig_reaction_forces.png` and `fig_reaction_torques.png`: blocked terminal reactions;
- IGES/STEP export of the winning centreline (`cad_export.py`, `EXPORT_CAD = True`).

Generated figures, logs, caches, data files and CAD exports are ignored by the supplied `.gitignore`.

## Reproducing the paper sections

| Manuscript topic | Script |
|---|---|
| Beam element validation (analytical frequencies, energy balance) | `validate_fem.py` |
| Modal superposition vs direct banded solver | `validate_modal.py` |
| Construction-rule validation (turn-angle profiles, geodesic field, bidirectional) | `validate_construction.py` |
| Search-space size and monotone-route count | `count_monotone.py` |
| Random-search benchmark and search-level ablations at equal FEM budget | `baseline_search.py` |
| Cost of one iteration by stage (construction, geometry checks, finite elements) | `time_iteration.py` |
| Smoothing-tolerance calibration | `calibrate_smoothing.py` |
| Balanced-budget figure | `make_fig_balanced_budget.py` |
| Paper and schematic figures | `make_paper_figures.py`, `make_schematic_figures.py`, `cost_figure.py`, `reaction_figure.py` |
| Self-clearance test suite | `validate_self_clearance.py` |
| Regression suites (construction, evaporation, objective, detour budget) | `validate_fixes.py`, `validate_slack_balance.py` |

`validate_construction.py` and `count_monotone.py` read the problem definition of a `run_*.py` case without modification, ensuring consistency between the validation and the reported runs.

## Repository layout

```text
aco.py                       ACO construction, update, restart and multi-start
fem_beam3d.py                3D Timoshenko beam FEM, direct banded and modal solvers
geometry.py                  smoothing spline, end tangents, exact min radius, meshing
grid.py                      lattice, adjacency, keep-out and geodesic distances
run_spline.py                main box-domain driver
run_staircase.py             non-convex staircase driver
run_sphere.py                spherical domain with keep-out and end cones
run_campaign.py              parallel multi-seed campaign runner (Windows-safe)
plotting.py                  geometry, convergence and spectrum plots
plot_style.py                shared figure style
baseline_routes.py           reference route generators
baseline_search.py           random and ablation baselines at equal FEM budget
cad_export.py                IGES/STEP export of the winning centreline
cost_figure.py               computational-cost plots
reaction_figure.py           blocked force and torque spectra
count_monotone.py            DP enumeration of monotone routes, feasible fraction
count_routes.py              route-count utilities
calibrate_smoothing.py       smoothing-tolerance sweep
make_fig_balanced_budget.py  balanced detour-budget figure
make_paper_figures.py        manuscript figure generation
make_schematic_figures.py    schematic figure generation
validate_fem.py              FEM and energy-balance checks
validate_modal.py            modal solver vs direct solver
validate_construction.py     turn-angle profiles along the route
validate_self_clearance.py   self-clearance test suite
validate_slack_balance.py    balanced detour-budget checks
validate_fixes.py            broader regression checks
verify_v6_winner.py          optional post-processing verification
time_iteration.py            cost of one ACO iteration by stage
docs/                        article/code cross-reference and user guides
```

## Reproducibility notes

- A run is exactly reproducible for a fixed seed; the seed is an arbitrary label, not a tuning parameter. Multi-start seeds are independent colonies and each holds its own route cache, so the reported finite-element count of a seed is the cost of that run alone.
- Parallel objective evaluation (`n_eval_workers`) and the parallel campaign runner change wall-clock time only; the search trajectory is identical to the serial execution.
- Integer rounding follows Python's round-half-to-even convention; for the ant populations used in the paper, `round(n_ants/2)` equals `floor(n_ants/2)`.

For strict repeatability of one run, use an explicit seed and serial execution:

```python
SEEDS = [3]
N_WORKERS = 1
```

Optionally restrict numerical libraries to one thread before launching Python:

Linux/macOS:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Windows PowerShell:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
```

## Scope and limitations

This is research code, not a certified pipe-design tool. In particular:

- the pipe is represented by a beam centreline, not a shell model;
- the current objective is structural active input power;
- internal fluid loading and fluid-structure coupling are not included in this repository;
- geometric feasibility is checked numerically on sampled splines;
- exhaustive global optimality is not certified;
- users remain responsible for stress, fatigue, collision, assembly, and manufacturing checks required by their application.

## About this repository

The research code in this repository was written by the authors. The authors uploaded the original source files and organized the repository with the assistance of [Claude Code](https://claude.com/claude-code) (Anthropic), which structured the presentation (this README, the article/code cross-reference in [`docs/`](docs/), and the reproduction map, including internal documentation) to ease the use of the code and its association with the paper. The method, implementation and results are the authors' own.

## Citation

Use the repository's [`CITATION.cff`](CITATION.cff) through GitHub's **Cite this repository** interface. Until the associated article has a final DOI, a software citation can be written as:

```bibtex
@software{silva_acoroute_2026,
  author  = {Silva, Olavo M. and Cardoso, Eduardo L. and Neves, Miguel M. and Lenzi, Arcanjo},
  title   = {ACOROUTE: Ant colony optimization of small-bore pipe routes for reduced active input power},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/MOPT-UFSC/ACOROUTE}
}
```

The paper citation should be added here after acceptance and DOI assignment.

## License

This project is released under the MIT License — see [`LICENSE`](LICENSE).

<p align="center">
   <img src="https://github.com/MOPT-UFSC/ACOROUTE/docs/MOPT4.PNG?raw=true" alt="MOPT logo" width="1100"/>


