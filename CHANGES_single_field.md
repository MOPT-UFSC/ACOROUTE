# Single-field revision (2026-10-05)

Alignment of the package with the manuscript: one pheromone field, multi-start
protocol with N_s = 5 explicit seeds and per-seed route caches.

## aco.py
- Second pheromone field (sigma) removed: `build_one_ant(grid, adj, tau, ...)`,
  `deposit(tau, ants, rho, params, best)`, `deposit_elitist(tau, best, params)`.
  Transition weight is `tau**alpha`; `beta`, `sigma0`, `Q_sigma`, `sigma_bounds`
  and `literal_slide_denominator` no longer exist in `default_params`/`auto_params`.
  The result dict no longer carries `"sigma"`.
- Behaviour is bit-identical to the previous code with `beta = 0` (same seed ->
  same best_hist, best_path, tau, n_fem_calls, restart iterations); verified on a
  synthetic objective with and without stagnation restarts.
- `optimize_multistart`: the serial path now gives every seed its own route cache
  (as the parallel path already did), so `n_fem_calls` of a seed is the true cost
  of that run. `result["multistart"]` gains `fem_calls`, `n_iter`,
  `n_stagnation_events` (per seed) and `J_min`, `J_median`, `J_max`.
- Restart comment updated to the four conditions actually implemented.

## run_spline.py, run_staircase.py, run_sphere.py
- `SEEDS = [3, 5, 7, 11, 13]` (paper protocol). After the search summary a
  per-seed table (J, FEM, iterations, kicks) and the min/median/max spread are
  printed; these feed the spread reported with each case.

## Callers of build_one_ant
- calibrate_smoothing.py, make_fig_balanced_budget.py, make_paper_figures.py,
  make_schematic_figures.py, validate_fixes.py, validate_slack_balance.py:
  second positional argument removed.

## Removed
- run_ablation.py (two-field ablation; no longer part of the study)
- aco_old.py, run_spline_old.py, reaction_figure_old.py

## Parallel seeds on any platform: run_campaign.py
- `python run_campaign.py run_spline.py [n_proc] [seeds]` runs the seeds of a run_*.py
  script in separate processes with the "spawn" start method (works on Windows),
  then executes the script's own post-processing on the combined result. The
  run_*.py scripts are unchanged; results are identical to the serial run with the
  same SEEDS (per-seed cache in both). One log per seed: <log>_seed<s>.txt.
- `aco.combine_multistart(results, seeds)` factored out of optimize_multistart and
  shared with run_campaign.py.
- BLAS threads are pinned to 1 per process so N processes use N cores.
- Each worker prints its progress to the shared console with a "[seed N]" prefix
  (`every=K` iterations, default 10; `print_every` param of aco.optimize).

## ELEM_REPORT
- run_spline.py / run_staircase.py / run_sphere.py: `ELEM_REPORT = None` next to `ELEM`.
  None keeps the mesh-convergence pick for the reported spectra, reactions and
  geometry figures; a number (e.g. 0.002) forces that element size, with the
  convergence table still printed for reference.

## One spline-cloud figure per seed
- `combine_multistart` adds `result["per_seed"][seed] = {evaluated_paths, best_path,
  best_power, n_fem_calls}`.
- run_spline.py / run_staircase.py / run_sphere.py: after the campaign figure
  (union of all seeds, global best solid) one figure per seed is saved,
  `<stem>_seed<s>.png`, with the routes explored by that seed and its own best solid.
  Only when more than one seed was run.

## Modal harmonic solver (optional, search loop only)
- fem_beam3d.py: `modal_basis_prescribed()` and `input_power_spectrum_modal()`; the
  free block (A translations prescribed, B clamped) is diagonalized once per route
  and the sweep is vectorized over the frequencies. `evaluate_mesh_power(...,
  solver="direct"|"modal", modal_fcut=3.0)`: with modal_fcut=None the complete basis
  is used (identical to the direct solver to round-off); with a number, the modes
  below modal_fcut x f_hi are kept and the rest enters through the residual-
  flexibility (mode-acceleration) correction, which is essential for prescribed
  displacement (without it J is wrong by factors). The adaptive frequency grid is
  built from the same eigenvalues, so no separate eigenproblem is needed.
- run_spline.py / run_staircase.py / run_sphere.py: `HARMONIC_SOLVER = "direct"` and
  `MODAL_FCUT = 3.0` next to ELEM_REPORT; they act on the objective of the search
  loop only. Reported spectra, reactions and the mesh-convergence study keep the
  direct solver.
- validate_modal.py: direct vs modal (full, 3 f2, 2 f2) on admissible routes of a
  run_*.py case. Measured on run_spline.py (5 mm, ~1000 DOF):
    0-1 kHz : direct 0.57 s/route | full 0.31 s (3e-8) | 3 f2 0.12 s (4e-5) | 2 f2 0.10 s (6e-5)
    3-8 kHz : direct 0.60 s/route | full 0.28 s (5e-12) | 3 f2 0.19 s (6e-4) | 2 f2 0.15 s (3e-3)
  Ranking of the routes preserved in every case.
- The per-seed spline-cloud figures are now also shown on screen (show=SHOW).

## Timing
- `aco.optimize` returns `wall_s` and writes the wall time in the last line of the log;
  `combine_multistart` adds `wall_s` per seed and `wall_total_s` (serial-equivalent).
- run_*.py print "search time: N s wall-clock for the whole search" (parallel or
  serial), a time[s] column in the per-seed table and the sum of per-seed times.

## Prescribed exit / arrival directions (optional)
- run_spline.py / run_staircase.py (run_sphere.py already had the cones): `EXIT_DIR_A`,
  `EXIT_CONE_DEG`, `ARRIVE_DIR_B`, `ARRIVE_CONE_DEG` (None = off). EXIT_DIR_A is the
  direction in which the pipe leaves A; ARRIVE_DIR_B the travel direction with which
  it enters B (e.g. (0,0,1) = arriving while going up). The first/last lattice step is
  filtered to the cone (aco.py start_dir/end_dir, already supported, also for the
  reversed ants). A feasibility check on the lattice runs before the search (arriving
  downwards into a node on the top face is impossible and is reported as such).
- `TANGENT_CLAMP = True`: geometry.fit_spline now honours `geometry.END_TANGENTS`
  (set by the scripts): the second and second-to-last control points are moved onto
  the prescribed directions, so the as-built curve leaves A and enters B exactly as
  prescribed (0.0 deg measured, against 20-25 deg with the lattice cone alone). The
  curve changes only in the first/last knot span; all checks and the FEM run on the
  clamped curve. `fit_spline(..., end_tangents=None)` opts out: the template
  baselines (helix / serpentine, dense point sets whose end spans are millimetres
  long) are fitted with free tangents everywhere (baseline_routes.py, plotting.py,
  run_*.py), and the scripts print a NOTE saying so when directions are prescribed.
- After the search the scripts print the spline tangent angles at A and B relative
  to the prescribed directions.
- plot_frf(..., baseline_spectrum=...): the baseline FRF is evaluated with free end
  tangents (spectrum(c, f, end_tangents=None) in the run scripts) while the optimized
  route keeps the prescribed-direction clamp; previously the dense baseline curve
  was clamped in the FRF evaluation.
