"""
run_staircase.py
Second demonstration case: a NON-CONVEX (staircase / serpentine) design domain.

The domain is the bounding box with two thin baffles removed at two heights,
attached to opposite walls so their gaps fall on opposite sides. The straight
A->B line is blocked, so the tube must serpentine: out to one side to clear the
lower baffle, back to the other to clear the upper one. This is a standalone
driver; it does NOT modify run_spline.py or the core modules. It reuses the same
machinery as the single keep-out box in run_spline.py, generalised from one
forbidden box to a LIST of forbidden boxes:

  * the node mask removes any node that falls on either baffle;
  * the adjacency is filtered so no lattice edge (including the 18/26 diagonals)
    crosses a baffle;
  * a route is rejected if its smooth spline enters either baffle, the same cheap
    containment test used for the single box, applied to both.

Run:  python run_staircase.py
"""

import numpy as np
import plot_style  # noqa: F401  (sets Times New Roman for all figures)
from scipy.interpolate import splev
import grid as G
import aco as A
import fem_beam3d as fem
import geometry as geo
import plotting as P

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# --- design domain: bounding box, then two alternating baffles ---------------
# A staircase / serpentine channel. Two thin baffles span most of the box at two
# heights, attached to opposite walls so their gaps are on opposite sides: the
# lower baffle leaves a gap on the RIGHT, the upper one a gap on the LEFT. The
# tube is forced to go right to clear the first, then left to clear the second,
# so the straight A->B line is blocked and the non-convex domain genuinely shapes
# the route. Baffles are defined by the lattice layers they block (robust to the
# grid), and the matching physical boxes feed the spline-containment test.
BOX_SIZE = (0.300, 0.150, 0.300)   # box extents (Lx, Ly, Lz) [m]
NODES    = (23, 12, 23)               # nodes per axis (nx, ny, nz)
CONNECT  = 26                      # 6 (faces), 18 (+edges), 26 (+body diagonals)

# Baffles. The x-columns are DERIVED from NODES so that each baffle really touches the
# wall it is meant to be attached to. They used to be hardcoded as (0,1,2,3) and (3,4,5,6),
# which is only correct while NODES[0] == 7: raising the resolution left the upper baffle
# floating as an island in mid-domain, with a gap on BOTH sides, so the channel no longer
# forced the alternating left-right detour it is supposed to demonstrate.
GAP_FRACTION = 3.0 / 7.0           # share of the x-columns left open at each baffle
_NX = NODES[0]
_n_open = max(1, int(round(GAP_FRACTION * _NX)))
_n_block = _NX - _n_open
BAFFLE_Z_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)   # alturas das placas como frações de Lz
LO_IZ = int(round(BAFFLE_Z_FRACTIONS[0] * (NODES[2] - 1))); LO_IX = tuple(range(0, _n_block))
# upper baffle: blocks z-layer UP_IZ, attached to the RIGHT wall (ix=NODES[0]-1), gap on the left
UP_IZ = int(round(BAFFLE_Z_FRACTIONS[1] * (NODES[2] - 1))); UP_IX = tuple(range(_NX - _n_block, _NX))
assert 0 in LO_IX and (_NX - 1) in UP_IX, "each baffle must reach its wall"

# --- endpoints (lattice indices): opposite corners across the channel --------
A_NODE = (0, 0, 0)                              # lower-left, source (drive)
B_NODE = (NODES[0]-1, NODES[1]-1, NODES[2]-1)   # upper-right, support
# --- prescribed exit / arrival directions (optional) ----------------------------
# EXIT_DIR_A: direction AND sense in which the pipe must leave A (e.g. (0, 0, 1) =
# vertically upwards). The first lattice step is filtered by the turn-angle test
# against this virtual arrival direction, with its own cone half-angle. ARRIVE_DIR_B:
# the TRAVEL direction with which the pipe must enter B (e.g. (0, 0, -1) = coming
# down into B); the final step into B is filtered to the cone. On the 26-connected
# lattice the angles available from an axis are 0, 45, 54.7 and 90 deg, so a cone
# below 45 forces the exact axis, below 55 admits the near diagonals, 90 only
# forbids the opposite sense. Both cones are geometric requirements on the lattice:
# arriving downwards into B needs lattice nodes ABOVE B, so B cannot be on the top
# face of the domain; the script checks this before the search. None disables.
EXIT_DIR_A      = (0.0, 0.0, 1.0)# None    # e.g. (0.0, 0.0, 1.0)
EXIT_CONE_DEG   = 45.0
ARRIVE_DIR_B    = (0.0, 0.0, 1.0) #None    # e.g. (0.0, 0.0, -1.0)
ARRIVE_CONE_DEG = 45.0
TANGENT_CLAMP   = True    # also clamp the TANGENT of the fitted spline to EXIT_DIR_A /
                          # ARRIVE_DIR_B (geometry.END_TANGENTS). The lattice cone alone
                          # fixes the first/last lattice step, but the smoothing spline
                          # leaves the end tangents free (deviations of 20-25 deg were
                          # measured); the clamp moves the second and second-to-last
                          # control points onto the prescribed directions, so the as-built
                          # curve leaves A and enters B exactly as prescribed. Applies to
                          # every spline of the run, baseline included.

# --- length budget ----------------------------------------------------------
# The baffles block the straight diagonal, so the tube must serpentine; allow slack.
LMAX_FACTOR = 3.0
LMAX_ABS    = None

# --- tube cross section, material (steel) -----------------------------------
OUTER_RADIUS = 0.001575; THICKNESS = 0.0005
OD = 2 * OUTER_RADIUS
# Centreline self-clearance [m]. None: not enforced during the search (the winner's
# clearance is always measured and printed after the run). Set e.g. OD to enforce
# d_min >= OD inside geom_fn; the arc window pi*R_MIN excludes curvature-driven
# proximity, so a bend at the radius limit is never a false positive.
SELF_CLEARANCE = OD  #None #OD
E = 210e9; nu = 0.3; rho = 7850.0

# --- excitation and target bands --------------------------------------------
# One or several frequency bands to minimise: a list of (lo, hi) tuples. A single
# band is just a one-element list, e.g. BANDS = [(6000.0, 10000.0)].
BANDS = [(6000.0, 10000.0)]                     # [Hz]
# One weight per band, set by you. The objective is the weighted sum of the
# per-band input-power integrals: sum_k BAND_WEIGHTS[k] * integral_over_band_k.
# Bands at different frequencies carry very different power levels (a higher band
# is often orders of magnitude larger), so the raw integrals are not comparable
# on their own; these weights are how you decide how much each band counts.
BAND_WEIGHTS = (1.0,)
RAY   = (0.0, 2e-7)
EXCITATION      = "prescribed"
# Complex displacement phasors prescribed at A in the global x, y and z directions [m].
# Relative phases define coherent multidirectional motion.
PRESCRIBED_DISP = (1e-6, 1e-6, 1e-6)
PRESCRIBED_MASK = (True, True, True)
DRIVE           = (0.0, 1.0, 0.0)

# --- manufacturability ------------------------------------------------------
R_MIN    = 0.008

# --- search / multi-start / mesh --------------------------------------------
SEEDS     = [3, 5, 7, 11, 13]   # multi-start protocol of the paper: N_s = 5 explicit seeds
N_WORKERS = None
# Number of warm-start corridors (1 reproduces the old single-shortest-path seeding).
N_SEED_ROUTES = 1
# Prospecting levers, in order of safety. (1) More multi-start seeds: independent
# colonies, the best is kept, so quality can only match or improve; every seed holds
# its own route cache, so its FEM count is the true cost of that run. (2) More seed corridors
# (N_SEED_ROUTES above). (3) ACO_OVERRIDES={"q0_rewind_on_kick": True}: after a
# stagnation restart the q0 schedule is rewound, so the re-diversified pheromone is genuinely
# re-explored instead of exploited greedily; changes post-restart dynamics only.
# FITPACK spline smoothing. SMOOTH=0 interpolates every lattice waypoint. Positive
# values preserve the endpoints but allow the fitted curve to deviate from internal
# waypoints, which regularizes fine-grid zig-zags. On NODES=(12,8,12), for example,
# approximately 1 route in 120 passed at SMOOTH=0, whereas a modest positive value opened
# a much larger feasible set. Smoothing does not itself enforce manufacturability: the
# minimum radius is always checked on the completed fitted spline. NODES controls the
# routing scaffold; ELEM independently controls the FEM discretization.
SMOOTH    = 1e-4
# Per-waypoint RMS smoothing tolerance [m] (see geometry.fit_spline).
# REPORTED CONFIGURATION: None -> the absolute FITPACK bound SMOOTH above is used,
# matching the numerical examples of the paper. Set a length in metres (e.g. 0.0025)
# to switch to the per-waypoint form s = N*SMOOTH_RMS^2 -- appropriate when the
# lattice waypoints carry positional meaning of their own (clearance-critical
# corridors the tube should follow closely), since the admissible deviation is then
# prescribed per waypoint instead of diluted by route length. Calibrate the value
# with calibrate_smoothing.py (tighter fits retain curvature and can lower J
# substantially, at the cost of minimum-radius pass rate).
SMOOTH_RMS = None
geo.DEFAULT_SMOOTH_RMS = SMOOTH_RMS
ELEM      = 0.005
ELEM_REPORT = None        # beam element size [m] for the reported spectra, reactions and
                          # geometry figures of the winning route. None: the mesh-convergence
                          # study below picks it (coarsest of MESH_CANDIDATES within 2 % of
                          # the 1 mm reference). A number (e.g. 0.002) forces that size; the
                          # convergence table is still printed for reference.
HARMONIC_SOLVER = "modal"  # harmonic solver INSIDE the optimization loop:
                          #   "direct" - banded LU at every frequency (reference solver)
                          #   "modal"  - modal superposition of the free block: one eigenproblem
                          #              per route, O(n) per frequency; ~2x faster with the
                          #              complete basis (identical to direct), 4-10x with a
                          #              truncated basis plus residual-flexibility correction
MODAL_FCUT = 3.0          # "modal" only: keep modes below MODAL_FCUT x f_hi of each band
                          # (None = complete basis, exact). With 3.0 the error in J is
                          # below 1e-3; the reported spectra/reactions always use "direct".
SHOW      = True

# Smoke-test budget: small so the run finishes quickly. Remove or raise these
# (delete the three keys) for a production run; auto_params then scales to the grid.
# "n_eval_workers": "auto" evaluates the iteration's routes in parallel processes
# (identical results, lower wall-clock; auto-disabled inside the parallel multi-start
# pool). "slack_balance" (0..1) rations the detour budget along the route so ants
# keep curvature capability near B.
ACO_OVERRIDES = {"n_eval_workers": "auto", "bidirectional": 0.5, "slack_balance":0.5} #{"n_ants": 10, "max_iter": 25, "patience": 10}
# ============================================================================ #

mat = {"E": E, "G": E / (2.0 * (1.0 + nu)), "rho": rho}
sec = fem.pipe_section(outer_radius=OUTER_RADIUS, thickness=THICKNESS, nu=nu)

assert all(n >= 2 for n in NODES) and all(L > 0 for L in BOX_SIZE)
SPACING = tuple(BOX_SIZE[k] / (NODES[k] - 1) for k in range(3))
sx, sy, sz = SPACING
Lx, Ly, Lz = BOX_SIZE
MAX_TURN = geo.max_turn_from_pitch(min(SPACING), R_MIN)   # 45.5 deg for the 16 mm pitch

def mask(ix, iy, iz):
    if iz == LO_IZ and ix in LO_IX:   # lower baffle (gap on the right)
        return False
    if iz == UP_IZ and ix in UP_IX:   # upper baffle (gap on the left)
        return False
    return True

# Physical box bracketing one baffle: one z-layer thick, spanning its blocked
# columns, and extended to the wall on the attached side so a spline cannot slip
# around the edge. The same boxes feed the spline-containment test below, so the
# node mask and the containment are guaranteed consistent.
def baffle_box(ix_cols, iz_layer):
    xmin = (min(ix_cols) - 0.5) * sx
    xmax = (max(ix_cols) + 0.5) * sx
    if min(ix_cols) == 0:            # attached to the x=0 wall
        xmin = -1.0
    if max(ix_cols) == NODES[0] - 1: # attached to the x=Lx wall
        xmax = Lx + 1.0
    zc = iz_layer * sz
    lo = np.array([xmin, -1.0, zc - 0.5 * sz])
    hi = np.array([xmax, Ly + 1.0, zc + 0.5 * sz])
    return (lo, hi)

keepout_boxes = [baffle_box(LO_IX, LO_IZ), baffle_box(UP_IX, UP_IZ)]

def seg_in_keepout(p, q, boxes, n=7):
    """True if the straight segment p->q passes through any removed corner."""
    for t in np.linspace(0.0, 1.0, n + 2)[1:-1]:
        r = p + t * (q - p)
        for lo, hi in boxes:
            if np.all((r > lo) & (r < hi)):
                return True
    return False

n_full = NODES[0] * NODES[1] * NODES[2]
grid = G.build_grid(nx=NODES[0], ny=NODES[1], nz=NODES[2], spacing=SPACING, mask=mask)
adj_raw = G.build_neighbours(grid, connectivity=CONNECT)

# drop edges that cut a removed corner (keeps the 18/26 diagonals honest)
coords = grid["coords"]
adj = {}
n_edges_raw = n_edges_kept = 0
for i, nbrs in adj_raw.items():
    kept = []
    for (j, L) in nbrs:
        n_edges_raw += 1
        if not seg_in_keepout(coords[i], coords[j], keepout_boxes):
            kept.append((j, L)); n_edges_kept += 1
    adj[i] = kept

if A_NODE not in grid["id_of"] or B_NODE not in grid["id_of"]:
    raise ValueError("an endpoint falls on a baffle; move A_NODE/B_NODE or adjust the baffle layers.")
sid = grid["id_of"][A_NODE]; tid = grid["id_of"][B_NODE]

straight_AB = float(np.linalg.norm(grid["coords"][tid] - grid["coords"][sid]))
Lmax = LMAX_ABS if LMAX_ABS is not None else LMAX_FACTOR * straight_AB
if Lmax < straight_AB:
    raise ValueError("Lmax below the straight A-B distance; raise LMAX_FACTOR.")

# design-volume bounds and the containment tolerance for the as-built spline (see geom_fn)
GMIN = grid["coords"].min(axis=0)
GMAX = grid["coords"].max(axis=0)
DOMAIN_TOL = 0.5 * min(SPACING)

print("Configuration (staircase domain)")
print("  box        : %.0f x %.0f x %.0f mm" % (Lx*1e3, Ly*1e3, Lz*1e3))
print("  nodes      : %d x %d x %d -> %d active of %d (corners removed: %d)"
      % (NODES[0], NODES[1], NODES[2], len(coords), n_full, n_full - len(coords)))
print("  edges      : %d kept of %d (%d cut a corner)"
      % (n_edges_kept, n_edges_raw, n_edges_raw - n_edges_kept))
print("  baffles    : lower z-layer %d (gap right), upper z-layer %d (gap left)" % (LO_IZ, UP_IZ))
print("  A -> B     : straight %.0f mm | Lmax %.0f mm (%.2fx)"
      % (straight_AB*1e3, Lmax*1e3, Lmax/straight_AB))
print("  tube OD %.1f mm | bands %s Hz | R_min %.0f mm"
      % (OD*1e3, ", ".join("%.0f-%.0f" % (b[0], b[1]) for b in BANDS), R_MIN*1e3))

# per-band weights, set by the user (exactly one per band)
if len(BAND_WEIGHTS) != len(BANDS):
    raise ValueError("BAND_WEIGHTS must have exactly one weight per band in BANDS.")
band_weights = list(BAND_WEIGHTS)
print("  band weights : %s" % ", ".join("%.3g" % w for w in band_weights))
print()

def objective_spline(coords):
    nodes, elems, _ = geo.mesh_spline(coords, ELEM, smooth=SMOOTH)
    return fem.band_integrated_power(nodes, elems, mat, sec, BANDS, weights=band_weights,
                                     freq_mode="adaptive", drive_dir=DRIVE, rayleigh=RAY,
                                     excitation=EXCITATION, prescribed_disp=PRESCRIBED_DISP,
                                     prescribed_mask=PRESCRIBED_MASK,
                                     solver=HARMONIC_SOLVER, modal_fcut=MODAL_FCUT)

def geom_fn(coords):
    tck = geo.fit_spline(coords, smooth=SMOOTH)
    R, _ = geo.min_radius(tck)
    L = geo.total_length(tck)
    pts = np.column_stack(splev(np.linspace(0, 1, 400), tck))
    ok = True
    for lo, hi in keepout_boxes:
        if np.any(np.all((pts > lo) & (pts < hi), axis=1)):
            ok = False; break
    # Domain containment of the as-built spline. The fitted curve may depart from the
    # internal lattice waypoints and can therefore leave the box. The natural scale of
    # this excursion is the lattice pitch (measured: mean ~10 mm and max ~26 mm on the
    # 7x3x7 lattice, and ~5 mm on 12x8x12). Excursions up to half the smallest pitch are
    # tolerated; larger excursions are rejected, which also removes rare wide loops
    # introduced by the smoothing fit.
    if ok:
        exc = max(float(np.maximum(GMIN - pts, 0.0).max()),
                  float(np.maximum(pts - GMAX, 0.0).max()))
        ok = exc <= DOMAIN_TOL
    # Optional hard constraint: the centreline must keep a clearance to itself. A route
    # that is a simple path on the lattice can still cross itself in space (two face
    # diagonals of one cell intersect at the cell centre using four distinct nodes).
    if ok and SELF_CLEARANCE:
        d_self, *_ = geo.min_self_distance(tck, n=400, arc_gap=np.pi * R_MIN)
        ok = d_self >= SELF_CLEARANCE
    return L, R, ok   # 3rd value: out of BOTH baffles and inside the box (tolerance)

BASE = A.auto_params(grid, overrides=ACO_OVERRIDES, verbose=True)
params = dict(BASE); params["max_turn_deg"] = MAX_TURN; params["R_min"] = R_MIN

print(f"\nOptimizing the smooth geometry, turn<={MAX_TURN:.0f} deg, "
      f"spline length<={Lmax*1e3:.0f} mm, R>={R_MIN*1e3:.0f} mm, both baffles enforced, "
      f"box excursion<={DOMAIN_TOL*1e3:.0f} mm")
# warm start: seed the search with a few short, mutually distinct lattice routes. The
# shortest path reaches the target cheaply and biases construction toward short, direct
# routes; the extra corridors (forced through via-nodes far from the shortest path) keep
# the early search from committing to a single corridor. On a fine lattice a purely
# random construction rarely threads the serpentine, so without seeding the optimizer
# can find nothing. The shortest path always comes first, so reachability is never
# worse than with one corridor.
seed_routes = G.diverse_seed_paths(adj, grid["coords"], sid, tid, k=N_SEED_ROUTES)
if seed_routes and len(seed_routes) > 1:
    print("  warm start : %d seed corridors (shortest first)" % len(seed_routes))
if TANGENT_CLAMP and (EXIT_DIR_A is not None or ARRIVE_DIR_B is not None):
    geo.END_TANGENTS = (EXIT_DIR_A, ARRIVE_DIR_B)
# prescribed exit / arrival cones: wiring and a lattice feasibility check
def _cone_feasible(node_id, direction, cone_deg, leaving):
    """True if at least one lattice edge at `node_id` satisfies the cone: for the exit,
    the step A->j must lie within cone_deg of `direction`; for the arrival, the step
    j->B must lie within cone_deg of `direction`."""
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    c = grid["coords"]; ok = []
    for j, _L in adj[node_id]:
        step = (c[j] - c[node_id]) if leaving else (c[node_id] - c[j])
        step = step / np.linalg.norm(step)
        ok.append(np.degrees(np.arccos(np.clip(np.dot(step, u), -1.0, 1.0))) <= cone_deg + 1e-9)
    return any(ok)
if EXIT_DIR_A is not None:
    if not _cone_feasible(sid, EXIT_DIR_A, EXIT_CONE_DEG, leaving=True):
        raise RuntimeError("EXIT_DIR_A: no lattice edge leaves A within the %.0f deg cone around %s; "
                           "move A away from that face of the domain or widen the cone"
                           % (EXIT_CONE_DEG, EXIT_DIR_A))
    params["start_dir"] = EXIT_DIR_A
    params["start_max_turn_deg"] = EXIT_CONE_DEG
if ARRIVE_DIR_B is not None:
    if not _cone_feasible(tid, ARRIVE_DIR_B, ARRIVE_CONE_DEG, leaving=False):
        raise RuntimeError("ARRIVE_DIR_B: no lattice edge enters B within the %.0f deg cone around %s "
                           "(arriving downwards needs nodes above B); move B away from that face "
                           "of the domain or widen the cone" % (ARRIVE_CONE_DEG, ARRIVE_DIR_B))
    params["end_dir"] = ARRIVE_DIR_B
    params["end_max_turn_deg"] = ARRIVE_CONE_DEG
import time as _time
_t_search = _time.time()
r = A.optimize_multistart(grid, adj, sid, tid, Lmax, objective_spline,
                          params=params, seeds=SEEDS, geom_fn=geom_fn,
                          n_workers=N_WORKERS, verbose=True,
                          seed_paths=seed_routes if seed_routes else None,
                          log_path="staircase_aco_log.txt")
route = r["best_path"]
if route is None:
    import sys
    print("\n  No feasible route was found: the search returned nothing "
          "(feasible ants 0, no FEM evaluations), so there is nothing to post-process.")
    print("  On a refined lattice this usually means the smoke-test budget is too small "
          "for the grid, or the constraints are too tight. Try, in order: remove "
          "ACO_OVERRIDES so auto_params scales the budget to the grid; raise LMAX_FACTOR; "
          "relax R_MIN or MAX_TURN.")
    sys.exit(1)
tck = geo.fit_spline(G.path_coords(grid, route), smooth=SMOOTH)
L_spline = geo.total_length(tck)
R_achieved, R_pt = geo.min_radius(tck)
print(f"\n  power            : {r['best_power']:.4e}")
print(f"  spline length    : {L_spline*1e3:.1f} mm  (<= {Lmax*1e3:.0f} mm)")
print(f"  min bend radius  : {R_achieved*1e3:.1f} mm  ({R_achieved/OD:.1f}x OD, >= {R_MIN*1e3:.0f} mm)")
d_self, _, _, s_i, s_j = geo.min_self_distance(tck, n=800, arc_gap=np.pi * R_MIN)
print(f"  self-clearance   : {d_self*1e3:.1f} mm  ({d_self/OD:.1f}x OD; arc window pi*R_min = {np.pi*R_MIN*1e3:.0f} mm)"
      + ("" if SELF_CLEARANCE is None else f"  [enforced >= {SELF_CLEARANCE*1e3:.1f} mm]"))
print(f"  routes rejected  : {r['n_reject_length']} length, "
      f"{r['n_reject_radius']} radius, {r['n_reject_keepout']} corner")
print(f"  search           : {r['n_fem_calls']} FEM evals")
print(f"  search time      : {_time.time() - _t_search:.0f} s wall-clock for the whole search")
if EXIT_DIR_A is not None or ARRIVE_DIR_B is not None:
    from scipy.interpolate import splev as _splev
    if EXIT_DIR_A is not None:
        _u0 = np.asarray(EXIT_DIR_A, float); _u0 = _u0 / np.linalg.norm(_u0)
        _t0 = np.array(_splev(0.0, tck, der=1), float).ravel(); _t0 = _t0 / (np.linalg.norm(_t0) + 1e-15)
        print(f"  exit direction   : spline tangent at A is {np.degrees(np.arccos(np.clip(np.dot(_t0, _u0), -1, 1))):.1f} deg "
              f"from EXIT_DIR_A {tuple(EXIT_DIR_A)} (first lattice step within {EXIT_CONE_DEG:.0f} deg)")
    if ARRIVE_DIR_B is not None:
        _u1 = np.asarray(ARRIVE_DIR_B, float); _u1 = _u1 / np.linalg.norm(_u1)
        _t1 = np.array(_splev(1.0, tck, der=1), float).ravel(); _t1 = _t1 / (np.linalg.norm(_t1) + 1e-15)
        print(f"  arrival direction: spline tangent at B is {np.degrees(np.arccos(np.clip(np.dot(_t1, _u1), -1, 1))):.1f} deg "
              f"from ARRIVE_DIR_B {tuple(ARRIVE_DIR_B)} (last lattice step within {ARRIVE_CONE_DEG:.0f} deg)")

ms = r.get("multistart")
if ms and len(ms["seeds"]) > 1:
    print("  per-seed results (reported route = seed %d):" % ms["winning_seed"])
    print("    %6s %14s %8s %6s %6s %8s" % ("seed", "J", "FEM", "iters", "restarts", "time[s]"))
    for s_, J_, F_, I_, K_, T_ in zip(ms["seeds"], ms["best_powers"], ms["fem_calls"], ms["n_iter"], ms["n_stagnation_events"], ms["wall_s"]):
        print("    %6d %14s %8d %6d %6d %8.0f" % (s_, ("%.4e" % J_) if J_ is not None else "none", F_, I_, K_, T_))
    print("    sum of the per-seed times: %.0f s (serial-equivalent); wall-clock above is what the run took"
          % ms["wall_total_s"])
    print("    spread: min %.4e | median %.4e | max %.4e (max/min = %.2f)"
          % (ms["J_min"], ms["J_median"], ms["J_max"], ms["J_max"] / ms["J_min"]))

if HARMONIC_SOLVER != "direct":
    # the search ranked the routes with the modal solver; the reported objective of the
    # winner is re-evaluated with the direct solver on the same (search) mesh
    _nn, _ee, _ = geo.mesh_spline(G.path_coords(grid, route), ELEM, smooth=SMOOTH)
    J_direct = fem.band_integrated_power(_nn, _ee, mat, sec, BANDS, weights=band_weights,
                                         freq_mode="adaptive", drive_dir=DRIVE, rayleigh=RAY,
                                         excitation=EXCITATION, prescribed_disp=PRESCRIBED_DISP,
                                         prescribed_mask=PRESCRIBED_MASK, solver="direct")
    print(f"  direct check     : winner re-evaluated with the direct solver on the {ELEM*1e3:.1f} mm mesh: "
          f"J = {J_direct:.4e} (search value {r['best_power']:.4e}, deviation {abs(J_direct/r['best_power']-1):.1e})")

# computational-cost figure for this case (and persist the cost data)
import cost_figure
cost_figure.save_cost_data(r, "fig_staircase_cost_data.npz")
cost_figure.make_cost_figure(r, "fig_staircase_cost.pdf", show=SHOW)
P.plot_radius_profile(tck, "fig_staircase_radius_profile.png", r_min=R_MIN, show=SHOW)
print("Saved: fig_staircase_cost.pdf  (cost data in fig_staircase_cost_data.npz), fig_staircase_radius_profile.png")

# mesh convergence of the manufacturable geometry
# Spatial convergence only. The frequency quadrature must already be converged, otherwise
# the difference attributed to the mesh also contains the integration error: a uniform
# frequency grid of a couple of hundred points does NOT resolve the resonance widths at low
# frequency. Both the reference and the candidates therefore use the eigenfrequency-guided
# grid, whose cluster width follows the modal bandwidth. The reference mesh is finer than
# every candidate, and the fallback is a size that was actually tested.
print("\n  Mesh convergence (pick the beam element size; converged quadrature):")
MESH_CANDIDATES = [0.010, 0.005, 0.002]
MESH_REF = 0.5 * min(MESH_CANDIDATES)      # reference strictly finer than any candidate
_qkw = dict(freq_mode="adaptive", drive_dir=DRIVE, rayleigh=RAY, excitation=EXCITATION,
            prescribed_disp=PRESCRIBED_DISP, prescribed_mask=PRESCRIBED_MASK)
refn, refe, _ = geo.resample_by_length(tck, MESH_REF)
W_ref = fem.band_integrated_power(refn, refe, mat, sec, BANDS, weights=band_weights, **_qkw)
print(f"    reference: elem={MESH_REF*1e3:4.1f} mm | {len(refe):3d} elem | W={W_ref:.4e}")
chosen = None
for h in MESH_CANDIDATES:
    nn, ee, _ = geo.resample_by_length(tck, h)
    W = fem.band_integrated_power(nn, ee, mat, sec, BANDS, weights=band_weights, **_qkw)
    err = 100 * (W - W_ref) / W_ref
    flag = ""
    if chosen is None and abs(err) < 2.0:
        chosen = h; flag = "  <- adequate"
    print(f"    elem={h*1e3:4.1f} mm | {len(ee):3d} elem | W={W:.4e} | err {err:+5.1f}%{flag}")
if chosen is None:
    chosen = min(MESH_CANDIDATES)          # no size met the tolerance: take the finest TESTED one
    print(f"    no candidate within 2%; falling back to the finest tested size "
          f"({chosen*1e3:.1f} mm)")
if ELEM_REPORT is not None:
    print(f"    ELEM_REPORT = {ELEM_REPORT*1e3:.1f} mm overrides the convergence pick ({chosen*1e3:.1f} mm)")
    chosen = float(ELEM_REPORT)
print(f"    reported mesh: {chosen*1e3:.1f} mm")

# geometry figure: demarcate the allowed lattice nodes (the domain) and the tube
def plot_staircase_route(grid, route, tck, fname, sid, tid, boxes, r_min=None, r_pt=None, show=False,
                         baseline_pts=None, baseline_label="baseline", legend=True, title=None):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import geometry as geo
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False                      # honour the explicit zorder of each artist
    gmin = grid["coords"].min(0); gmax = grid["coords"].max(0)

    # allowed lattice nodes as a faint background; the baffles, not the nodes, show the domain shape
    C = grid["coords"] * 1e3
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=5, c="0.55", alpha=0.15, depthshade=False,
               edgecolor="none", zorder=1)

    # blocked regions (the baffles), clipped to the box
    for lo, hi in boxes:
        dlo = np.maximum(lo, gmin); dhi = np.minimum(hi, gmax)
        P._add_box(ax, (dlo, dhi))
    ax.plot([], [], [], color=P.OBSTACLE_EDGE, lw=P.OBSTACLE_EDGE_LW, label="keep-out")

    if baseline_pts is not None:
        bl = geo.eval_spline(geo.fit_spline(np.asarray(baseline_pts), smooth=0.0, end_tangents=None), 300) * 1e3
        ax.plot(bl[:, 0], bl[:, 1], bl[:, 2], "--", color="#f0a800", lw=1.4, alpha=0.9,
                label=baseline_label, zorder=3)

    wp = grid["coords"][np.asarray(route)] * 1e3
    ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], "--", color="#27496d", lw=1.0, marker="o", ms=3,
            label="ACO route (waypoints)", zorder=4)
    curve = geo.eval_spline(tck, 400) * 1e3
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "-", color="#b3122c", lw=3.0, label="smooth spline",
            zorder=6, path_effects=[pe.withStroke(linewidth=5.5, foreground="white")])
    ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], s=14, c="#27496d", edgecolor="white", linewidth=0.4, zorder=7)

    A = grid["coords"][sid] * 1e3; B = grid["coords"][tid] * 1e3
    ax.scatter(*A, s=95, c="#f0a800", edgecolor="k", zorder=8)
    ax.scatter(*B, s=95, c="#1b7837", edgecolor="k", zorder=8)
    if r_pt is not None:
        p = np.asarray(r_pt) * 1e3
        ax.scatter(*p, s=70, marker="x", c="k", zorder=9,
                   label=f"tightest bend R={r_min*1e3:.0f} mm" if r_min else None)

    if title:
        ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    if legend:
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    ext = gmax - gmin
    if ext.min() > 0:
        ax.set_box_aspect(tuple(ext))
    fig.tight_layout(); fig.savefig(fname, dpi=140)
    if not show:
        plt.close(fig)

# evolution of the input power over the optimization (best-so-far and iteration mean)
P.plot_convergence(r, "fig_staircase_convergence.png", show=SHOW)
print("Saved: fig_staircase_convergence.png  (band-integrated input power vs iteration)")

# reference baseline: a fixed-axis helix cannot thread these gaps, and a lattice route
# (shortest path or a random draw) turns too sharply at the baffle corners on a fine mesh
# and breaks the bend-radius limit. So the naive reference is a smooth serpentine built
# in space, threading the centre of each gap; it is feasible at any mesh resolution.
import baseline_routes as BL
base_pts, base_info = BL.serpentine_gap_baseline(grid["coords"][sid], grid["coords"][tid],
                                                 keepout_boxes, grid["coords"].min(0),
                                                 grid["coords"].max(0), Lmax, R_MIN,
                                                 smooth=SMOOTH, verbose=True)
if base_pts is not None and geo.END_TANGENTS != (None, None):
    print("  NOTE: the baseline route keeps free end tangents (dense curve, fitted with end_tangents=None);"
          " only the optimized route satisfies the prescribed exit/arrival directions.")
if base_pts is not None:
    bn, be, _ = geo.resample_by_length(geo.fit_spline(base_pts, smooth=SMOOTH, end_tangents=None), ELEM)
    W_base = fem.band_integrated_power(bn, be, mat, sec, BANDS, weights=band_weights,
                                       freq_mode="adaptive", drive_dir=DRIVE, rayleigh=RAY,
                                       excitation=EXCITATION, prescribed_disp=PRESCRIBED_DISP,
                                       prescribed_mask=PRESCRIBED_MASK)
    print("  serpentine baseline power : %.4e  (optimized %.4e, i.e. %.1fx lower than the baseline)"
          % (W_base, r["best_power"], W_base / r["best_power"]))
    P.plot_all_splines(grid, r["evaluated_paths"], route, "fig_staircase_all_splines.png", sid, tid,
                       smooth=SMOOTH, boxes=keepout_boxes, baseline_pts=base_pts,
                       baseline_label="serpentine baseline", show=SHOW)
else:
    print("  serpentine baseline       : no feasible route found; drawing routes without a baseline")
    P.plot_all_splines(grid, r["evaluated_paths"], route, "fig_staircase_all_splines.png", sid, tid,
                       smooth=SMOOTH, boxes=keepout_boxes, show=SHOW)

plot_staircase_route(grid, route, tck, "fig_staircase.png", sid, tid, keepout_boxes,
                     r_min=R_achieved, r_pt=R_pt, show=SHOW, baseline_pts=base_pts, baseline_label="serpentine baseline")

plot_staircase_route(grid, route, tck, "fig_staircase_view2.png", sid, tid, keepout_boxes,
                     r_min=R_achieved, r_pt=R_pt, show=SHOW, baseline_pts=base_pts, baseline_label="serpentine baseline", legend=False)

print("Saved: fig_staircase_all_splines.png  (every tested route faint, best solid)")

# one spline-cloud figure per seed (routes explored by that seed, its own best solid)
for _s, _ps in r.get("per_seed", {}).items():
    if len(r["per_seed"]) < 2 or _ps["best_path"] is None:
        continue
    P.plot_all_splines(grid, _ps["evaluated_paths"], _ps["best_path"], "fig_staircase_all_splines_seed%d.png" % _s, sid, tid,
                       smooth=SMOOTH, boxes=keepout_boxes, baseline_pts=base_pts, baseline_label="serpentine baseline", show=SHOW)
if len(r.get("per_seed", {})) > 1:
    print("Saved: fig_staircase_all_splines_seed<s>.png  (one spline cloud per seed; best of that seed solid)")

# domain-only presentation figures for the staircase case
P.plot_domain(grid, "fig_staircase_domain.png", sid, tid, boxes=keepout_boxes, show=SHOW)
print("Saved: fig_staircase_domain.png  (design domain, lattice and baffles only)")
if base_pts is not None:
    P.plot_domain(grid, "fig_staircase_domain_baseline.png", sid, tid, boxes=keepout_boxes,
                  baseline_pts=base_pts, baseline_label="serpentine baseline",
                  smooth=SMOOTH, show=SHOW)
    print("Saved: fig_staircase_domain_baseline.png  (domain with the serpentine baseline only)")

# input-power spectrum of the final tube (at the converged element size)
def spectrum(coords, freqs, end_tangents="global"):
    nn, ee, _ = geo.resample_by_length(geo.fit_spline(coords, smooth=SMOOTH, end_tangents=end_tangents), chosen)
    if EXCITATION == "prescribed":
        return fem.input_power_spectrum_prescribed(nn, ee, mat, sec, 0, len(nn)-1, freqs,
                                                   prescribed_disp=PRESCRIBED_DISP, rayleigh=RAY,
                                                   prescribed_mask=PRESCRIBED_MASK)
    return fem.input_power_spectrum(nn, ee, mat, sec, 0, DRIVE, len(nn)-1, freqs, rayleigh=RAY)
P.plot_frf(grid, {"best_path": route}, "fig_staircase_frf.png",
           mat, sec, BANDS, spectrum, freq_range=(0.0, 10000.0),
           baseline_coords=base_pts, baseline_label="serpentine baseline", show=SHOW,
           baseline_spectrum=lambda c, f: spectrum(c, f, end_tangents=None))

# reaction force/torque spectra at the clamped end B: the blocked load reaching B
if EXCITATION == "prescribed":
    import reaction_figure as RF
    nodes_r, elems_r, _ = geo.resample_by_length(tck, chosen)
    base_nodes = base_elems = None
    if base_pts is not None:
        base_nodes, base_elems, _ = geo.resample_by_length(geo.fit_spline(base_pts, smooth=SMOOTH, end_tangents=None), chosen)
    RF.compute_and_plot(nodes_r, elems_r, mat, sec, BANDS,
                        forces_path="fig_staircase_reaction_forces.pdf",
                        torques_path="fig_staircase_reaction_moments.pdf",
                        freq_range=(0.0, 10000.0), rayleigh=RAY, show=SHOW,
                        prescribed_disp=PRESCRIBED_DISP, prescribed_mask=PRESCRIBED_MASK,
                        base_nodes=base_nodes, base_elements=base_elems,
                        base_label="serpentine baseline")
    print("Saved: fig_staircase_reaction_forces.pdf, fig_staircase_reaction_torques.pdf")

print(f"\n  final element size: {chosen*1e3:.0f} mm")
print("Saved: fig_staircase.png, fig_staircase_frf.png")
if SHOW:
    P.show_interactive()
