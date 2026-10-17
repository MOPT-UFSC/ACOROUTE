"""
run_sphere.py
Variant of run_spline.py with a SPHERICAL admissible domain: the lattice remains a
regular Cartesian grid, but only the nodes inside a ball are retained as candidate
waypoints, and the completed spline must stay inside the same ball (radial
containment). An internal rectangular keep-out obstacle is still supported.

Two findings drive the recommended workflow:

  (1) The band-integrated input power is highly sensitive to the exact geometry.
      Optimising the piecewise-linear path and then splining the winner
      (post-processing) moves the resonances and ruins the result (PART 1).

  (2) Geometric smoothness is not equivalent to manufacturability. With SMOOTH=0,
      the B-spline interpolates all waypoints; with SMOOTH>0, it preserves the fixed
      endpoints but may deviate from the internal waypoints. In either case, zig-zag
      routes can produce bends that are too tight. A turn-angle pre-filter reduces such
      candidates, and the minimum bend radius is checked on the completed fitted spline.
      The smooth geometry must therefore be evaluated inside the optimization loop
      (PART 2).

Run:  python run_sphere.py
"""

import numpy as np
from scipy.interpolate import splev
import grid as G
import aco as A
import fem_beam3d as fem
import geometry as geo
import plotting as P
import cad_export as CAD

# ============================================================================ #
# CONFIGURATION  --  edit these to change the problem. All lengths in metres.
# ============================================================================ #

# --- design domain ----------------------------------------------------------
# Set the box by its physical size and the number of nodes per axis. The node
# spacing (distance between adjacent nodes) is computed automatically as
# size/(nodes-1), per axis, so you control the box and the resolution directly
# and never have to keep them consistent by hand.
BOX_SIZE = (0.300, 0.300, 0.300)   # box extents (Lx, Ly, Lz) [m]
NODES    = (21, 21, 21)               # nodes per axis (nx, ny, nz); each >= 2
CONNECT  = 26                      # route directions per node: 6 (faces), 18 (+edges), 26 (+body diagonals)
# examples:
#   bigger box, same node count : BOX_SIZE = (0.40, 0.40, 0.40)               -> coarser grid
#   finer grid, same box        : NODES    = (9, 9, 9)                        -> spacing 37.5 mm
#   flat wide box               : BOX_SIZE = (0.40, 0.40, 0.12); NODES = (8, 8, 3)

# --- spherical admissible domain ---------------------------------------------
# Only lattice nodes inside the ball of radius SPHERE_RADIUS about SPHERE_CENTER are
# retained, and the as-built spline is checked against the same ball with the usual
# half-pitch tolerance (radial test in geom_fn). BOX_SIZE is then just the bounding
# box that positions the Cartesian lattice.
SPHERE_CENTER = None      # [m] ball centre; None -> centre of BOX_SIZE
SPHERE_RADIUS = None      # [m] ball radius; None -> inscribed ball, 0.5*min(BOX_SIZE)

# --- keep-out region (optional obstacle the tube must avoid) -----------------
# Carve a central block of candidate nodes out of the cloud so no waypoint can
# land there; the tube is then forced to route around it. Give the fraction of
# each axis to remove, centered. None disables; a scalar applies to all axes.
#   EXCLUDE_CENTER = 0.4              -> remove the central 40% cube
#   EXCLUDE_CENTER = (0.5, 0.5, 0.2) -> a central slab
EXCLUDE_CENTER  = 0.4
# Removing nodes only stops waypoints inside the region; the smooth spline could
# still clip its corner. With this on, routes whose spline enters the box are
# also rejected (a true keep-out), tested cheaply before any FEM solve.
ENFORCE_KEEPOUT = True

# --- endpoints ---------------------------------------------------------------
# Give lattice index triples (they must be retained nodes inside the ball), or leave
# both as None to auto-select the two retained nodes with extreme projection along
# ENDPOINT_AXIS, i.e. the antipodal poles of the ball along that direction.
A_NODE = None
B_NODE = None
ENDPOINT_AXIS = (1.0, 1.0, -1.0)
# Optionally start the route at a chosen corner of the central keep-out box: a sign
# triple picks the corner, e.g. (-1,-1,-1) = the lo corner, (+1,-1,+1) = hi_x, lo_y,
# hi_z. A is snapped to the retained lattice node nearest to that corner (exact when
# the box faces are aligned with the lattice, i.e. EXCLUDE_CENTER = 1 - 2k/(N-1)).
# None disables; requires EXCLUDE_CENTER. B keeps the rule above (B_NODE or the pole
# along ENDPOINT_AXIS).
A_KEEPOUT_CORNER = (+1, -1, +1) #None

# --- exit direction at A (optional) ------------------------------------------
# Prescribe the direction AND sense in which the pipe leaves A: the first lattice step
# is filtered by the same turn-angle test used along the route, measured against
# EXIT_DIR_A as a virtual arrival direction at the start node. On the 26-connected
# lattice the available angles from an axis are 0 (the axis itself), 45 (edge
# diagonals), 54.7 (body diagonals) and 90 deg (in-plane moves), so EXIT_CONE_DEG < 45
# forces the exact axis, < 55 admits the near diagonals, and 90 only forbids leaving
# against the prescribed sense. None disables.
EXIT_DIR_A    = (0.0, 0.0, 1.0) #None
EXIT_CONE_DEG = 45.0
# Prescribe the arrival at B analogously: ARRIVE_DIR_B is the TRAVEL direction with
# which the pipe must enter B (the tangent at arrival; a flange whose outward normal
# is +n receives the pipe arriving along -n). The final lattice step into B is
# filtered to the cone; a route reaching B outside it simply backtracks. None disables.
ARRIVE_DIR_B    = (0.0, 0.0, -1.0) #None    # e.g. (0.0, 0.0, -1.0)
ARRIVE_CONE_DEG = 45.0
TANGENT_CLAMP   = True    # also clamp the spline END TANGENTS to EXIT_DIR_A / ARRIVE_DIR_B
                          # (geometry.END_TANGENTS); the lattice cone alone leaves the
                          # smoothing-spline tangent free (20-25 deg deviations measured)

# --- length budget ----------------------------------------------------------
# Maximum developed (arc) length of the tube, set as a multiple of the straight
# A-B distance so it auto-scales with the box. 1.0 = straight line (no slack,
# usually no feasible route); larger = more routing freedom. Set LMAX_ABS to a
# value in metres to override the factor (leave None to use the factor).
LMAX_FACTOR = 4         # multiple of the straight A-B distance
LMAX_ABS    = None        # e.g. 0.900 to force an absolute limit [m]

# --- tube cross section (circular) ------------------------------------------
OUTER_RADIUS = 0.001575   # [m]
THICKNESS    = 0.0005     # [m] wall thickness
OD = 2 * OUTER_RADIUS     # outer diameter [m]; used only for reporting (e.g. "R = 3.8x OD")
# Centreline self-clearance [m]. None: not enforced during the search (the winner's
# clearance is always measured and printed after the run). Set e.g. OD to enforce
# d_min >= OD inside geom_fn; the arc window pi*R_MIN excludes curvature-driven
# proximity, so a bend at the radius limit is never a false positive.
SELF_CLEARANCE = None #OD

# --- material (steel) -------------------------------------------------------
E = 210e9; nu = 0.3; rho = 7850.0   # Young [Pa], Poisson [-], density [kg/m^3]

# --- excitation and evaluation ----------------------------------------------
# One or several target bands to minimise: a list of (lo, hi) tuples. A single
# band is just a one-element list, e.g. BANDS = [(6000.0, 10000.0)].
BANDS = [(4000.0, 8000.0)]    # [Hz]
# One weight per band, that YOU set. The objective is the weighted sum of the
# per-band input-power integrals: sum_k BAND_WEIGHTS[k] * integral_over_band_k.
# Bands at different frequencies carry very different raw power, so these weights
# are how you trade between them; with one band the weight is immaterial.
BAND_WEIGHTS = (1.0,)
RAY   = (0.0, 2e-7)         # Rayleigh damping (alpha on M, beta on K)

# Excitation at the start node A. "prescribed" imposes selected translational
# displacement phasors and measures the active input power delivered by their reaction
# forces; "force" applies a point-force phasor along DRIVE instead. Relative complex
# phases in PRESCRIBED_DISP define coherent multidirectional motion. Scaling every
# prescribed component by the same complex factor scales J by its magnitude squared
# without changing the optimum route.
EXCITATION      = "prescribed"        # "prescribed" or "force"
PRESCRIBED_DISP = (1e-6, 1e-6, 1e-6)  # complex displacement phasors (x, y, z) at A [m]
# Which translations at A are prescribed. False leaves that direction FREE to respond
# (a support that imposes some directions and lets the node slide in the others);
# a prescribed component set to 0.0 is instead HELD at zero (clamped in that direction).
#   (True,  True,  True)  -> all three prescribed; zero components are clamped
#   (False, True,  False) -> only y prescribed; x and z free to respond
PRESCRIBED_MASK = (True, True, True)
DRIVE           = (0.0, 1.0, 0.0)     # force direction at A, used only when EXCITATION == "force"

# --- manufacturability ------------------------------------------------------
R_MIN    = 0.008          # hard min bend radius [m] on the as-built spline (bench-bending limit, ~2.5x OD)

# --- search / multi-start ---------------------------------------------------
RUN_PART1 = False         # PART 1 (polyline-objective demo) only feeds the sensitivity
                          # figure; skip it for result runs (halves the optimisation cost)
SEEDS     = [3, 5, 7, 11, 13]  # independent ACO runs (multi-start protocol of the paper:
                          # N_s = 5 explicit seeds, best-of-seeds reported together with
                          # the min/median/max of the converged objectives). Accepted forms:
                          #   5        -> five runs, seeds 0..4
                          #   [3]      -> one run with seed 3
                          #   [10,20]  -> two runs, those exact seeds (exact reproducibility)
                          # Every seed holds its own route cache, so the FEM count of a
                          # seed is the true cost of that run; the landscape is strongly
                          # multimodal, so one seed is a sample, not a characterization.
N_WORKERS = None          # multi-start workers (None = automatic; 1 = serial).
                          # Fork-capable systems can use separate processes. On platforms
                          # without fork, the current implementation falls back to serial
                          # execution. Search results are unchanged by the execution mode.

# --- spline / mesh ----------------------------------------------------------
SMOOTH = 1e-4              # FITPACK smoothing parameter; 0 interpolates all waypoints,
                            # >0 allows deviation from internal waypoints while endpoints stay fixed
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
ELEM   = 0.005            # beam element size [m] used inside the optimisation loop
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

# --- output -----------------------------------------------------------------
SHOW = True               # open the final 3D shape in an interactive window you can rotate/zoom
                          # (set False for headless/batch runs; PNGs are saved either way)
EXPORT_CAD = True         # write the winning centreline as IGES/STEP B-spline curves
CAD_UNITS  = "MM"         # CAD systems expect millimetres; the code works in metres

# --- optimiser (auto-scaled to the grid) ------------------------------------
# The ACO budget (ants, iterations, and the stagnation/early-stop windows) is
# set automatically from the grid size, so adding nodes for a finer geometry
# does not need manual retuning. Put entries in ACO_OVERRIDES to pin or tune any
# parameter. For premature stagnation of the objective, the two relevant knobs:
#   "stagnation_reset" : <int>    iters of no improvement that trigger a restart (smaller = react sooner)
#   "stagnation_smooth": <0..1>   restart intensity: 1.0 = full reset of tau to tau_max, <1 = gentle pull toward tau_max
# Lowering q0_anneal start or raising tau_bounds[0] also fights early convergence.
# Performance and route-shape knobs (see aco.py for the full documentation):
#   "n_eval_workers": "auto"  evaluate the iteration's routes in parallel processes
#                             (int = worker count). Results are IDENTICAL to serial,
#                             only wall-clock drops. Automatically disabled inside the
#                             parallel multi-start pool to avoid nested pools; on
#                             platforms without fork it silently stays serial.
#   "slack_balance": <0..1>   ration the detour budget along the route so ants keep
#                             curvature capability near B (0 = off, classic bound).
ACO_OVERRIDES = {"n_eval_workers": "auto", "bidirectional": 0.5, "slack_balance":0.5}   # e.g. add {"stagnation_reset": 8, "slack_balance": 0.5}
# ============================================================================ #

mat = {"E": E, "G": E / (2.0 * (1.0 + nu)), "rho": rho}
sec = fem.pipe_section(outer_radius=OUTER_RADIUS, thickness=THICKNESS, nu=nu)

# spacing per axis is derived from the box and the node count
assert all(n >= 2 for n in NODES), "each entry of NODES must be >= 2"
assert all(L > 0 for L in BOX_SIZE), "BOX_SIZE extents must be positive"
SPACING = tuple(BOX_SIZE[k] / (NODES[k] - 1) for k in range(3))

MAX_TURN = geo.max_turn_from_pitch(min(SPACING), R_MIN)

# spherical domain: resolve centre/radius defaults
SPHERE_CENTER = (np.array([0.5 * L for L in BOX_SIZE]) if SPHERE_CENTER is None
                 else np.asarray(SPHERE_CENTER, float))
SPHERE_RADIUS = 0.5 * min(BOX_SIZE) if SPHERE_RADIUS is None else float(SPHERE_RADIUS)

# optional central keep-out: the physical box whose interior is forbidden
# (note: any falsy value disables it, unlike the `is not None` test of run_spline.py,
# where EXCLUDE_CENTER = False silently created a degenerate zero-size box)
keepout_box = None
if EXCLUDE_CENTER:
    frac = (EXCLUDE_CENTER,) * 3 if np.isscalar(EXCLUDE_CENTER) else tuple(EXCLUDE_CENTER)
    lo = np.array([0.5 * (1 - frac[k]) * BOX_SIZE[k] for k in range(3)])
    hi = np.array([0.5 * (1 + frac[k]) * BOX_SIZE[k] for k in range(3)])
    keepout_box = (lo, hi)

# node mask: inside the ball AND outside the (open) keep-out box
def mask(ix, iy, iz):
    x = np.array([ix * SPACING[0], iy * SPACING[1], iz * SPACING[2]])
    if np.linalg.norm(x - SPHERE_CENTER) > SPHERE_RADIUS + 1e-12:
        return False
    if keepout_box is not None:
        lo, hi = keepout_box
        if np.all(x > lo) and np.all(x < hi):
            return False
    return True

n_full = NODES[0] * NODES[1] * NODES[2]
grid = G.build_grid(nx=NODES[0], ny=NODES[1], nz=NODES[2], spacing=SPACING, mask=mask)
if len(grid["coords"]) < 2:
    raise ValueError("the ball retains fewer than two lattice nodes; enlarge SPHERE_RADIUS or NODES.")
adj = G.build_neighbours(grid, connectivity=CONNECT)

# endpoints: given index triples, or the antipodal poles along ENDPOINT_AXIS
if A_NODE is not None and B_NODE is not None:
    if A_NODE not in grid["id_of"] or B_NODE not in grid["id_of"]:
        raise ValueError("A_NODE/B_NODE is not a retained node (outside the ball or inside the keep-out).")
    sid = grid["id_of"][A_NODE]; tid = grid["id_of"][B_NODE]
else:
    u_ax = np.asarray(ENDPOINT_AXIS, float)
    u_ax = u_ax / np.linalg.norm(u_ax)
    proj = grid["coords"] @ u_ax
    sid = int(np.argmin(proj)); tid = int(np.argmax(proj))

# optional override: A anchored at a chosen corner of the central keep-out box
if A_KEEPOUT_CORNER is not None:
    if keepout_box is None:
        raise ValueError("A_KEEPOUT_CORNER requires EXCLUDE_CENTER to define the central box.")
    _s = np.asarray(A_KEEPOUT_CORNER, float)
    _corner = np.where(_s > 0, keepout_box[1], keepout_box[0])
    _d = np.linalg.norm(grid["coords"] - _corner, axis=1)
    sid = int(np.argmin(_d))
    if sid == tid:
        raise ValueError("A collapsed onto B; choose another corner or set B_NODE explicitly.")
    print("  A anchored at keep-out corner (%+d,%+d,%+d): node at (%.0f, %.0f, %.0f) mm, "
          "%.1f mm from the exact corner"
          % (int(_s[0]), int(_s[1]), int(_s[2]),
             grid["coords"][sid][0]*1e3, grid["coords"][sid][1]*1e3, grid["coords"][sid][2]*1e3,
             _d[sid]*1e3))

# the masked lattice is not guaranteed to be connected: verify A reaches B
def _connected(adj, s, t):
    seen = {s}; front = [s]
    while front:
        nxt = []
        for i in front:
            for j, _len in adj[i]:          # adjacency entries are (node_id, edge_length)
                if j not in seen:
                    if j == t:
                        return True
                    seen.add(j); nxt.append(j)
        front = nxt
    return s == t
if not _connected(adj, sid, tid):
    raise ValueError("A and B are not connected on the masked lattice; enlarge SPHERE_RADIUS, "
                     "refine NODES or shrink EXCLUDE_CENTER.")

# straight A-B distance and the resulting length budget
straight_AB = float(np.linalg.norm(grid["coords"][tid] - grid["coords"][sid]))
Lmax = LMAX_ABS if LMAX_ABS is not None else LMAX_FACTOR * straight_AB

# design-volume bounds and the containment tolerance for the as-built spline (see geom_fn)
GMIN = grid["coords"].min(axis=0)
GMAX = grid["coords"].max(axis=0)
DOMAIN_TOL = 0.5 * min(SPACING)
if Lmax < straight_AB:
    raise ValueError("Lmax (%.0f mm) is below the straight A-B distance (%.0f mm); "
                     "no route can exist. Increase LMAX_FACTOR (>1) or LMAX_ABS."
                     % (Lmax * 1e3, straight_AB * 1e3))

print("Configuration")
print("  bounding box : %.0f x %.0f x %.0f mm"
      % (BOX_SIZE[0]*1e3, BOX_SIZE[1]*1e3, BOX_SIZE[2]*1e3))
print("  sphere  : centre (%.0f, %.0f, %.0f) mm | radius %.0f mm | %d of %d lattice nodes retained"
      % (SPHERE_CENTER[0]*1e3, SPHERE_CENTER[1]*1e3, SPHERE_CENTER[2]*1e3,
         SPHERE_RADIUS*1e3, len(grid["coords"]), n_full))
print("  nodes   : %d x %d x %d = %d   (spacing %.0f x %.0f x %.0f mm)"
      % (NODES[0], NODES[1], NODES[2], len(grid["coords"]),
         SPACING[0]*1e3, SPACING[1]*1e3, SPACING[2]*1e3))
print("  A -> B  : straight %.0f mm" % (straight_AB*1e3))
print("  Lmax    : %.0f mm  (%.2fx straight)" % (Lmax*1e3, Lmax/straight_AB))
print("  tube OD : %.1f mm | bands %s Hz | R_min %.0f mm"
      % (OD*1e3, ", ".join("%.0f-%.0f" % (b[0], b[1]) for b in BANDS), R_MIN*1e3))
if keepout_box is not None:
    lo, hi = keepout_box
    print("  keep-out: box [%.0f-%.0f]x[%.0f-%.0f]x[%.0f-%.0f] mm  (%d of %d nodes removed; %s)"
          % (lo[0]*1e3, hi[0]*1e3, lo[1]*1e3, hi[1]*1e3, lo[2]*1e3, hi[2]*1e3,
             n_full - len(grid["coords"]), n_full,   # removed by ball AND keep-out together
             "spline containment enforced" if ENFORCE_KEEPOUT else "nodes removed only"))
print()

# per-band weights, set by the user (one per band)
if len(BAND_WEIGHTS) != len(BANDS):
    raise ValueError("BAND_WEIGHTS must have exactly one weight per band in BANDS.")
band_weights = list(BAND_WEIGHTS)

def objective_poly(coords):
    return sum(w * fem.evaluate_path_power(coords, mat, sec, b, freq_mode="adaptive",
                                           max_elem_len=0.035, drive_dir=DRIVE, rayleigh=RAY,
                                           excitation=EXCITATION, prescribed_disp=PRESCRIBED_DISP,
                                           prescribed_mask=PRESCRIBED_MASK)
               for b, w in zip(BANDS, band_weights))

def objective_spline(coords):
    nodes, elems, _ = geo.mesh_spline(coords, ELEM, smooth=SMOOTH)
    return fem.band_integrated_power(nodes, elems, mat, sec, BANDS, weights=band_weights,
                                     freq_mode="adaptive", drive_dir=DRIVE, rayleigh=RAY,
                                     excitation=EXCITATION, prescribed_disp=PRESCRIBED_DISP,
                                     prescribed_mask=PRESCRIBED_MASK,
                                     solver=HARMONIC_SOLVER, modal_fcut=MODAL_FCUT)

def geom_fn(coords):
    # exact as-built geometry of the route: spline arc length and minimum bend radius
    tck = geo.fit_spline(coords, smooth=SMOOTH)
    R, _ = geo.min_radius(tck)
    L = geo.total_length(tck)
    pts = np.column_stack(splev(np.linspace(0, 1, 400), tck))
    ok = True
    if ENFORCE_KEEPOUT and keepout_box is not None:
        lo, hi = keepout_box
        if bool(np.all((pts > lo) & (pts < hi), axis=1).any()):
            ok = False               # spline enters the keep-out box
    # Domain containment of the as-built spline. The fitted curve may depart from the
    # lattice scaffold between waypoints and can therefore leave the design box; the natural scale
    # of that bulge is the lattice pitch (measured on this 6x6x6 case: mean ~9 mm,
    # max ~21 mm). Excursions up to half the smallest pitch are tolerated; beyond
    # that the route genuinely leaves the design volume and is rejected.
    if ok:
        exc = float(np.maximum(np.linalg.norm(pts - SPHERE_CENTER, axis=1)
                               - SPHERE_RADIUS, 0.0).max())
        ok = exc <= DOMAIN_TOL
    # Optional hard constraint: the centreline must keep a clearance to itself. A route
    # that is a simple path on the lattice can still cross itself in space (two face
    # diagonals of one cell intersect at the cell centre using four distinct nodes).
    if ok and SELF_CLEARANCE:
        d_self, *_ = geo.min_self_distance(tck, n=400, arc_gap=np.pi * R_MIN)
        ok = d_self >= SELF_CLEARANCE
    return L, R, ok   # 3rd value: out of the keep-out box and inside the ball (tolerance)

BASE = A.auto_params(grid, overrides=ACO_OVERRIDES, verbose=True)

# ======================================================================== #
# PART 1 - post-processing the polyline winner (demonstrates the WRONG way)
# Only feeds the geometry-sensitivity figure; skipped when RUN_PART1 is False.
# ======================================================================== #
w_poly = None
if RUN_PART1:
    print("PART 1: optimise polyline, then spline the winner")
    r1 = A.optimize(grid, adj, sid, tid, Lmax, objective_poly, params=BASE, seed=SEEDS[0], verbose=False)
    w_poly = r1["best_power"]
    nn, ee, _ = geo.mesh_spline(G.path_coords(grid, r1["best_path"]), ELEM, smooth=SMOOTH)
    w_spl = fem.band_integrated_power(nn, ee, mat, sec, BANDS, weights=band_weights,
                                      freq_mode="adaptive", drive_dir=DRIVE, rayleigh=RAY,
                                      excitation=EXCITATION, prescribed_disp=PRESCRIBED_DISP,
                                      prescribed_mask=PRESCRIBED_MASK)
    print(f"  polyline-optimised power : {w_poly:.4e}")
    print(f"  same route, splined      : {w_spl:.4e}  ({w_spl/w_poly:.0f}x WORSE)")
    print("  -> post-processing destroys the optimum (geometry sensitivity).\n")

# ======================================================================== #
# PART 2 - optimise the smooth geometry directly, WITH the bend-radius limit
# ======================================================================== #
print(f"PART 2: optimise smooth geometry directly, turn<={MAX_TURN:.0f} deg, "
      f"exact spline length<={Lmax*1e3:.0f} mm and R>={R_MIN*1e3:.0f} mm enforced")
params = dict(BASE); params["max_turn_deg"] = MAX_TURN; params["R_min"] = R_MIN
if TANGENT_CLAMP and (EXIT_DIR_A is not None or ARRIVE_DIR_B is not None):
    geo.END_TANGENTS = (EXIT_DIR_A, ARRIVE_DIR_B)
if EXIT_DIR_A is not None:
    params["start_dir"] = EXIT_DIR_A
    params["start_max_turn_deg"] = EXIT_CONE_DEG
if ARRIVE_DIR_B is not None:
    params["end_dir"] = ARRIVE_DIR_B
    params["end_max_turn_deg"] = ARRIVE_CONE_DEG
import time as _time
_t_search = _time.time()
r2 = A.optimize_multistart(grid, adj, sid, tid, Lmax, objective_spline,
                           params=params, seeds=SEEDS, geom_fn=geom_fn,
                           n_workers=N_WORKERS, verbose=True,
                           log_path="sphere_aco_log.txt")
route = r2["best_path"]
if route is None:
    raise RuntimeError("no feasible route was found: every ant failed the constraints. "
                       "Check that EXIT_DIR_A / ARRIVE_DIR_B cones are geometrically "
                       "achievable at the chosen endpoints (a pole of the ball cannot be "
                       "entered from outside it), and that Lmax and the keep-out leave room.")
tck = geo.fit_spline(G.path_coords(grid, route), smooth=SMOOTH)
L_spline = geo.total_length(tck)
R_achieved, R_pt = geo.min_radius(tck)
print(f"  power            : {r2['best_power']:.4e}")
print(f"  spline length    : {L_spline*1e3:.1f} mm  (<= {Lmax*1e3:.0f} mm)")
print(f"  min bend radius  : {R_achieved*1e3:.1f} mm  ({R_achieved/OD:.1f}x OD, >= {R_MIN*1e3:.0f} mm)")
if EXIT_DIR_A is not None:
    _u0 = np.asarray(EXIT_DIR_A, float); _u0 = _u0 / np.linalg.norm(_u0)
    _t0 = np.array(splev(0.0, tck, der=1), float).ravel()
    _t0 = _t0 / (np.linalg.norm(_t0) + 1e-15)
    _ang = np.degrees(np.arccos(np.clip(np.dot(_t0, _u0), -1.0, 1.0)))
    print(f"  exit direction   : spline tangent at A is {_ang:.1f} deg from EXIT_DIR_A "
          f"(first step filtered to a {EXIT_CONE_DEG:.0f} deg cone)")
if ARRIVE_DIR_B is not None:
    _u1 = np.asarray(ARRIVE_DIR_B, float); _u1 = _u1 / np.linalg.norm(_u1)
    _t1 = np.array(splev(1.0, tck, der=1), float).ravel()
    _t1 = _t1 / (np.linalg.norm(_t1) + 1e-15)
    _ang1 = np.degrees(np.arccos(np.clip(np.dot(_t1, _u1), -1.0, 1.0)))
    print(f"  arrival direction: spline tangent at B is {_ang1:.1f} deg from ARRIVE_DIR_B "
          f"(final step filtered to a {ARRIVE_CONE_DEG:.0f} deg cone)")
d_self, _, _, s_i, s_j = geo.min_self_distance(tck, n=800, arc_gap=np.pi * R_MIN)
print(f"  self-clearance   : {d_self*1e3:.1f} mm  ({d_self/OD:.1f}x OD; arc window pi*R_min = {np.pi*R_MIN*1e3:.0f} mm)"
      + ("" if SELF_CLEARANCE is None else f"  [enforced >= {SELF_CLEARANCE*1e3:.1f} mm]"))
print(f"  routes rejected  : {r2['n_reject_length']} on spline length, "
      f"{r2['n_reject_radius']} on bend radius, "
      f"{r2['n_reject_keepout']} on keep-out")
print(f"  search           : {r2['n_fem_calls']} FEM evals, "
      f"{r2['n_stagnation_events']} stagnation restart(s)")
print(f"  search time      : {_time.time() - _t_search:.0f} s wall-clock for the whole search")

ms = r2.get("multistart")
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
          f"J = {J_direct:.4e} (search value {r2['best_power']:.4e}, deviation {abs(J_direct/r2['best_power']-1):.1e})")

if EXPORT_CAD:
    _cad_scale = 1e3 if CAD_UNITS.upper() == "MM" else 1.0
    _igs = CAD.write_iges(tck, "route_sphere_best.igs", scale=_cad_scale, units=CAD_UNITS)
    _stp = CAD.write_step(tck, "route_sphere_best.stp", scale=_cad_scale, units=CAD_UNITS)
    _err = max(CAD.roundtrip_error(tck, _igs, scale=_cad_scale),
               CAD.roundtrip_error(tck, _stp, scale=_cad_scale))
    print(f"  CAD export       : {_igs}, {_stp} "
          f"({CAD.control_points(tck)[1].shape[0]} control points, "
          f"degree {CAD.control_points(tck)[2]}; round-trip error {_err:.1e} m)")

# computational-cost figure for this case (and persist the cost data for later)
import cost_figure
cost_figure.save_cost_data(r2, "fig_sphere_cost_data.npz")
cost_figure.make_cost_figure(r2, "fig_sphere_cost.png", show=SHOW)
P.plot_radius_profile(tck, "fig_sphere_radius_profile.png", r_min=R_MIN, show=SHOW)
print("Saved: fig_cost.png  (cost data in fig_cost_data.npz), fig_radius_profile.png")

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

nodes, elems, _ = geo.resample_by_length(tck, chosen)

# evolution of the input power over the optimisation (best-so-far and iteration mean)
P.plot_convergence(r2, "fig_sphere_spline_convergence.png", show=SHOW)
print("Saved: fig_spline_convergence.png  (band-integrated input power vs iteration)")

# open-helix reference baseline: a naive space-filling route A->B within the same limits.
# helix_baseline() validates containment against a BOX, so the candidate loop is redone
# here with the radial test of the spherical domain (same length/R_min/keep-out checks).
import baseline_routes as BL
base_boxes = [keepout_box] if keepout_box is not None else None
def _sphere_feasible(pts_cand):
    tckb = geo.fit_spline(np.asarray(pts_cand), smooth=SMOOTH)
    if geo.total_length(tckb) > Lmax:
        return False
    Rb, _ = geo.min_radius(tckb)
    if Rb < R_MIN:
        return False
    smp = geo.eval_spline(tckb, 400)
    exc = float(np.maximum(np.linalg.norm(smp - SPHERE_CENTER, axis=1) - SPHERE_RADIUS, 0.0).max())
    if exc > DOMAIN_TOL:
        return False
    if keepout_box is not None and ENFORCE_KEEPOUT:
        lo, hi = keepout_box
        if bool(np.any(np.all((smp > lo) & (smp < hi), axis=1))):
            return False
    if EXIT_DIR_A is not None:
        t0 = np.array(splev(0.0, tckb, der=1), float).ravel()
        t0 = t0 / (np.linalg.norm(t0) + 1e-15)
        u0 = np.asarray(EXIT_DIR_A, float); u0 = u0 / np.linalg.norm(u0)
        if np.degrees(np.arccos(np.clip(np.dot(t0, u0), -1.0, 1.0))) > EXIT_CONE_DEG:
            return False
    if ARRIVE_DIR_B is not None:
        t1 = np.array(splev(1.0, tckb, der=1), float).ravel()
        t1 = t1 / (np.linalg.norm(t1) + 1e-15)
        u1 = np.asarray(ARRIVE_DIR_B, float); u1 = u1 / np.linalg.norm(u1)
        if np.degrees(np.arccos(np.clip(np.dot(t1, u1), -1.0, 1.0))) > ARRIVE_CONE_DEG:
            return False
    return True
base_pts, base_info = None, None
for _turns in (2, 1, 3):
    for _rad in np.linspace(0.55, 0.10, 10) * SPHERE_RADIUS:
        _cand = BL.helix_points(grid["coords"][sid], grid["coords"][tid],
                                turns=_turns, radius=float(_rad), n=400)
        if _sphere_feasible(_cand):
            base_pts = np.asarray(_cand)
            base_info = {"turns": _turns, "radius": float(_rad)}
            print("  helix baseline: turns=%d, winding radius %.0f mm (feasible in the ball)"
                  % (_turns, _rad * 1e3))
            break
    if base_pts is not None:
        break
base_label = "helix baseline"
if base_pts is None:
    # fallback reference: the shortest admissible lattice route (Dijkstra by edge
    # length), splined and checked by the same geom_fn. With a central obstacle or a
    # corner-anchored A the spindle helix rarely fits, but the "naive direct route"
    # always gives a meaningful comparison against the optimized trajectory.
    print("  helix baseline: no feasible helix inside the ball; "
          "falling back to the shortest admissible route")
    import heapq
    def _dijkstra(weight_fn):
        dist = {sid: 0.0}; prev = {}
        pq = [(0.0, sid)]
        while pq:
            dd, i = heapq.heappop(pq)
            if i == tid:
                break
            if dd > dist.get(i, np.inf):
                continue
            for j, seg in adj[i]:
                nd = dd + weight_fn(i, j, seg)
                if nd < dist.get(j, np.inf) - 1e-15:
                    dist[j] = nd; prev[j] = i
                    heapq.heappush(pq, (nd, j))
        if tid not in prev and tid != sid:
            return None
        p = [tid]
        while p[-1] != sid:
            p.append(prev[p[-1]])
        return p[::-1]
    def _near_box(i, j):
        if keepout_box is None:
            return False
        m = 0.5 * min(SPACING)
        mid = 0.5 * (grid["coords"][i] + grid["coords"][j])
        lo, hi = keepout_box
        return bool(np.all(mid > lo - m) and np.all(mid < hi + m))
    _cands = [_dijkstra(lambda i, j, seg: seg)]
    _cands.append(_dijkstra(lambda i, j, seg: seg * (4.0 if _near_box(i, j) else 1.0)))
    for _p in _cands:
        if _p is None:
            continue
        _c = G.path_coords(grid, _p)
        _L, _R, _ok = geom_fn(_c)
        if _ok:
            base_pts = np.asarray(_c)
            base_info = {"route": "shortest admissible", "nodes": len(_p),
                         "length": float(_L)}
            base_label = "shortest-route baseline"
            print("  shortest-route baseline: %d nodes, spline length %.0f mm, R_min %.1f mm"
                  % (len(_p), _L * 1e3, _R * 1e3))
            break
    if base_pts is None:
        print("  reference baseline: no feasible reference route (skipped)")
if base_pts is not None:
    bn, be, _ = geo.resample_by_length(geo.fit_spline(base_pts, smooth=SMOOTH, end_tangents=None), ELEM)
    W_base = fem.band_integrated_power(bn, be, mat, sec, BANDS, weights=band_weights,
                                       freq_mode="adaptive", drive_dir=DRIVE, rayleigh=RAY,
                                       excitation=EXCITATION, prescribed_disp=PRESCRIBED_DISP,
                                       prescribed_mask=PRESCRIBED_MASK)
    print("  %s power : %.4e  (optimized %.4e, i.e. %.1fx lower than the baseline)"
          % (base_label, W_base, r2["best_power"], W_base / r2["best_power"]))
else:
    print("  reference baseline: not available for this configuration (comparison skipped)")

P.plot_spline(grid, route, tck, nodes, "fig_sphere_spline_manufacturable.png", sid, tid, R_achieved, R_pt,
              show=SHOW, keepout=keepout_box, baseline_pts=base_pts, baseline_label=base_label)

P.plot_spline(grid, route, tck, nodes, "fig_sphere_spline_manufacturable_view2.png", sid, tid, R_achieved, R_pt,
              show=SHOW, keepout=keepout_box, baseline_pts=base_pts, baseline_label=base_label, legend=False)

P.plot_all_splines(grid, r2["evaluated_paths"], route, "fig_sphere_all_splines.png", sid, tid,
                   smooth=SMOOTH, boxes=([keepout_box] if keepout_box is not None else None),
                   baseline_pts=base_pts, baseline_label=base_label, show=SHOW)
print("Saved: fig_all_splines.png  (every tested route faint, best solid)")

# one spline-cloud figure per seed (routes explored by that seed, its own best solid)
for _s, _ps in r2.get("per_seed", {}).items():
    if len(r2["per_seed"]) < 2 or _ps["best_path"] is None:
        continue
    P.plot_all_splines(grid, _ps["evaluated_paths"], _ps["best_path"], "fig_sphere_all_splines_seed%d.png" % _s, sid, tid,
                       smooth=SMOOTH, boxes=([keepout_box] if keepout_box is not None else None), baseline_pts=base_pts, baseline_label=base_label, show=SHOW)
if len(r2.get("per_seed", {})) > 1:
    print("Saved: fig_sphere_all_splines_seed<s>.png  (one spline cloud per seed; best of that seed solid)")

# domain-only presentation figures: the bare design domain (with any keep-out), and the
# same domain with only the deterministic baseline, without the cloud of tested splines
P.plot_domain(grid, "fig_sphere_domain.png", sid, tid, boxes=base_boxes, show=SHOW)
print("Saved: fig_domain.png  (design domain, lattice and obstacles only)")
if base_pts is not None:
    P.plot_domain(grid, "fig_sphere_domain_baseline.png", sid, tid, boxes=base_boxes,
                  baseline_pts=base_pts, baseline_label=base_label,
                  smooth=SMOOTH, show=SHOW)
    print("Saved: fig_sphere_domain_baseline.png  (domain with the reference baseline only)")
def spectrum(coords, freqs, end_tangents="global"):
    nn, ee, _ = geo.resample_by_length(geo.fit_spline(coords, smooth=SMOOTH, end_tangents=end_tangents), chosen)
    if EXCITATION == "prescribed":
        return fem.input_power_spectrum_prescribed(nn, ee, mat, sec, 0, len(nn)-1, freqs,
                                                   prescribed_disp=PRESCRIBED_DISP, rayleigh=RAY,
                                                   prescribed_mask=PRESCRIBED_MASK)
    return fem.input_power_spectrum(nn, ee, mat, sec, 0, DRIVE, len(nn)-1, freqs, rayleigh=RAY)
P.plot_frf(grid, {"best_path": route}, "fig_sphere_spline_manufacturable_frf.png",
           mat, sec, BANDS, spectrum, freq_range=(0.0, 10000.0),
           baseline_coords=base_pts, baseline_label=base_label, show=SHOW,
           baseline_spectrum=lambda c, f: spectrum(c, f, end_tangents=None))

# reaction force/torque spectrum at the clamped end B: with B clamped, its reactions
# are the blocked force and torque transmitted downstream. Plotting them next to the
# input power shows whether minimising the input power also lowers what reaches B.
if EXCITATION == "prescribed":
    import reaction_figure as RF
    nodes_r, elems_r, _ = geo.resample_by_length(tck, chosen)
    base_nodes = base_elems = None
    if base_pts is not None:
        base_nodes, base_elems, _ = geo.resample_by_length(geo.fit_spline(base_pts, smooth=SMOOTH, end_tangents=None), chosen)
    RF.compute_and_plot(nodes_r, elems_r, mat, sec, BANDS,
                        forces_path="fig_sphere_reaction_forces.png",
                        torques_path="fig_sphere_reaction_torques.png",
                        freq_range=(0.0, 10000.0), rayleigh=RAY, show=SHOW,
                        prescribed_disp=PRESCRIBED_DISP, prescribed_mask=PRESCRIBED_MASK,
                        base_nodes=base_nodes, base_elements=base_elems,
                        base_label=base_label)
    print("Saved: fig_reaction_forces.pdf, fig_reaction_torques.pdf")
print(f"\n  final element size: {chosen*1e3:.0f} mm | {len(elems)} elements")
_poly = f"polyline opt was {w_poly:.3e}" if w_poly is not None else "PART 1 skipped"
print(f"  summary: power {r2['best_power']:.3e} ({_poly}), "
      f"spline length {L_spline*1e3:.0f} mm, R_min {R_achieved*1e3:.0f} mm, buildable.")
print("Saved: fig_spline_manufacturable.pdf, fig_spline_manufacturable_frf.pdf")

if SHOW:
    print("Opening the final shape in an interactive window (close it to finish)...")
    P.show_interactive()