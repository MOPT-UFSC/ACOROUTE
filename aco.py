"""
aco.py
Ant colony optimization for vibration-oriented pipe-route design.

A single bounded pheromone field tau_ij lives on the undirected edges of the lattice
(MAX-MIN Ant System mechanics: iteration-best deposit, best-so-far elitism, global
evaporation, [tau_min, tau_max] bounds and a stagnation restart that pulls every
trail back toward tau_max). At each construction step, with probability q0(t) the
next node is sampled with probability proportional to tau_ij**alpha over the
admissible neighbours; otherwise it is sampled uniformly among them. The objective
enters the search only through the ranking of complete routes: the deposit
magnitude is constant, so the dynamics are invariant to a rescaling of J.

The module also retains alternative deposit modes (``deposit_mode='all'`` and
``'best_so_far'``) for numerical experiments; the reported configuration is
``'iteration_best'``.

Construction-time length pruning and the balanced detour budget
----------------------------------------------------------------
During construction an ant at position x with consumed length l only steps to a
neighbour when the route can still be completed within the length bound. The
classic test uses the straight-line lower bound d(x, B) on the remaining distance
(an admissible A*-style heuristic):

    l + d(x, B) <= Lmax.                                   (gamma = 0)

This test treats the detour budget as a first-come-first-served resource: turns
near A consume slack that straight steps never pay back, so close to B only
near-beeline edges remain admissible and the constructed routes lose the ability
to curve as they approach the target. With ``slack_balance`` = gamma in (0, 1]
the bound switches to the geodesic remaining distance and is inflated by the
route's own discretionary detour ratio,

    l + dG(x, B) * (1 + gamma * S / D0) <= Lmax,
    dG = shortest-path distance on the masked lattice,
    D0 = dG(A, B),   S = Lmax - D0  (the discretionary detour budget),

which reserves a share of the budget proportional to the remaining geodesic
distance (equivalently: the remaining slack Lmax - l - dG(x, B) must be at least
gamma * S * dG(x, B) / D0). At gamma = 1 the test takes the normalized form

    l / Lmax + dG(x, B) / D0 <= 1,

a uniform rationing of the detour budget along the route. Using dG rather than
the straight-line distance matters in obstacle domains (baffles, keep-outs):
detours *forced* by the geometry are part of D0, not of S, so they are never
charged against the discretionary slack. This yields a guarantee: along any
graph-shortest route l = D0 - dG(x, B) at every node, so the test reduces to
l + dG <= D0 + gamma * S * dG / D0 <= Lmax, and every graph-shortest route
satisfies the length rule for all gamma <= 1 in any domain. The guarantee
concerns the length rule alone: a geodesic route may still be excluded by the
turn-angle pre-filter, and the stochastic constructor is not guaranteed to
sample it within a finite operation budget. (With the straight-line bound in the
reserve, the guarantee would hold only where the straight route is feasible; in
a serpentine domain high gamma would then over-prune.) The mechanism is
analogous to weighted A*: gamma = 0 keeps an exact lower bound (complete
construction, biased turn distribution), gamma > 0 trades constructive
completeness for a balanced turn distribution. Note the deliberate asymmetry at
gamma = 0: the formula above with gamma = 0 would use the geodesic bound, which
is slightly tighter than the straight-line one (dG >= d always, since the
lattice metric dominates the Euclidean distance); for exact backward
compatibility the implementation treats gamma = 0 as a special case and keeps
the straight-line test, so the admissible set has a small discontinuity between
gamma = 0 and gamma -> 0+. The reserve vanishes at the target (dG(B, B) = 0),
so every constructed route still satisfies the true bound l <= Lmax and the
exact as-built feasibility check on the spline is unchanged; only the set of
routes the construction can produce shrinks monotonically with gamma, excluding
routes whose discretionary detour is strongly front-loaded. The geodesic field
dG(., B) is one Dijkstra sweep per optimize() call (grid.distances_from),
shared by all ants; it also prunes, at no extra cost, any pocket of the FULL
lattice from which B is unreachable within the remaining budget. Dead ends
created dynamically by the tabu list are not captured by this static field and
are still resolved by backtracking.
"""

import numpy as np
from grid import edge_key, path_coords, path_length, distances_from


# --------------------------------------------------------------------------- #
# Ant path construction
# --------------------------------------------------------------------------- #
def build_one_ant(grid, adj, tau, start_id, target_id, Lmax, params, rng,
                  q0=None, dist_target=None):
    """Construct a single ant path from start to target, with backtracking.

    Backtracking (randomised DFS biased by pheromone) lets an ant recover from a
    dead end instead of failing outright, which greatly increases the number of
    successful ants per iteration and therefore the quality of the pheromone update.
    Returns (path_ids, length) on success, or (None, None) if no path exists within
    the operation budget.
    """
    alpha = params["alpha"]
    if q0 is None:
        q0 = params["q0"]
    tau0 = params["tau0"]
    max_ops = params.get("max_ops", 50 * len(grid["coords"]))
    # optional manufacturability: limit how sharply the route may turn at a node.
    max_turn = params.get("max_turn_deg", None)
    cos_max = np.cos(np.radians(max_turn)) if max_turn is not None else None
    # optional prescribed exit direction at the start node: the first step is filtered
    # by the same turn-angle test, measured against this virtual arrival direction,
    # with its own cone angle (defaults to max_turn_deg, else 90 deg)
    start_dir = params.get("start_dir", None)
    cos_start = None
    if start_dir is not None:
        start_dir = np.asarray(start_dir, float)
        start_dir = start_dir / (np.linalg.norm(start_dir) + 1e-15)
        _st = params.get("start_max_turn_deg", None)
        if _st is None:
            _st = max_turn if max_turn is not None else 90.0
        cos_start = np.cos(np.radians(_st))
    # optional prescribed arrival direction at the target node: the final step into the
    # target (the travel direction with which the route enters it) must lie inside the
    # corresponding cone; a non-conforming edge into the target is simply inadmissible
    end_dir = params.get("end_dir", None)
    cos_end = None
    if end_dir is not None:
        end_dir = np.asarray(end_dir, float)
        end_dir = end_dir / (np.linalg.norm(end_dir) + 1e-15)
        _et = params.get("end_max_turn_deg", None)
        if _et is None:
            _et = max_turn if max_turn is not None else 90.0
        cos_end = np.cos(np.radians(_et))

    coords = grid["coords"]
    # Balanced detour budget (see the module docstring). With gamma > 0 the pruning
    # bound on the remaining length is the geodesic (shortest-path) distance to the
    # target on the masked lattice, so detours forced by obstacles are never charged
    # against the discretionary budget S = Lmax - D0; any graph-shortest route
    # satisfies the length rule for every gamma <= 1 (the turn-angle filter may still
    # exclude it). `dist_target` (a dict from grid.distances_from(adj, target_id))
    # can be precomputed by the caller and shared across ants; optimize() does so.
    # gamma = 0 is a deliberate special case: it keeps the classic straight-line
    # lower bound bit-identically and needs no geodesic distances (the geodesic bound
    # is slightly tighter, so the admissible set is discontinuous at gamma -> 0+).
    gamma = params.get("slack_balance", 0.0)
    use_graph = gamma > 0.0
    reserve = 0.0
    if use_graph:
        if dist_target is None:
            dist_target = distances_from(adj, target_id)
        D0 = dist_target.get(start_id, np.inf)
        if np.isfinite(D0) and D0 > 0.0:
            reserve = gamma * max(0.0, Lmax - D0) / D0
        else:
            use_graph = False       # unreachable/degenerate: fall back to gamma = 0

    path = [start_id]
    visited = {start_id}
    length = 0.0
    tried = [set()]          # neighbours already attempted from each stack frame

    ops = 0
    while ops < max_ops:
        ops += 1
        current = path[-1]
        if current == target_id:
            return path, length

        # direction of the last segment (for the turn-angle constraint)
        prev_dir = None
        cos_lim = cos_max
        if cos_max is not None and len(path) >= 2:
            prev_dir = coords[current] - coords[path[-2]]
            prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-15)
        elif start_dir is not None and len(path) == 1:
            prev_dir = start_dir          # virtual arrival direction at the start node
            cos_lim = cos_start

        feasible = []
        for (j, seg) in adj[current]:
            if j in visited or j in tried[-1]:
                continue
            new_len = length + seg
            rem = (dist_target.get(j, np.inf) if use_graph
                   else np.linalg.norm(coords[j] - coords[target_id]))
            if new_len + rem * (1.0 + reserve) > Lmax + 1e-9:
                continue
            if prev_dir is not None or (end_dir is not None and j == target_id):
                new_dir = coords[j] - coords[current]
                new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-15)
                if prev_dir is not None and float(np.dot(prev_dir, new_dir)) < cos_lim - 1e-9:
                    continue        # turn too sharp -> would violate min bend radius
                if (end_dir is not None and j == target_id
                        and float(np.dot(new_dir, end_dir)) < cos_end - 1e-9 ):
                    continue        # arrival direction at the target outside the cone
            feasible.append((j, seg))

        if not feasible:
            # dead end: backtrack one step
            if len(path) == 1:
                return None, None
            dead = path.pop()
            visited.discard(dead)
            tried.pop()
            # account for the length of the removed edge
            length -= np.linalg.norm(coords[dead] - coords[path[-1]])
            tried[-1].add(dead)
            continue

        if rng.random() <= q0:
            t = np.array([tau.get(edge_key(current, j), tau0) for (j, _) in feasible])
            w = t ** alpha
            w = w / w.sum()
            choice = rng.choice(len(feasible), p=w)
        else:
            choice = rng.integers(len(feasible))

        j, seg = feasible[choice]
        path.append(j)
        visited.add(j)
        tried.append(set())
        length += seg

    return None, None


# --------------------------------------------------------------------------- #
# Pheromone deposit
# --------------------------------------------------------------------------- #
def _reward(cost, cmin, cmax):
    """Map a cost to a deposit reward in [0,1]; best (lowest) cost -> 1."""
    if cmax - cmin < 1e-30:
        return np.ones_like(cost)
    return (cmax - cost) / (cmax - cmin)


def deposit(tau, ants, rho, params, best=None):
    """Evaporate, then deposit according to params['deposit_mode'].

    "all"            : every successful ant deposits, reward-weighted in [0,1].
    "iteration_best" : only the lowest-power ant of this iteration deposits.
    "best_so_far"    : only the global-best ant deposits (strongest intensification).
    `ants` is a list of dicts {path, length, power}. Pheromone is clamped to the
    Max-Min bounds if provided (params['tau_bounds']).
    """
    Q_tau = params["Q_tau"]
    mode = params.get("deposit_mode", "all")

    # Evaporation. The field is pre-populated on every admissible edge by optimize(),
    # so this loop is a genuinely global evaporation, as the Max-Min ant system prescribes.
    # Otherwise an edge that had never been chosen would keep its initial value forever
    # while a tried-and-abandoned edge decayed toward tau_min, and never-visited edges
    # would end up far MORE attractive than abandoned ones.
    for e in list(tau.keys()):
        tau[e] *= (1.0 - rho)

    if mode == "best_so_far":
        depositors = [best] if best is not None else []
        rP = np.array([1.0] * len(depositors))
    elif mode == "iteration_best":
        if ants:
            ib = ants[int(np.argmin([a["power"] for a in ants]))]
            depositors = [ib]; rP = np.array([1.0])
        else:
            depositors = []; rP = np.array([])
    else:  # "all"
        depositors = ants
        if ants:
            P = np.array([a["power"] for a in ants])
            rP = _reward(P, P.min(), P.max())
        else:
            rP = np.array([])

    for a, rp in zip(depositors, rP):
        path = a["path"]
        for k in range(len(path) - 1):
            e = edge_key(path[k], path[k + 1])
            tau[e] = tau.get(e, params["tau0"]) + Q_tau * rp

    _clamp(tau, params.get("tau_bounds"))


def deposit_elitist(tau, best, params):
    """Extra reinforcement on the best-so-far path (drives convergence)."""
    if best is None:
        return
    w = params.get("elitist_weight", 0.0)
    if w <= 0.0:
        return
    path = best["path"]
    for k in range(len(path) - 1):
        e = edge_key(path[k], path[k + 1])
        tau[e] = tau.get(e, params["tau0"]) + w * params["Q_tau"]


# --------------------------------------------------------------------------- #
# Parallel objective evaluation (the routes of one iteration are independent)
# --------------------------------------------------------------------------- #
_EVAL_CTX = {}


def _limit_blas_threads(n=1):
    """Pin the BLAS/LAPACK thread count inside a worker process.

    Each FEM evaluation is small (hundreds of DOF), so one process per route with
    single-threaded BLAS is the efficient configuration; without this, every worker
    spawns a full OpenBLAS thread pool and the spinning threads oversubscribe the
    machine (measured 13x SLOWER than serial). Environment variables do not work
    here because fork-children inherit an already-initialised BLAS, so the loaded
    library is located via /proc/self/maps and its runtime setter is called
    directly (the 64-bit SciPy/NumPy builds suffix the symbol with '64_').
    """
    import ctypes, re
    try:
        with open("/proc/self/maps") as f:
            txt = f.read()
    except OSError:
        return
    libs = sorted(set(re.findall(r"/\S*openblas\S*\.so\S*", txt)))
    for path in libs:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for sym in ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads",
                    "openblas_set_num_threads64_", "openblas_set_num_threads"):
            fn = getattr(lib, sym, None)
            if fn is not None:
                try:
                    fn(int(n))
                    break
                except Exception:
                    pass


def _eval_pool_init():
    _limit_blas_threads(1)


def _eval_worker(coords):
    """Worker for the per-iteration parallel objective evaluation (fork-inherited)."""
    return _EVAL_CTX["objective_fn"](coords)


def _make_eval_pool(objective_fn, n_workers):
    """Fork-based process pool evaluating objective_fn, or None if unavailable.

    The objective (often a closure over the FEM problem) is inherited by the
    workers through fork, exactly like the multi-start pool; on platforms without
    fork the evaluation silently stays serial.
    """
    if not n_workers or n_workers == 1:
        return None
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None
    global _EVAL_CTX
    _EVAL_CTX = {"objective_fn": objective_fn}
    return ctx.Pool(processes=(None if n_workers in (True, "auto") else int(n_workers)),
                    initializer=_eval_pool_init)


# --------------------------------------------------------------------------- #
# Main optimisation loop
# --------------------------------------------------------------------------- #
def default_params():
    # Generic fallback values. The reported cases call auto_params(), which overrides
    # the search budget and the principal ACO dynamics parameters.
    return {
        "alpha": 1.0,            # exponent of the pheromone field tau in the transition rule
        "q0": 0.7,               # exploitation probability (slide value)
        "q0_iters": None,        # iterations over which q0_anneal runs (None: max_iter). Fixing it
                                 # decouples the exploration schedule from the iteration budget, so
                                 # a longer max_iter does not stretch the exploratory phase.
        "improve_tol": 1e-2,     # relative gain on the incumbent that counts as progress
                                 # for the stagnation restart and the patience stop                         
        "max_turn_deg": None,    # limit turn angle per node (cheap construction pre-filter); None = off
        "start_dir": None,       # prescribed exit direction at the start node (unit vector); None = off
        "start_max_turn_deg": None,  # cone half-angle [deg] for the first step; None = max_turn_deg (else 90)
        "end_dir": None,         # prescribed arrival (travel) direction at the target node; None = off
        "end_max_turn_deg": None,    # cone half-angle [deg] for the final step; None = max_turn_deg (else 90)
        "R_min": 0.0,            # hard min bend radius on the as-built spline (0 = off)
        "slack_balance": 0.0,    # gamma in [0,1]: ration the detour budget Lmax-d(A,B)
                                 # proportionally along the route, so ants keep the same
                                 # turning capability near B as near A. 0 = classic
                                 # first-come-first-served bound (bit-identical runs).
        "rho": 0.1,              # evaporation rate
        "tau0": 1.0,             # initial pheromone level
        "Q_tau": 1.0,            # deposit scale (constant: J only ranks the routes)
        "elitist_weight": 2.0,   # best-so-far reinforcement (0 disables)
        "deposit_mode": "all",   # "all" | "iteration_best" | "best_so_far"
        "tau_bounds": None,      # (tau_min, tau_max) Max-Min Ant System clamping
        "stagnation_reset": 0,   # re-init pheromone after N stale iters (0 off)
        "kick_concentration": 3.0,
        "stagnation_smooth": 1.0,# kick intensity on stagnation: 1.0 = full reset to tau_max, <1 = gentle smoothing
        "global_evaporation": True,  # evaporate on every admissible edge, not only on the
                                     # edges that happen to carry a deposit already
        "q0_rewind_on_kick": False,  # restart the q0 anneal after a stagnation kick, so the
                                     # re-diversified pheromone is explored at low q0 instead of
                                     # exploited greedily at the annealed (high) q0. Off by
                                     # default to leave existing runs bit-identical.
        "bidirectional": 0.0,    # fraction of each iteration's ants that construct their
                                 # route from B toward A (0 = all from A, 0.5 = half). A
                                 # B-started ant spends its detour slack near B, so the
                                 # combined population explores curvature at BOTH ends
                                 # without restricting the route space. Completed routes
                                 # are canonicalized to the A->B orientation before the
                                 # geometry checks, the objective and the cache, so the
                                 # excitation/support roles and J are unchanged; pheromone
                                 # lives on undirected edges, so both populations reinforce
                                 # one coherent field. Prescribed exit/arrival cones are
                                 # mapped automatically for the reversed construction.
        "n_eval_workers": None,  # parallel FEM evaluations per iteration: an int, or
                                 # "auto" for one worker per core. The routes of one
                                 # iteration are independent cases, so the result is
                                 # identical to the serial run; only wall-clock drops.
                                 # None/1 = serial. Prefer this with a single seed;
                                 # with several multi-start seeds prefer the seed-level
                                 # pool (n_workers) and leave this off to avoid
                                 # oversubscribing the machine.
        "n_ants": 20,            # ants per iteration (slide value)
        "max_iter": 200,         # max iterations (slide value)
        "patience": 40,          # early stop if best unchanged this many iters
        "store_paths_every": 5,  # snapshot all ant paths every N iters (for plots)
        "print_every": 10,       # verbose progress line every N iterations
    }


def auto_params(grid, overrides=None, verbose=False):
    """Return ACO parameters scaled to the size of the lattice.

    The search budget is tied to the exploration schedule. The q0 ramp lasts
        T_q = clamp(5 h, 40, 120)   iterations,   h = mean(nx, ny, nz) - 1,
    where h is the number of hops in a direct route and therefore the number of
    decisions an ant makes per route. The stagnation window, the patience and
    the iteration cap are derived from T_q so that the ramp always ends before
    the run can stop, every restart has half a ramp to re-converge before the
    patience can trigger, and the cap is normally not reached. The dynamics
    parameters (alpha, rho, q0 range and Max-Min bounds) are size
    independent. The coefficient 5 reproduces T_q = 60 on the 13x13x13 lattice,
    where the population is observed to concentrate around iteration 60. Pass
    `overrides` (a dict) to pin any parameter.
    """
    h = (grid["nx"] + grid["ny"] + grid["nz"]) / 3.0 - 1.0
    clamp = lambda v, lo, hi: int(max(lo, min(hi, round(v))))
    q0_iters = clamp(5.0 * h, 40, 120)
    params = {
        "n_ants":           clamp(2.8 * h, 12, 60),
        "q0_iters":         q0_iters,
        "patience":         q0_iters,            # the ramp ends before the run can stop
        "stagnation_reset": q0_iters // 2,       # patience = 2 x stall window
        "max_iter":         4 * q0_iters,        # safety cap, normally not reached
        "stagnation_smooth": 1.0,
        "alpha": 1.0, "rho": 0.15,
        "q0_anneal": (0.2, 0.9),
        "deposit_mode": "iteration_best",
        "tau_bounds": (0.05, 8.0),
        "elitist_weight": 1.0,
        "store_paths_every": 100,
    }
    if overrides:
        params.update(overrides)
    if verbose:
        print("ACO auto-params (grid h=%.1f): n_ants=%d, q0_iters=%d, patience=%d, "
              "stagnation_reset=%d, max_iter=%d"
              % (h, params["n_ants"], params["q0_iters"], params["patience"],
                 params["stagnation_reset"], params["max_iter"]))
    return params


def _q0_at(p, it):
    """Annealed exploitation probability at iteration `it`.

    q0 rises linearly from q0_anneal[0] to q0_anneal[1] over `q0_iters` iterations
    (max_iter when q0_iters is None) and stays at q0_anneal[1] afterwards, so the
    schedule is unchanged when the run is allowed to continue past q0_iters."""
    if not p.get("q0_anneal"):
        return p["q0"]
    q0, q1 = p["q0_anneal"]
    n = p.get("q0_iters") or p["max_iter"]
    frac = min(1.0, it / max(1, n - 1))
    return q0 + (q1 - q0) * frac


def _clamp(field, bounds):
    if not bounds:
        return
    lo, hi = bounds
    for e in field:
        if field[e] < lo:
            field[e] = lo
        elif field[e] > hi:
            field[e] = hi


def optimize(grid, adj, start_id, target_id, Lmax, objective_fn,
             params=None, seed=0, verbose=True, geom_fn=None, seed_paths=None,
             cache=None, log_path=None):
    """Run ACO. `objective_fn(coords_Nx3) -> positive scalar (input power)`.

    Optional `geom_fn(coords_Nx3) -> (spline_length, min_radius[, ok])` enforces the
    as-built geometry: a completed route is rejected when its spline arc length
    exceeds Lmax or its minimum bend radius is below params["R_min"]. The returned
    spline length is stored with the candidate and is used for reporting and by the
    optional length-based reward in `deposit_mode="all"`; it does not define a
    separate length objective in the reported `iteration_best` configuration. An
    optional third return value, a boolean, rejects the route on an extra geometric
    constraint such as a keep-out region. The turn-angle limit and the
    polyline length bound stay as cheap construction-time pre-filters. The geometry
    is checked before the objective, so rejected routes cost no FEM solve.

    Returns a results dict with the best path, history, and path snapshots.
    Identical paths are cached so the expensive FEM call runs at most once per path.
    """
    import time as _time
    _t_start = _time.time()
    p = default_params()
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)

    tau = {}
    # Materialize the field on EVERY admissible edge of the lattice, so that the
    # evaporation in deposit() is global (Max-Min ant system semantics, and the dynamics
    # the paper describes). The cost is negligible: a 6x6x6 lattice with 26-connectivity
    # carries a few thousand edges. Opt out with params["global_evaporation"]=False to
    # reproduce the earlier sparse behaviour, in which unvisited edges never evaporated.
    if p.get("global_evaporation", True):
        for _i, _nb in adj.items():
            for _j, _L in _nb:
                tau[edge_key(_i, _j)] = p["tau0"]
    best = None
    best_hist, mean_hist, feasible_hist = [], [], []
    iter_best_hist = []   # lowest objective among the feasible ants of each iteration
    iter_log = []         # full per-iteration diagnostic breakdown (also written to log_path)
    reset_iters = []      # iterations at which a stagnation kick re-diversified the colony
    initial_best_from_seed = False   # True when the warm-start seed sets the initial best
    fem_hist = []   # cumulative full-pipe FEM evaluations after each iteration
    snapshots = []
    # route -> objective memo. An external dict may be supplied to share it between
    # runs; the objective is deterministic and the cache only short-circuits the FEM
    # call, so sharing never changes a run's trajectory, only its n_fem_calls. The
    # multi-start protocol gives every seed its own cache so that n_fem_calls is the
    # true cost of that run.
    cache = {} if cache is None else cache
    no_improve = 0
    since_kick = 0
    n_fem = 0
    n_reject_len = 0
    n_reject_rad = 0
    n_reject_keepout = 0
    n_stagnation = 0

    # optional warm start: bias the pheromone toward one or more given lattice routes so
    # ants reach the target and prefer short, direct routes from the first iteration. This
    # bootstraps the search on a fine lattice, where a purely random construction rarely
    # threads a tight domain and the few routes that do reach the target are too long. A
    # seed that is itself geometrically feasible is taken as the initial best.
    if seed_paths:
        tb = p.get("tau_bounds")
        seed_val = min(tb[1], 6.0 * p["tau0"]) if tb else 6.0 * p["tau0"]
        for spath in seed_paths:
            if not spath or len(spath) < 2:
                continue
            for k in range(len(spath) - 1):
                e = edge_key(spath[k], spath[k + 1])
                tau[e] = max(tau.get(e, p["tau0"]), seed_val)
            # Every feasible corridor is evaluated and the best of them becomes the initial
            # incumbent. A previous version stopped at the FIRST feasible corridor (the
            # guard was `best is None`), so a later, better corridor was never evaluated
            # and the run could start from, and report, a worse design than one of its own
            # seeds. The cost is at most one objective evaluation per corridor.
            if geom_fn is not None:
                coords = path_coords(grid, spath)
                g = geom_fn(coords)
                slen, mrad = g[0], g[1]
                kok = g[2] if len(g) > 2 else True
                if slen <= Lmax + 1e-9 and (p["R_min"] <= 0.0 or mrad >= p["R_min"]) and kok:
                    key = tuple(spath)
                    power = cache.get(key)
                    if power is None:
                        power = objective_fn(coords); cache[key] = power; n_fem += 1
                    if best is None or power < best["power"]:
                        best = {"path": list(spath), "length": slen, "power": power}
                        initial_best_from_seed = True
        _clamp(tau, p.get("tau_bounds"))

    # optional per-iteration diagnostic log written to a text file
    logf = None
    if log_path is not None:
        logf = open(log_path, "w")
        logf.write("# ACO iteration log\n")
        logf.write("# seed=%d  n_ants=%d  max_iter=%d  R_min=%.1f mm  Lmax=%.0f mm  q0_anneal=%s\n"
                   % (seed, p["n_ants"], p["max_iter"], p["R_min"] * 1e3, Lmax * 1e3,
                      str(p.get("q0_anneal"))))
        if initial_best_from_seed:
            logf.write("# initial best set by a warm-start corridor: J=%.6e\n" % best["power"])
        logf.write("# per iteration: of n_ants ants, some do not reach B, some reach it but are\n")
        logf.write("#   rejected by the geometry filter (length, radius, keep-out/containment),\n")
        logf.write("#   and the rest are feasible and FEM-evaluated (fem_new new solves, the others\n")
        logf.write("#   reused from the route cache). reached = n_ants - no_reach.\n")
        logf.write("#\n")
        logf.write("# %5s %8s %9s %8s %8s %9s %9s %8s %10s %6s %14s %10s\n"
                   % ("iter", "reached", "no_reach", "rej_len", "rej_rad", "rej_keep",
                      "feasible", "fem_new", "cache_hit", "q0", "best_J", "best_len_mm"))
        logf.flush()

    # geodesic distance-to-target field for the balanced detour budget: one Dijkstra
    # sweep, shared by every ant of every iteration (only needed when gamma > 0)
    dist_target = distances_from(adj, target_id) if p.get("slack_balance", 0.0) > 0.0 else None

    # Bidirectional construction: n_rev of the n_ants build from B toward A. Their
    # completed node sequence is reversed to the canonical A->B orientation before
    # everything downstream (geometry, cache, objective, deposit), so only the
    # construction-time exploration changes: a B-started ant consumes its detour
    # slack near B, which an A-started ant structurally cannot. The prescribed
    # direction cones swap roles and flip sign for the reversed walk: the travel
    # direction with which an A->B route must ENTER B is minus the direction with
    # which a B-started ant LEAVES B, and vice versa at A.
    n_rev = int(round(p.get("bidirectional", 0.0) * p["n_ants"]))
    p_rev = None
    dist_source = None
    if n_rev > 0:
        p_rev = dict(p)
        _sd, _ed = p.get("start_dir"), p.get("end_dir")
        p_rev["start_dir"] = (-np.asarray(_ed, float)) if _ed is not None else None
        p_rev["end_dir"] = (-np.asarray(_sd, float)) if _sd is not None else None
        p_rev["start_max_turn_deg"] = p.get("end_max_turn_deg")
        p_rev["end_max_turn_deg"] = p.get("start_max_turn_deg")
        if p.get("slack_balance", 0.0) > 0.0:
            dist_source = distances_from(adj, start_id)

    # optional pool of worker processes for the per-iteration objective evaluations
    eval_pool = _make_eval_pool(objective_fn, p.get("n_eval_workers"))

    last_kick = 0     # iteration after the most recent stagnation kick (for q0 rewind)
    for it in range(p["max_iter"]):
        q0_it = _q0_at(p, it - last_kick if p.get("q0_rewind_on_kick") else it)
        ants = []
        raw_paths = []
        it_no_reach = 0                          # ants that never reached B
        it_rej_len = it_rej_rad = it_rej_keep = 0  # reached B but failed the geometry filter
        it_fem_new = 0                           # new FEM solves this iteration
        it_cache_hit = 0                         # feasible routes reused from the cache
        cand = []      # feasible candidates of this iteration: (path, length, key, coords)
        for a_idx in range(p["n_ants"]):
            if a_idx < n_rev:
                # reversed construction: walk B -> A, then canonicalize to A -> B
                path, length = build_one_ant(grid, adj, tau,
                                             target_id, start_id, Lmax, p_rev, rng,
                                             q0=q0_it, dist_target=dist_source)
                if path is not None:
                    path = path[::-1]
            else:
                path, length = build_one_ant(grid, adj, tau,
                                             start_id, target_id, Lmax, p, rng,
                                             q0=q0_it, dist_target=dist_target)
            if path is None:
                it_no_reach += 1
                continue
            coords = path_coords(grid, path)
            # exact as-built feasibility on the spline (length and curvature),
            # checked before the objective so infeasible routes cost no FEM solve
            if geom_fn is not None:
                g = geom_fn(coords)
                spline_len, min_rad = g[0], g[1]
                keepout_ok = g[2] if len(g) > 2 else True  # optional 3rd value: extra geometry feasible (e.g. keep-out)
                if spline_len > Lmax + 1e-9:
                    n_reject_len += 1
                    it_rej_len += 1
                    continue
                if p["R_min"] > 0.0 and min_rad < p["R_min"]:
                    n_reject_rad += 1
                    it_rej_rad += 1
                    continue
                if not keepout_ok:
                    n_reject_keepout += 1
                    it_rej_keep += 1
                    continue
                # Store the as-built spline length for reporting and for the optional
                # length-based reward used only by deposit_mode="all".
                length = spline_len
            raw_paths.append(path)
            cand.append((path, length, tuple(path), coords))

        # Objective evaluation of the iteration's candidates. The routes are
        # independent cases, so the uncached ones can be solved in parallel worker
        # processes (eval_pool); the values, and therefore the search dynamics, are
        # identical to the serial evaluation - only the wall-clock changes.
        new_keys, new_coords, new_set = [], [], set()
        for (_, _, key, coords) in cand:
            if key not in cache and key not in new_set:
                new_set.add(key)
                new_keys.append(key)
                new_coords.append(coords)
        if eval_pool is not None and len(new_coords) > 1:
            new_powers = eval_pool.map(_eval_worker, new_coords)
        else:
            new_powers = [objective_fn(c) for c in new_coords]
        for key, power in zip(new_keys, new_powers):
            cache[key] = power
            n_fem += 1
            it_fem_new += 1
        first_seen = set()
        for (path, length, key, _) in cand:
            if key in new_set and key not in first_seen:
                first_seen.add(key)          # the occurrence that triggered the solve
            else:
                it_cache_hit += 1            # pre-cached, or duplicate within the iteration
            ants.append({"path": path, "length": length, "power": cache[key]})

        if ants:
            powers = np.array([a["power"] for a in ants])
            it_best = ants[int(np.argmin(powers))]
            improved = best is None or it_best["power"] < best["power"]
            # a gain below improve_tol still updates the incumbent, but does not count as
            # progress for the stagnation and patience counters: otherwise sub-percent
            # gains keep postponing the restart while the search is, in practice, stalled
            significant = best is None or \
                it_best["power"] < (1.0 - p.get("improve_tol", 0.0)) * best["power"]
            if improved:
                best = {k: it_best[k] for k in ("path", "length", "power")}
            if significant:
                no_improve = 0
            else:
                no_improve += 1
            best_hist.append(best["power"])
            mean_hist.append(float(powers.mean()))
            iter_best_hist.append(float(powers.min()))
        else:
            best_hist.append(best["power"] if best else np.nan)
            mean_hist.append(np.nan)
            iter_best_hist.append(np.nan)
            no_improve += 1
        feasible_hist.append(len(ants))
        fem_hist.append(n_fem)

        # per-iteration diagnostic record (returned as result["iter_log"] and logged)
        n_reached = p["n_ants"] - it_no_reach
        rec = {"iter": it, "reached": n_reached, "no_reach": it_no_reach,
               "rej_len": it_rej_len, "rej_rad": it_rej_rad, "rej_keep": it_rej_keep,
               "feasible": len(ants), "fem_new": it_fem_new, "cache_hit": it_cache_hit,
               "q0": q0_it, "best_J": (best["power"] if best else float("nan")),
               "best_len_mm": (best["length"] * 1e3 if best else float("nan"))}
        iter_log.append(rec)
        if logf is not None:
            logf.write("  %5d %8d %9d %8d %8d %9d %9d %8d %10d %6.2f %14.6e %10.1f\n"
                       % (it, n_reached, it_no_reach, it_rej_len, it_rej_rad, it_rej_keep,
                          len(ants), it_fem_new, it_cache_hit, q0_it,
                          rec["best_J"], rec["best_len_mm"]))
            logf.flush()

        deposit(tau, ants, p["rho"], p, best=best)
        deposit_elitist(tau, best, p)
        _clamp(tau, p.get("tau_bounds"))

        # stagnation-triggered pheromone reset (MMAS): pull every trail toward tau_max.
        # Stagnation means the population has concentrated around an incumbent that no
        # longer improves, so four conditions are required: (i) no significant gain for
        # stagnation_reset iterations, (ii) at least stagnation_reset iterations since
        # the last kick or the start, (iii) the exploration schedule has ended (during
        # the q0 ramp the colony explores by design), and (iv) the feasible population
        # is concentrated: the median objective of the iteration lies within
        # kick_concentration x the incumbent. Without (iv) a lucky early incumbent
        # would trigger a reset while the population is still descending toward it
        # (observed on the 18^3 lattice).
        since_kick += 1
        concentrated = bool(ants) and best is not None and \
            float(np.median(powers)) < p.get("kick_concentration", 3.0) * best["power"]
        if p.get("stagnation_reset") and p.get("tau_bounds") \
                and no_improve >= p["stagnation_reset"] \
                and since_kick >= p["stagnation_reset"] \
                and it >= (p.get("q0_iters") or 0) and concentrated:
            tmax = p["tau_bounds"][1]
            delta = p.get("stagnation_smooth", 1.0)
            for e in tau:
                tau[e] = tau[e] + delta * (tmax - tau[e])
            n_stagnation += 1
            reset_iters.append(it)
            since_kick = 0
            if p.get("q0_rewind_on_kick"):
                last_kick = it + 1
            if verbose:
                how = "reset" if delta >= 1.0 else f"smoothing (delta={delta:.2f})"
                print(f"  iter {it:3d} | stagnation kick #{n_stagnation}: {how} "
                      f"after {no_improve} stale iters")

        if p["store_paths_every"] and (it % p["store_paths_every"] == 0):
            snapshots.append({"iter": it,
                              "paths": [list(pp) for pp in raw_paths],
                              "best": list(best["path"]) if best else None})

        if verbose and (it % int(p.get("print_every", 10)) == 0 or it == p["max_iter"] - 1):
            bp = best["power"] if best else float("nan")
            bl = best["length"] if best else float("nan")
            if geom_fn is not None:
                print(f"  iter {it:3d} | reached {p['n_ants']-it_no_reach:2d}/{p['n_ants']} | "
                      f"feasible {len(ants):2d} | rej L/{it_rej_len} R/{it_rej_rad} K/{it_rej_keep} | "
                      f"FEM +{it_fem_new} (cache {it_cache_hit}, {n_fem} total) | q0={q0_it:.2f} | "
                      f"best {bp:.4e} ({bl*1e3:7.1f} mm)")
            else:
                print(f"  iter {it:3d} | feasible ants {len(ants):2d} | q0={q0_it:.2f} | "
                      f"best power {bp:.4e} | best len {bl*1e3:7.1f} mm | FEM {n_fem}")

        if no_improve >= p["patience"]:
            if verbose:
                print(f"  early stop at iter {it} (no improvement for {p['patience']} iters)")
            if logf is not None:
                logf.write("# early stop at iter %d (no improvement for %d iters)\n"
                           % (it, p["patience"]))
            break

    if eval_pool is not None:
        eval_pool.close()
        eval_pool.join()

    if logf is not None:
        logf.write("# final best J=%s  length=%s mm  |  distinct routes FEM-evaluated=%d  |  wall time %.1f s\n"
                   % (("%.6e" % best["power"]) if best else "none",
                      ("%.1f" % (best["length"] * 1e3)) if best else "none", n_fem,
                      _time.time() - _t_start))
        logf.close()

    return {
        "best_path": best["path"] if best else None,
        "best_power": best["power"] if best else None,
        "best_length": best["length"] if best else None,
        "best_hist": np.array(best_hist),
        "mean_hist": np.array(mean_hist),
        "iter_best_hist": np.array(iter_best_hist),
        "iter_log": iter_log,
        "reset_iters": list(reset_iters),
        "initial_best_from_seed": initial_best_from_seed,
        "feasible_hist": np.array(feasible_hist),
        "fem_hist": np.array(fem_hist),
        "snapshots": snapshots,
        "evaluated_paths": list(cache.keys()),   # every feasible route that was FEM-evaluated
        "cache": cache,                          # route tuple -> objective (search mesh)
        "tau": tau,
        "n_fem_calls": n_fem,
        "n_reject_length": n_reject_len,
        "n_reject_radius": n_reject_rad,
        "n_reject_keepout": n_reject_keepout,
        "n_stagnation_events": n_stagnation,
        "params": p,
        "wall_s": _time.time() - _t_start,       # wall-clock time of this run [s]
    }


# --------------------------------------------------------------------------- #
# Multi-start: run several independent seeds and keep the best
# --------------------------------------------------------------------------- #
_MS_CTX = {}


def _ms_worker(seed):
    """Worker for the parallel multi-start (problem is inherited via fork)."""
    c = _MS_CTX
    return optimize(c["grid"], c["adj"], c["sid"], c["tid"], c["Lmax"],
                    c["objective_fn"], params=c["params"], seed=seed,
                    geom_fn=c["geom_fn"], verbose=False, seed_paths=c.get("seed_paths"),
                    log_path=c.get("log_paths", {}).get(seed))


def optimize_multistart(grid, adj, start_id, target_id, Lmax, objective_fn,
                        params=None, seeds=(0,), geom_fn=None,
                        n_workers=None, verbose=True, seed_paths=None, log_path=None):
    """Run optimize() from several seeds and return the best result.

    Each seed is an independent ACO run (a multi-start), which guards against the
    search settling in a poor local optimum on this multimodal landscape: additional
    seeds increase the chance of reaching a better basin, at a cost that grows roughly
    linearly with the number of seeds. The seeds are independent, so they are run in
    parallel processes when ``n_workers != 1`` and fork is available. On platforms
    without fork, including Windows and standard macOS spawn configurations, the current
    implementation falls back to serial execution.

    Returns the best-of-seeds result dict (the one with the lowest input power),
    augmented with ``result["multistart"] = {seeds, best_powers, winning_seed,
    total_fem_calls}``. With a single seed this is identical to optimize().

    ``seeds`` may be an integer (the number of runs, using seeds 0..n-1) or an
    explicit list of seeds (for exact reproducibility).
    """
    if isinstance(seeds, int):
        seeds = list(range(seeds))     # a plain count -> seeds 0, 1, ..., n-1
    else:
        seeds = list(seeds)            # an explicit list of seeds

    def _seed_log(s):
        # one log per seed; with several seeds the seed index is inserted before the
        # extension (foo.txt -> foo_seed3.txt), otherwise the given path is used as is
        if log_path is None:
            return None
        if len(seeds) == 1:
            return log_path
        root, dot, ext = log_path.rpartition(".")
        return ("%s_seed%d.%s" % (root, s, ext)) if dot else ("%s_seed%d" % (log_path, s))

    def _serial():
        out = []
        # every seed gets its own route cache (as in the parallel path), so that the
        # n_fem_calls reported for a seed is the true cost of that run; sharing the
        # cache would not change any trajectory, only under-count the later seeds
        for i, s in enumerate(seeds, 1):
            if verbose and len(seeds) > 1:
                print("  seed %d/%d (seed=%d) running ..." % (i, len(seeds), s), flush=True)
            r = optimize(grid, adj, start_id, target_id, Lmax, objective_fn,
                         params=params, seed=s, geom_fn=geom_fn,
                         verbose=(verbose and len(seeds) == 1), seed_paths=seed_paths,
                         cache=None, log_path=_seed_log(s))
            if verbose and len(seeds) > 1:
                p = r["best_power"] if r["best_power"] is not None else float("inf")
                print("    seed=%d done: power %.4e" % (s, p), flush=True)
            out.append(r)
        return out

    if len(seeds) == 1 or n_workers == 1:
        results = _serial()
    else:
        import multiprocessing as mp
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = None
        if ctx is not None:
            # true parallelism via separate processes (Linux fork): no GIL, full speedup.
            # The cores are already claimed by the seed processes, so the per-iteration
            # evaluation pool is disabled inside each worker: nesting the two pools would
            # oversubscribe the machine without changing any result.
            ms_params = dict(params) if params else {}
            ms_params["n_eval_workers"] = None
            global _MS_CTX
            _MS_CTX = dict(grid=grid, adj=adj, sid=start_id, tid=target_id, Lmax=Lmax,
                           objective_fn=objective_fn, params=ms_params, geom_fn=geom_fn,
                           seed_paths=seed_paths, log_paths={s: _seed_log(s) for s in seeds})
            if verbose:
                print("  running %d seeds in parallel (%s processes)"
                      % (len(seeds), n_workers or "auto"), flush=True)
            with ctx.Pool(processes=n_workers) as pool:
                results = pool.map(_ms_worker, seeds)
        else:
            # No fork (Windows, or macOS spawn). Threads do NOT help here: these beam-FEM
            # systems are small, so each run is largely Python-bound and thread contention
            # makes it slower than serial. Run serially with per-seed progress instead. For
            # true parallelism on Windows, process-based execution is needed (a __main__ guard
            # plus a picklable objective); ask if you want that set up.
            if verbose:
                print("  (no fork on this platform; running %d seeds serially)" % len(seeds),
                      flush=True)
            results = _serial()

    return combine_multistart(results, seeds, verbose)


def combine_multistart(results, seeds, verbose=True):
    """Best-of-seeds result dict from the per-seed optimize() results (shared by
    optimize_multistart and by run_campaign.py, which runs the seeds in separate
    processes on platforms without fork)."""
    seeds = list(seeds)
    powers = [(r["best_power"] if r["best_power"] is not None else np.inf) for r in results]
    iwin = int(np.argmin(powers))
    best = results[iwin]
    # union of the routes evaluated across all seeds (deduplicated), for the spline-cloud plot
    _union = {}
    for r in results:
        for pth, J in r.get("cache", {}).items():
            _union[tuple(pth)] = J      # deterministic objective: duplicates carry the same J
    best["evaluated_paths"] = list(_union.keys())
    # per-seed route sets, for one spline-cloud figure per seed
    best["per_seed"] = {s: {"evaluated_paths": list(r.get("cache", {}).keys()),
                            "best_path": r["best_path"], "best_power": r["best_power"],
                            "n_fem_calls": int(r["n_fem_calls"])}
                        for s, r in zip(seeds, results)}
    best["cache_union"] = _union        # every distinct route evaluated by any seed -> J
    finite = [q for q in powers if np.isfinite(q)]
    best["multistart"] = {
        "seeds": seeds,
        "best_powers": [r["best_power"] for r in results],
        "fem_calls": [int(r["n_fem_calls"]) for r in results],
        "n_iter": [int(len(r["best_hist"])) for r in results],
        "n_stagnation_events": [int(r["n_stagnation_events"]) for r in results],
        "wall_s": [float(r.get("wall_s", float("nan"))) for r in results],
        "wall_total_s": float(sum(r.get("wall_s", 0.0) for r in results)),   # serial-equivalent
        "winning_seed": seeds[iwin],
        "total_fem_calls": int(sum(r["n_fem_calls"] for r in results)),
        # seed-to-seed spread of the converged objective (reported with the best route)
        "J_min": float(min(finite)) if finite else None,
        "J_median": float(np.median(finite)) if finite else None,
        "J_max": float(max(finite)) if finite else None,
    }
    if verbose:
        spread = ", ".join("s%d=%.3e" % (s, q) for s, q in zip(seeds, powers))
        print("  multi-start: winning seed %d | %s | total FEM %d across %d seeds"
              % (seeds[iwin], spread, best["multistart"]["total_fem_calls"], len(seeds)))
        if len(finite) > 1:
            ms = best["multistart"]
            print("  seed spread: min %.3e | median %.3e | max %.3e (max/min = %.2f)"
                  % (ms["J_min"], ms["J_median"], ms["J_max"], ms["J_max"] / ms["J_min"]))
    return best

