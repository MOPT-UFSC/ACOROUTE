"""
validate_construction.py
Construction-rule validation of the paper (Section 'Construction-rule validation'),
reproduced on the lattice of a run_*.py case without any finite-element analysis:

    python validate_construction.py [run_spline.py] [n_routes] [part]

part = 1 runs the xi / bidirectional profiles (any case), part = 2 the geodesic-field
check (obstacle cases, e.g. run_staircase.py), default both.

For gamma (slack_balance) in {0, 0.5, 1}, `n_routes` routes are completed with a
UNIFORM pheromone field (q0 = 0, so only the construction rules act: length rule,
turn-angle pre-filter, end cones, tabu list, backtracking). For every completed route
the turn angle at each interior waypoint is assigned to one of ten equal intervals of
normalized polyline arc length from A to B; the table reports the mean turn angle per
interval over all routes, the mean polyline length and the completion rate. The same
is then done with the bidirectional construction (half of the routes started at B).
"""
import sys, io, contextlib, time
import numpy as np

script = sys.argv[1] if len(sys.argv) > 1 else "run_spline.py"
n_routes = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PART = sys.argv[3] if len(sys.argv) > 3 else "both"   # "1" (xi / bidirectional profiles), "2" (geodesic field) or "both"

src = open(script, encoding="utf-8").read()
i = src.index("A.optimize_multistart"); head = src[:src.rfind("\n", 0, i)].replace("SHOW = True", "SHOW = False")
ns = {"__name__": "validate_construction_head"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, script + "[head]", "exec"), ns)
G, grid, adj, sid, tid, Lmax, A = ns["G"], ns["grid"], ns["adj"], ns["sid"], ns["tid"], ns["Lmax"], ns["A"]
from grid import distances_from
base = A.default_params(); base.update(ns["BASE"])
base["max_turn_deg"] = ns["MAX_TURN"]; base["R_min"] = ns["R_MIN"]
for k in ("start_dir", "end_dir", "start_max_turn_deg", "end_max_turn_deg"):
    base.pop(k, None)
if ns.get("EXIT_DIR_A") is not None:
    base["start_dir"] = ns["EXIT_DIR_A"]; base["start_max_turn_deg"] = ns["EXIT_CONE_DEG"]
if ns.get("ARRIVE_DIR_B") is not None:
    base["end_dir"] = ns["ARRIVE_DIR_B"]; base["end_max_turn_deg"] = ns["ARRIVE_CONE_DEG"]
coords = grid["coords"]
dist_t = distances_from(adj, tid); dist_s = distances_from(adj, sid)
print("%s | lattice %s | Lmax %.1f mm | theta_max %.1f deg | %d routes per setting"
      % (script, "x".join(str(n) for n in ns["NODES"]), Lmax * 1e3, ns["MAX_TURN"], n_routes))


def reversed_params(p):
    pr = dict(p)
    sd, ed = p.get("start_dir"), p.get("end_dir")
    pr["start_dir"] = (-np.asarray(ed, float)) if ed is not None else None
    pr["end_dir"] = (-np.asarray(sd, float)) if sd is not None else None
    pr["start_max_turn_deg"] = p.get("end_max_turn_deg"); pr["end_max_turn_deg"] = p.get("start_max_turn_deg")
    return pr


def turn_profile(path):
    """(decile index, turn angle [deg]) for every interior waypoint of a lattice route."""
    pts = coords[np.asarray(path)]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)]) / seg.sum()
    d = np.diff(pts, axis=0); d /= np.linalg.norm(d, axis=1)[:, None]
    cosang = np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    dec = np.minimum((s[1:-1] * 10).astype(int), 9)
    return dec, ang, seg.sum()


def run(gamma, bidirectional, seed=1):
    rng = np.random.default_rng(seed)
    p = dict(base); p["slack_balance"] = gamma
    pr = reversed_params(p)
    sums = np.zeros(10); counts = np.zeros(10); lengths = []; tries = 0
    max_tries = 20 * n_routes            # a setting that cannot complete routes stops here
    while len(lengths) < n_routes and tries < max_tries:
        tries += 1
        if bidirectional and rng.random() < 0.5:
            path, _ = A.build_one_ant(grid, adj, {}, tid, sid, Lmax, pr, rng, q0=0.0,
                                      dist_target=dist_s if gamma > 0 else None)
            path = path[::-1] if path is not None else None
        else:
            path, _ = A.build_one_ant(grid, adj, {}, sid, tid, Lmax, p, rng, q0=0.0,
                                      dist_target=dist_t if gamma > 0 else None)
        if path is None:
            continue
        dec, ang, L = turn_profile(path)
        np.add.at(sums, dec, ang); np.add.at(counts, dec, 1); lengths.append(L)
    prof = sums / np.maximum(counts, 1)
    return prof, (np.mean(lengths) if lengths else float("nan")), len(lengths) / tries


if PART in ("1", "both"):
  print("\nmean turn angle [deg] per tenth of normalized arc length (A -> B)", flush=True)
  print("  %-22s %s | mean L [mm] | completion" % ("setting", " ".join("%5s" % ("%d" % (10 * (k + 1))) for k in range(10))), flush=True)
  for bidir in (False, True):
    for gamma in (0.0, 0.5, 1.0):
        t0 = time.perf_counter()
        prof, Lmean, rate = run(gamma, bidir)
        tag = "xi=%.1f %s" % (gamma, "bidirectional" if bidir else "A-started")
        print("  %-22s %s |   %7.1f   |   %.2f   (%.0f s)"
              % (tag, " ".join("%5.1f" % v for v in prof), Lmean * 1e3, rate, time.perf_counter() - t0), flush=True)


# --------------------------------------------------------------------------- part 2
# Geodesic-field check (meaningful only when the domain has obstacles): completion
# rate of 300 construction attempts with the classic straight-line length test
# (xi = 0), with the balanced rule using the straight-line distance in the reserve term
# (xi = 0.5, Euclidean), and with the balanced rule on the geodesic field (xi = 0.5).
# The turn-angle filter and the end cones are switched off so that only the length
# rule acts, as in the paper.
N_ATT = 300
if PART not in ("2", "both"):
    sys.exit(0)
p0 = dict(base); p0["max_turn_deg"] = None
for k in ("start_dir", "end_dir", "start_max_turn_deg", "end_max_turn_deg"):
    p0.pop(k, None)
eucl_t = {j: float(np.linalg.norm(coords[j] - coords[tid])) for j in range(len(coords))}
print("\ngeodesic-field check: completed routes out of %d attempts (no turn filter, no cones)" % N_ATT)
for tag, gamma, field in (("xi=0, straight-line test", 0.0, None),
                          ("xi=0.5, straight-line reserve", 0.5, eucl_t),
                          ("xi=0.5, geodesic field", 0.5, dist_t)):
    rng = np.random.default_rng(7); p = dict(p0); p["slack_balance"] = gamma
    done = 0; t0 = time.perf_counter()
    for _ in range(N_ATT):
        path, _L = A.build_one_ant(grid, adj, {}, sid, tid, Lmax, p, rng, q0=0.0, dist_target=field)
        done += path is not None
    print("  %-32s %3d / %d   (%.0f s)" % (tag, done, N_ATT, time.perf_counter() - t0), flush=True)
