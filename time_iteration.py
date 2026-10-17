"""
time_iteration.py
Where does the time of one ACO iteration go? Loads the header of a run script, builds
n routes with a uniform pheromone field (construction only), applies the geometric
tests (geom_fn) and evaluates the first admissible routes with the objective used in
the search. Prints the mean cost of each stage and the implied cost of one iteration
(n_ants constructions + geometry tests + FEM on the admissible ones).

    python time_iteration.py run_sphere.py [n_routes] [n_fem]
"""
import sys, io, contextlib, time
import numpy as np

script = sys.argv[1] if len(sys.argv) > 1 else "run_sphere.py"
n_routes = int(sys.argv[2]) if len(sys.argv) > 2 else 60
n_fem = int(sys.argv[3]) if len(sys.argv) > 3 else 5

src = open(script, encoding="utf-8").read()
i = src.index("A.optimize_multistart"); head = src[:src.rfind("\n", 0, i)].replace("SHOW = True", "SHOW = False")
ns = {"__name__": "time_head"}
t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, script + "[head]", "exec"), ns)
print("%s | header %.1f s | %d retained nodes | Lmax %.0f mm" % (script, time.perf_counter() - t0, len(ns["grid"]["coords"]), ns["Lmax"] * 1e3))
A, G, grid, adj, sid, tid, Lmax = ns["A"], ns["G"], ns["grid"], ns["adj"], ns["sid"], ns["tid"], ns["Lmax"]
params = A.default_params(); params.update(ns["params"] if "params" in ns else ns["BASE"])
geom_fn, objective = ns["geom_fn"], ns["objective_spline"]
from grid import distances_from
dist_t = distances_from(adj, tid)

try:
    import threadpoolctl
    info = threadpoolctl.threadpool_info()
    print("BLAS threads in this process: %s" % sorted({(d["internal_api"], d["num_threads"]) for d in info}))
except Exception:
    pass

rng = np.random.default_rng(1)
t_build, t_geom, t_fem = [], [], []
n_done = n_ok = 0; feasible = []
for k in range(n_routes):
    t0 = time.perf_counter()
    path, _ = A.build_one_ant(grid, adj, {}, sid, tid, Lmax, params, rng, q0=0.0, dist_target=dist_t)
    t_build.append(time.perf_counter() - t0)
    if path is None:
        continue
    n_done += 1
    c = G.path_coords(grid, path)
    t0 = time.perf_counter(); L, R, ok = geom_fn(c); t_geom.append(time.perf_counter() - t0)
    if ok:
        n_ok += 1; feasible.append(c)
for c in feasible[:n_fem]:
    t0 = time.perf_counter(); J = objective(c); t_fem.append(time.perf_counter() - t0)

n_ants = params.get("n_ants", n_routes)
print("construction : mean %.3f s, max %.3f s  (%d of %d completed)" % (np.mean(t_build), np.max(t_build), n_done, n_routes))
print("geom_fn      : mean %.3f s                (%d of %d admissible)" % (np.mean(t_geom) if t_geom else float("nan"), n_ok, n_done))
if t_fem:
    print("objective    : mean %.2f s per route (%d evaluated)" % (np.mean(t_fem), len(t_fem)))
    frac = n_ok / max(n_done, 1)
    est = n_ants * (np.mean(t_build) + np.mean(t_geom)) + n_ants * frac * np.mean(t_fem)
    print("one iteration of %d ants, serial, at this admissibility: about %.0f s (%.0f s of it in the FEM)"
          % (n_ants, est, n_ants * frac * np.mean(t_fem)))
