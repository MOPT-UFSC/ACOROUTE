"""
validate_modal.py
Checks the modal harmonic solver against the direct banded solver on admissible routes
of the case defined in run_spline.py (or another run_*.py given on the command line):

    python validate_modal.py [run_spline.py] [n_routes]

For every band of the script, the band-integrated power J of each route is computed
with HARMONIC_SOLVER = "direct", "modal" with the complete basis (MODAL_FCUT = None)
and "modal" with the truncated basis plus residual-flexibility correction
(MODAL_FCUT = 3 and 2). It reports the time per route, the largest relative deviation
from the direct solver and whether the ranking of the routes is preserved. Figures
to quote: the complete basis reproduces the direct solver to round-off; the
truncated basis with MODAL_FCUT = 3 stays below 1e-3 on the cases tested.
"""
import sys, time, io, contextlib
import numpy as np

script = sys.argv[1] if len(sys.argv) > 1 else "run_spline.py"
n_routes = int(sys.argv[2]) if len(sys.argv) > 2 else 6

src = open(script, encoding="utf-8").read()
i = src.index("A.optimize_multistart"); head = src[:src.rfind("\n", 0, i)].replace("SHOW = True", "SHOW = False")
ns = {"__name__": "validate_modal_head"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, script + "[head]", "exec"), ns)
G, grid, adj, sid, tid, Lmax = ns["G"], ns["grid"], ns["adj"], ns["sid"], ns["tid"], ns["Lmax"]
geom, A, obj = ns["geom_fn"], ns["A"], ns["objective_spline"]
p = A.default_params(); p.update(ns["BASE"]); p["max_turn_deg"] = ns["MAX_TURN"]; p["R_min"] = ns["R_MIN"]

rng = np.random.default_rng(21); routes = []
while len(routes) < n_routes:
    path, _L = A.build_one_ant(grid, adj, {}, sid, tid, Lmax, p, rng, q0=0.0, dist_target=None)
    if path is None:
        continue
    c = G.path_coords(grid, path); g = geom(c)
    if g[0] <= Lmax and g[1] >= p["R_min"] and (g[2] if len(g) > 2 else True):
        routes.append(c)
print("%s | %d admissible routes | element %.1f mm | bands %s" % (script, len(routes), ns["ELEM"] * 1e3, ns["BANDS"]))

bands_all = list(ns["BANDS"])
for band in bands_all:
    ns["BANDS"][:] = [band]
    res = {}
    for label, solver, fcut in [("direct", "direct", None), ("modal full", "modal", None),
                                ("modal 3 f2", "modal", 3.0), ("modal 2 f2", "modal", 2.0)]:
        ns["HARMONIC_SOLVER"] = solver; ns["MODAL_FCUT"] = fcut
        t = time.perf_counter(); J = np.array([obj(c) for c in routes]); dt = (time.perf_counter() - t) / len(routes)
        res[label] = (J, dt)
    ref = res["direct"][0]
    print("band %s" % (band,))
    for label, (J, dt) in res.items():
        print("  %-11s %.3f s/route | max |dJ/J| = %.1e | ranking preserved: %s"
              % (label, dt, np.max(np.abs(J / ref - 1)), np.array_equal(np.argsort(J), np.argsort(ref))))
ns["BANDS"][:] = bands_all
