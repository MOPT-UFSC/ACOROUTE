"""calibrate_smoothing.py
Calibration sweep for the per-waypoint smoothing tolerance SMOOTH_RMS.

The smoothing level is a first-order design lever: tighter fits retain the
lattice-scale curvature that lowers the band-integrated input power, at the cost
of more minimum-bend-radius rejections. This tool quantifies that trade on a
fixed pool of constructed routes (no ACO, ~2 min): for each candidate tolerance
it reports the feasibility pass rate and the median and minimum objective of the
feasible set. Choose the smallest tolerance whose pass rate is still acceptable.

Edit the configuration block to match the case, then run:
    python calibrate_smoothing.py

Methodology:
Fixed pool of constructed routes (13^3, xi=0.5, theta_max=90). For each rule:
filter by ITS OWN feasibility (R_min, Lmax), evaluate J of its feasible set,
report pass rate, median J and -- the search-relevant number -- the MINIMUM J.
Rules: legacy s=5e-4 | eps = 4.17mm (h/6) | 3.5 | 3.0 | 2.5 mm.
"""
import sys, numpy as np
import grid as G
import aco as A
import geometry as geo
import fem_beam3d as fem

NODES, BOX = 13, 0.300
E, nu, rho = 210e9, 0.3, 7850.0
mat = {"E": E, "G": E/(2*(1+nu)), "rho": rho}
sec = fem.pipe_section(0.001575, 0.0005, nu)
BAND = (4000.0, 6000.0); RAY = (0.0, 2e-7); ELEM = 0.005
R_MIN = 0.008
NPOOL = 40

h = BOX/(NODES-1)
g = G.build_grid(NODES, NODES, NODES, h)
adj = G.build_neighbours(g, 26)
sid, tid = g["id_of"][(0,0,0)], g["id_of"][(NODES-1,)*3]
d0 = float(np.linalg.norm(g["coords"][tid]-g["coords"][sid]))
Lmax = 1.7*d0

p = A.default_params(); p["q0"]=0.2; p["max_turn_deg"]=90.0; p["slack_balance"]=0.5
rng = np.random.default_rng(21)
dist_t = G.distances_from(adj, tid)

pool = []
while len(pool) < NPOOL:
    path, _ = A.build_one_ant(g, adj, {}, sid, tid, Lmax, p, rng, dist_target=dist_t)
    if path is not None:
        pool.append(g["coords"][np.asarray(path)])
print(f"pool: {NPOOL} routes, waypoints {min(len(c) for c in pool)}-{max(len(c) for c in pool)}")

RULES = [("legacy s=5e-4", dict(smooth=5e-4)),
         ("eps 4.17mm(h/6)", dict(smooth_rms=h/6)),
         ("eps 3.5mm", dict(smooth_rms=0.0035)),
         ("eps 3.0mm", dict(smooth_rms=0.0030)),
         ("eps 2.5mm", dict(smooth_rms=0.0025))]

print(f"{'rule':16s} {'pass':>6s} {'min J':>13s} {'median J':>13s} {'min R [mm]':>11s}")
for name, kw in RULES:
    Js, Rs = [], []
    for coords in pool:
        tck = geo.fit_spline(coords, **kw)
        R, _ = geo.min_radius(tck)
        L = geo.total_length(tck)
        if L > Lmax or R < R_MIN:
            continue
        nodes, elems, _ = geo.resample_by_length(tck, ELEM)
        J = fem.band_integrated_power(nodes, elems, mat, sec, [BAND],
                                      freq_mode="adaptive", rayleigh=RAY,
                                      excitation="prescribed",
                                      prescribed_disp=(1e-6,1e-6,1e-6),
                                      prescribed_mask=(True,True,True))
        Js.append(J); Rs.append(R)
    if Js:
        print(f"{name:16s} {len(Js):3d}/{NPOOL:<3d} {min(Js):13.4e} {np.median(Js):13.4e} {min(Rs)*1e3:11.1f}")
    else:
        print(f"{name:16s} {0:3d}/{NPOOL:<3d} {'-':>13s} {'-':>13s}")
