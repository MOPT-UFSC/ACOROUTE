"""
make_fig_balanced_budget.py
Figure for the balanced-detour-budget subsection of the paper: mean turn angle
versus arc-length fraction for xi in {0, 0.5, 1} on the reference configuration
(300 mm cube, 11x11x11 lattice, 26-connectivity, chi_L = 1.7, theta_max = 90 deg),
with uniform pheromone fields and q0 = 0 so that only the construction rules act.
The measurement is construction-level (no spline filter), matching the claim in
the text: with the classic bound (xi = 0) the mean turn angle collapses over the
final part of the route, whereas the balanced rule preserves the turning
capability up to the terminal node.

Run:  python make_fig_balanced_budget.py
Writes fig_balanced_budget.pdf and fig_balanced_budget.png in the current
directory (copy them into the Overleaf project).
"""
import numpy as np
import plot_style  # noqa: F401  (sets the paper's serif typography on import)
import matplotlib
import geometry as geo
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grid import build_grid, build_neighbours, distances_from
from aco import build_one_ant, default_params

NODES = 11          # reference lattice: 11 nodes per axis
BOX = 0.300         # [m] cubic envelope side
LMAX_FACTOR = 1.7   # chi_L
NANTS = 400         # completed routes per xi
NBINS = 10          # arc-length bins from A to B
XIS = (0.0, 0.5, 1.0)
SEED = 1

R_MIN    = 0.008    # [m] minimum bend radius of the reported cases
MAX_TURN = None     # set below from the pitch and R_MIN (eq:theta_max)

h = BOX / (NODES - 1)
g = build_grid(NODES, NODES, NODES, h)

MAX_TURN = geo.max_turn_from_pitch(h, R_MIN)
print("turn-angle pre-filter: theta_max = %.1f deg (h = %.1f mm, R_min = %.0f mm)"
      % (MAX_TURN, h * 1e3, R_MIN * 1e3))

adj = build_neighbours(g, 26)
sid = g["id_of"][(0, 0, 0)]
tid = g["id_of"][(NODES - 1, NODES - 1, NODES - 1)]
d0 = float(np.linalg.norm(g["coords"][tid] - g["coords"][sid]))
Lmax = LMAX_FACTOR * d0

curves = {}
for xi in XIS:
    p = default_params()
    p["q0"] = 0.0                  # uniform sampling: construction bias only
    p["max_turn_deg"] = MAX_TURN
    p["slack_balance"] = xi
    rng = np.random.default_rng(SEED)
    dist_t = distances_from(adj, tid) if xi > 0 else None
    tsum = np.zeros(NBINS)
    tcnt = np.zeros(NBINS)
    built = 0
    for _ in range(NANTS):
        path, _L = build_one_ant(g, adj, {}, sid, tid, Lmax, p, rng,
                                 dist_target=dist_t)
        if path is None:
            continue
        built += 1
        c = g["coords"][np.asarray(path)]
        seg = np.diff(c, axis=0)
        L = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(L)])
        tot = cum[-1]
        for k in range(1, len(seg)):
            u = seg[k - 1] / L[k - 1]
            v = seg[k] / L[k]
            ang = np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
            b = min(NBINS - 1, int(NBINS * cum[k] / tot))
            tsum[b] += ang
            tcnt[b] += 1
    curves[xi] = np.where(tcnt > 0, tsum / np.maximum(tcnt, 1), np.nan)
    print("xi=%.1f: %d/%d routes, profile: %s"
          % (xi, built, NANTS, np.array2string(curves[xi], precision=1)))

x = (np.arange(NBINS) + 0.5) / NBINS
fig, ax = plt.subplots(figsize=(4.6, 3.1))
styles = {0.0: dict(color="0.15", ls="-", marker="o"),
          0.5: dict(color="tab:blue", ls="-", marker="s"),
          1.0: dict(color="tab:red", ls="--", marker="^")}
for xi in XIS:
    m = curves[xi]
    ok = ~np.isnan(m)
    ax.plot(x[ok], m[ok], ms=4, lw=1.4, label=r"$\xi=%g$" % xi, **styles[xi])
ax.set_xlabel(r"arc-length fraction from $A$ to $B$")
ax.set_ylabel(r"mean turn angle at waypoints [deg]")
ax.set_xlim(0, 1)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, loc="lower left")
fig.savefig("fig_balanced_budget.pdf")
fig.savefig("fig_balanced_budget.png", dpi=300)
print("Saved: fig_balanced_budget.pdf, fig_balanced_budget.png")
