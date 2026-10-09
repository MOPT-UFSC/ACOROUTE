"""
validate_slack_balance.py
Validation of the balanced detour budget (params["slack_balance"], see aco.py).

Three domains, chosen to match the reported cases:

  1. empty cube        (like run_spline.py,    Lmax = 1.7x straight)
  2. staircase mask    (like run_staircase.py, Lmax = 3.0x straight, two baffles)
  3. ball + keep-out   (like run_sphere.py,    Lmax = 4.0x and a tight 1.7x)

For each domain and gamma in {0, 0.5, 1} the script builds ants with uniform
pheromone and q0 = 0 (a pure random feasible walk), isolating the construction
bias from the pheromone dynamics, and reports

  * construction success (ants that reach B within budget),
  * mean route length / straight distance,
  * the mean turn angle per fifth of arc length from A to B.

Expected outcome: at gamma = 0 the turn profile collapses toward B (routes lose
the ability to curve near the target); at gamma = 0.5 the profile is roughly
uniform in every domain; obstacle domains keep (or improve) their success rate
because the geodesic bound also prunes pockets that cannot reach B.

Run:  python validate_slack_balance.py [n_ants_per_case]     (default 300;
      the staircase case at gamma = 0 is slow because most walks dead-end)
"""
import sys
import numpy as np
import aco as A
from grid import build_grid, build_neighbours, shortest_path

NTRY = int(sys.argv[1]) if len(sys.argv) > 1 else 300
NBINS = 5


def run_case(name, g, adj, sid, tid, lmax_factor, gammas=(0.0, 0.5, 1.0)):
    coords = g["coords"]
    d0 = float(np.linalg.norm(coords[tid] - coords[sid]))
    Lmax = lmax_factor * d0
    sp = shortest_path(adj, sid, tid)
    Dg = sum(np.linalg.norm(coords[sp[i + 1]] - coords[sp[i]]) for i in range(len(sp) - 1))
    print(f"\n### {name}")
    print(f"    straight A-B {d0:.3f} | graph-shortest {Dg:.3f} ({Dg/d0:.2f}x) | "
          f"Lmax {Lmax:.3f} ({lmax_factor:g}x)")
    for gamma in gammas:
        p = A.default_params()
        p["q0"] = 0.0                    # pure random feasible walk: construction bias only
        p["slack_balance"] = gamma
        rng = np.random.default_rng(1)
        tau = {}
        ok = 0
        len_sum = 0.0
        turn_sum = np.zeros(NBINS)
        turn_cnt = np.zeros(NBINS)
        for _ in range(NTRY):
            path, length = A.build_one_ant(g, adj, tau, sid, tid, Lmax, p, rng)
            if path is None:
                continue
            ok += 1
            len_sum += length
            c = coords[np.asarray(path)]
            seg = np.diff(c, axis=0)
            L = np.linalg.norm(seg, axis=1)
            cum = np.concatenate([[0.0], np.cumsum(L)])
            total = cum[-1]
            for k in range(1, len(seg)):
                u = seg[k - 1] / L[k - 1]
                v = seg[k] / L[k]
                ang = np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
                b = min(NBINS - 1, int(NBINS * cum[k] / total))
                turn_sum[b] += ang
                turn_cnt[b] += 1
        prof = "  ".join(f"{turn_sum[b]/turn_cnt[b]:5.1f}" if turn_cnt[b] else "  n/a"
                         for b in range(NBINS))
        ml = len_sum / ok / d0 if ok else float("nan")
        print(f"    gamma={gamma:3.1f}: success {ok:4d}/{NTRY}  mean len/d0 {ml:5.2f}  "
              f"turn angle per fifth A->B [deg]: {prof}")


# --- 1. empty cube (run_spline-like) -----------------------------------------
g = build_grid(7, 7, 7, 0.1)
adj = build_neighbours(g, 26)
run_case("empty cube, Lmax = 1.7x",
         g, adj, g["id_of"][(0, 0, 0)], g["id_of"][(6, 6, 6)], 1.7)

# --- 2. staircase: two alternating baffles (run_staircase-like) --------------
LO_IX = set(range(0, 4))     # lower baffle attached to the left wall, gap right
UP_IX = set(range(3, 7))     # upper baffle attached to the right wall, gap left


def mask_stair(ix, iy, iz):
    if iz == 2 and ix in LO_IX:
        return False
    if iz == 4 and ix in UP_IX:
        return False
    return True


g = build_grid(7, 7, 7, 0.1, mask=mask_stair)
adj = build_neighbours(g, 26)
run_case("staircase (two baffles), Lmax = 3x",
         g, adj, g["id_of"][(0, 0, 0)], g["id_of"][(6, 6, 6)], 3.0)

# --- 3. ball domain with a central keep-out box (run_sphere-like) ------------
ctr = np.array([0.4, 0.4, 0.4])
R = 0.4
lo, hi = np.array([0.25] * 3), np.array([0.55] * 3)


def mask_ball(ix, iy, iz):
    x = np.array([ix, iy, iz]) * 0.1
    if np.linalg.norm(x - ctr) > R + 1e-12:
        return False
    if np.all(x > lo + 1e-12) and np.all(x < hi - 1e-12):
        return False
    return True


g = build_grid(9, 9, 9, 0.1, mask=mask_ball)
adj = build_neighbours(g, 26)
sid, tid = g["id_of"][(2, 2, 2)], g["id_of"][(6, 6, 6)]
run_case("ball + central keep-out, Lmax = 4x", g, adj, sid, tid, 4.0)
run_case("ball + central keep-out, Lmax = 1.7x (tight)", g, adj, sid, tid, 1.7)
