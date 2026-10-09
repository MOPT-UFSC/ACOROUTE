"""Size of the admissible route search space, for comparison with the number of
objective evaluations the optimizer actually performs.

A natural objection to a population search on a cheap-to-evaluate model is that it
might be enumerating the whole space and keeping the best. To answer it we count the
admissible routes the search ranges over and compare with the evaluations spent.

The count enumerates, by depth-first search, every simple path from the source to the
target that satisfies the same construction-time admissibility as the ant operator of
build_one_ant: a node is never re-entered (simple path), the length already used plus
the next edge plus the straight-line distance to the target stays within the budget
(the A* lower bound), and, when a turn-angle limit is given, the angle between
consecutive segments stays within it (the manufacturability pre-filter that bounds the
bend radius). Counting simple paths between two nodes is #P-hard in general, so a cap
and a time limit are provided; when they are reached the returned count is a lower
bound on the true size, which is enough to establish that the evaluations are a small
fraction of the space.

Note on the balanced detour budget: the admissibility used here is the CLASSIC
length test (slack_balance = 0), so the count measures the feasible design space
that an exhaustive evaluation would need to enumerate. With slack_balance > 0 the
rationed construction can produce only a subset of these routes (the rule excludes
front-loaded detours), so this count is NOT a lower bound on the set of routes the
balanced constructor can generate -- it bounds the space to be searched, not the
sampler.

Run as a script, it reports the size for the configuration of make_paper_figures.
"""
import sys
import time
import numpy as np


def count_admissible_routes(grid, adj, start_id, target_id, Lmax,
                            max_turn_deg=None, cap=60_000_000, time_limit=120.0):
    """Number of admissible simple paths start->target.

    grid, adj, start_id, target_id, Lmax and max_turn_deg are the same objects the
    optimizer uses. Returns (count, capped): if capped is True the search hit the cap
    or the time limit and `count` is a lower bound on the true number.
    """
    coords = grid["coords"]
    n = len(coords)
    nbrs = [[] for _ in range(n)]
    for i in range(n):
        for (j, seg) in adj[i]:
            d = coords[j] - coords[i]
            nbrs[i].append((j, float(seg), d / (np.linalg.norm(d) + 1e-15)))
    dist_t = np.linalg.norm(coords - coords[target_id], axis=1)
    cos_max = None if max_turn_deg is None else float(np.cos(np.radians(max_turn_deg)))

    sys.setrecursionlimit(100000)
    visited = np.zeros(n, bool)
    t_end = time.time() + time_limit
    state = {"count": 0, "capped": False}

    def dfs(cur, length, prev_dir):
        if cur == target_id:
            state["count"] += 1
            if state["count"] >= cap or time.time() > t_end:
                state["capped"] = True
            return
        for (j, seg, ud) in nbrs[cur]:
            if visited[j]:
                continue
            if length + seg + dist_t[j] > Lmax + 1e-9:
                continue
            if cos_max is not None and prev_dir is not None and float(prev_dir @ ud) < cos_max:
                continue
            visited[j] = True
            dfs(j, length + seg, ud)
            visited[j] = False
            if state["capped"]:
                return

    visited[start_id] = True
    dfs(start_id, 0.0, None)
    return state["count"], state["capped"]


def _main():
    import make_paper_figures as M
    coords = M.grid["coords"]
    span = float(np.linalg.norm(coords[M.TID] - coords[M.SID]))
    print("configuration: %dx%dx%d lattice, 26-connectivity, "
          "Lmax/|B-A| = %.2f, turn <= %g deg"
          % (M.NODES[0], M.NODES[1], M.NODES[2], M.LMAX / span, M.MAX_TURN))
    for label, mt in [("length budget only", None),
                      ("length budget + turn limit (search space)", M.MAX_TURN)]:
        t0 = time.time()
        cnt, capped = count_admissible_routes(M.grid, M.adj, M.SID, M.TID, M.LMAX,
                                              max_turn_deg=mt, time_limit=75.0)
        rel = ">=" if capped else "="
        print("  admissible routes %s %s %s  (%.1fs, %s)"
              % (label, rel, format(cnt, ","), time.time() - t0,
                 "lower bound" if capped else "exact"))


if __name__ == "__main__":
    _main()
