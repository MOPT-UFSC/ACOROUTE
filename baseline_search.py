"""
baseline_search.py
Search-level controls for a run_*.py case, at the SAME finite-element budget as the
ACO campaign, so that the contribution of the pheromone-guided search can be separated
from (a) the route constructor plus selection and (b) plain local search.

    python baseline_search.py run_spline.py                 # budgets read from the ACO logs
    python baseline_search.py run_spline.py budget=6800     # same budget for every run
    python baseline_search.py run_staircase.py seeds=3,5 nproc=2 controls=random

Controls
--------
random : the ACO route constructor with the pheromone branch switched off (q0 = 0):
         admissibility filters, balanced detour budget, bidirectional construction,
         prescribed exit/arrival cones and the spline-level checks are all kept; every
         admissible route is evaluated and the best one is retained. This is "the
         constructor plus selection": if it matches the ACO, the pheromone adds nothing.
hill   : random-restart hill-climber on the lattice. Start from the first admissible
         constructed route; repeat: apply one random local mutation to the incumbent
         (replace an interior node by a common neighbour of its two neighbours, insert
         a detour node, or remove a node whose neighbours are adjacent), keep the mutant
         if it is admissible and lowers J; when every mutant of the incumbent is
         inadmissible or already evaluated, restart from a new constructed route and
         keep the global best. This is "local refinement without a
         population or memory": if it matches the ACO, the colony adds nothing beyond
         local search around the first basin found.

Budget and protocol
-------------------
One control run per ACO seed, each with the finite-element budget that the ACO seed
spent (read from <log>_seed<s>.txt: "distinct routes FEM-evaluated=N"), so the
per-seed comparison is at equal cost and the best-of-seeds comparison is at equal
total cost. `budget=N` overrides with one value for all runs. Every run holds its own
route cache (a repeated route is free, as in the ACO), and the budget counts distinct
FEM evaluations, exactly as n_fem_calls does in aco.optimize.

Outputs: baseline_search_<case>.csv (one row per run), baseline_search_<case>.png
(best-so-far J versus FEM evaluations: ACO seeds from the logs, controls from the
runs), and a console table that can be pasted next to the per-seed ACO table.
The run_*.py scripts are not modified; the problem definition is executed from the
script exactly as run_campaign.py does. Runs are independent processes (spawn).
"""
import os
import re
import sys
import time
import glob
import numpy as np

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import run_campaign as RC          # _exec_head / _seed_log: same problem loading as the campaign


# --------------------------------------------------------------------------- helpers
def _neighbour_sets(adj):
    return {i: set(j for j, _L in nb) for i, nb in adj.items()}


def _route_ok(path, grid, geom_fn, Lmax, p):
    """Spline-level admissibility of a lattice route (same checks as aco.optimize)."""
    from grid import path_coords
    coords = path_coords(grid, path)
    g = geom_fn(coords)
    slen, mrad = g[0], g[1]
    kok = g[2] if len(g) > 2 else True
    ok = slen <= Lmax + 1e-9 and (p.get("R_min", 0.0) <= 0.0 or mrad >= p["R_min"]) and kok
    return ok, coords, slen


def _construct(A, grid, adj, sid, tid, Lmax, p, rng, dist_t, dist_s, p_rev, n_rev_frac):
    """One route from the ACO constructor with q0 = 0 (uniform branch only), honouring
    the bidirectional fraction and the prescribed cones exactly as optimize() does."""
    if rng.random() < n_rev_frac:
        path, L = A.build_one_ant(grid, adj, {}, tid, sid, Lmax, p_rev, rng, q0=0.0, dist_target=dist_s)
        return (path[::-1] if path is not None else None), L
    return A.build_one_ant(grid, adj, {}, sid, tid, Lmax, p, rng, q0=0.0, dist_target=dist_t)


def _turn_ok(a, b, c, coords, cos_max):
    """Turn-angle pre-filter at node b between segments a->b and b->c."""
    if cos_max is None:
        return True
    u = coords[b] - coords[a]; v = coords[c] - coords[b]
    u = u / (np.linalg.norm(u) + 1e-15); v = v / (np.linalg.norm(v) + 1e-15)
    return float(np.dot(u, v)) >= cos_max - 1e-9


def _mutate(path, nb, coords, rng, cos_max, lock_ends):
    """One random local mutation of a lattice route; returns a new simple path or None.
    lock_ends: number of nodes at each end that must not change (prescribed cones)."""
    n = len(path)
    lo, hi = 1 + lock_ends, n - 2 - lock_ends          # mutable interior indices [lo, hi]
    if hi < lo:
        return None
    inpath = set(path)
    kind = rng.integers(3)
    i = int(rng.integers(lo, hi + 1))
    a, b, c = path[i - 1], path[i], path[i + 1]
    if kind == 0:
        # replace b by another common neighbour of a and c
        cands = [j for j in (nb[a] & nb[c]) if j not in inpath]
        if not cands:
            return None
        j = cands[int(rng.integers(len(cands)))]
        new = path[:i] + [j] + path[i + 1:]
        checks = [(path[i - 2], a, j)] if i >= 2 else []
        checks += [(a, j, c)] + ([(j, c, path[i + 2])] if i + 2 < n else [])
    elif kind == 1:
        # insert a detour: a -> j -> k -> c replacing a -> b -> c  (j, k new, adjacent)
        cj = [j for j in nb[a] if j not in inpath]
        if not cj:
            return None
        j = cj[int(rng.integers(len(cj)))]
        ck = [k for k in (nb[j] & nb[c]) if k not in inpath and k != j]
        if not ck:
            return None
        k = ck[int(rng.integers(len(ck)))]
        new = path[:i] + [j, k] + path[i + 1:]
        checks = ([(path[i - 2], a, j)] if i >= 2 else []) + [(a, j, k), (j, k, c)] + ([(k, c, path[i + 2])] if i + 2 < n else [])
    else:
        # remove b when a and c are adjacent
        if c not in nb[a]:
            return None
        new = path[:i] + path[i + 1:]
        checks = ([(path[i - 2], a, c)] if i >= 2 else []) + ([(a, c, path[i + 2])] if i + 2 < n else [])
    for t in checks:
        if not _turn_ok(t[0], t[1], t[2], coords, cos_max):
            return None
    return new


# --------------------------------------------------------------------------- one control run
def _run_control(job):
    """(script, control, seed, budget) -> dict. Runs in its own process."""
    script, control, seed, budget = job
    t0 = time.time()
    ns, _var, cap, _tail = RC._exec_head(script, quiet=True)
    A = ns["A"]
    args, kw = cap["args"], dict(cap["kwargs"])
    grid, adj, sid, tid, Lmax, objective_fn = args[:6]
    geom_fn = kw.get("geom_fn")
    p = A.default_params(); p.update(kw.get("params") or {})
    from grid import distances_from
    rng = np.random.default_rng(seed)
    # same construction aids as optimize(): geodesic fields, reversed-ant parameters
    dist_t = distances_from(adj, tid) if p.get("slack_balance", 0.0) > 0.0 else None
    dist_s = None
    n_rev_frac = float(p.get("bidirectional", 0.0))
    p_rev = dict(p)
    if n_rev_frac > 0.0:
        sd, ed = p.get("start_dir"), p.get("end_dir")
        p_rev["start_dir"] = (-np.asarray(ed, float)) if ed is not None else None
        p_rev["end_dir"] = (-np.asarray(sd, float)) if sd is not None else None
        p_rev["start_max_turn_deg"] = p.get("end_max_turn_deg")
        p_rev["end_max_turn_deg"] = p.get("start_max_turn_deg")
        if p.get("slack_balance", 0.0) > 0.0:
            dist_s = distances_from(adj, sid)
    cache = {}
    best = None; best_hist = []; fem_hist = []
    n_fem = 0; n_tried = 0; n_adm = 0

    def evaluate(path):
        nonlocal n_fem, best, n_adm
        key = tuple(path)
        if key in cache:
            return cache[key]
        ok, coords, slen = _route_ok(path, grid, geom_fn, Lmax, p)
        if not ok:
            cache[key] = None          # remembered as inadmissible: never re-checked
            return None
        n_adm += 1
        J = float(objective_fn(coords)); cache[key] = J; n_fem += 1
        if best is None or J < best["J"]:
            best = {"J": J, "path": list(path), "length": slen}
        best_hist.append(best["J"]); fem_hist.append(n_fem)
        return J

    if control == "random":
        while n_fem < budget:
            path, _L = _construct(A, grid, adj, sid, tid, Lmax, p, rng, dist_t, dist_s, p_rev, n_rev_frac)
            n_tried += 1
            if path is not None:
                evaluate(path)
    elif control == "hill":
        nb = _neighbour_sets(adj); coords = grid["coords"]
        mt = p.get("max_turn_deg"); cos_max = np.cos(np.radians(mt)) if mt is not None else None
        lock = 1 if (p.get("start_dir") is not None or p.get("end_dir") is not None) else 0
        # first admissible constructed route = starting incumbent
        while best is None and n_tried < 100000:
            path, _L = _construct(A, grid, adj, sid, tid, Lmax, p, rng, dist_t, dist_s, p_rev, n_rev_frac)
            n_tried += 1
            if path is not None:
                evaluate(path)
        # climb from the incumbent of the current start; when its mutation neighbourhood is
        # exhausted (every mutant inadmissible or already evaluated) restart from a new
        # constructed route and keep climbing, retaining the global best: random-restart
        # hill climbing. `cur` is the local incumbent, `best` the global one.
        cur = dict(best); stall = 0; n_restart = 0; n_bad_local = 0
        MAX_STALL, MAX_BAD = 20000, 300     # exhaustion: repeats/invalid mutants, inadmissible mutants
        while n_fem < budget:
            new = _mutate(cur["path"], nb, coords, rng, cos_max, lock)
            n_tried += 1
            exhausted = False
            if new is None or tuple(new) in cache:
                stall += 1
                exhausted = stall >= MAX_STALL
            else:
                stall = 0
                J = evaluate(new)              # updates the global best when better
                if J is None:
                    n_bad_local += 1           # geometrically inadmissible mutant (no FEM spent)
                    exhausted = n_bad_local >= MAX_BAD
                elif J < cur["J"]:
                    cur = {"J": J, "path": list(new)}; n_bad_local = 0
            if exhausted:                      # random restart from a new constructed route
                n_restart += 1; stall = 0; n_bad_local = 0
                while n_fem < budget:
                    path, _L = _construct(A, grid, adj, sid, tid, Lmax, p, rng, dist_t, dist_s, p_rev, n_rev_frac)
                    n_tried += 1
                    if path is not None and tuple(path) not in cache and evaluate(path) is not None:
                        cur = {"J": cache[tuple(path)], "path": list(path)}; break
    else:
        raise ValueError("unknown control %r" % control)

    return {"control": control, "seed": seed, "budget": int(budget),
            "J": (best["J"] if best else None), "path": (best["path"] if best else None),
            "length": (best["length"] if best else None), "n_fem": n_fem,
            "n_tried": n_tried, "n_admissible": n_adm,
            "n_restart": (n_restart if control == "hill" else 0),
            "best_hist": np.asarray(best_hist), "fem_hist": np.asarray(fem_hist),
            "wall_s": time.time() - t0}


# --------------------------------------------------------------------------- ACO logs
def _parse_aco_logs(log_path, seeds):
    """Per-seed (final J, FEM, cumulative-FEM curve, best-J curve) from the campaign logs."""
    out = {}
    for s in seeds:
        f = RC._seed_log(log_path, s) if log_path else None
        if f is None or not os.path.exists(f):
            continue
        fem = []; bestJ = []; final = None; nfem = None
        for line in open(f, encoding="utf-8"):
            if line.startswith("#"):
                m = re.search(r"final best J=([0-9.eE+-]+).*?FEM-evaluated=(\d+)", line)
                if m:
                    final, nfem = float(m.group(1)), int(m.group(2))
                continue
            cols = line.split()
            if len(cols) >= 12:
                fem.append(int(cols[7])); bestJ.append(float(cols[10]))
        if fem:
            out[s] = {"J": final, "n_fem": nfem, "fem_hist": np.cumsum(fem), "best_hist": np.asarray(bestJ)}
    return out


# --------------------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    script = sys.argv[1]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    controls = opts.get("controls", "random,hill").split(",")
    import multiprocessing as mp

    ns, _var, cap, _tail = RC._exec_head(script, quiet=True)
    kw = cap["kwargs"]
    seeds = kw.get("seeds", ns.get("SEEDS", [3]))
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    if "seeds" in opts:
        seeds = [int(s) for s in opts["seeds"].split(",")]
    aco = _parse_aco_logs(kw.get("log_path"), seeds)
    if "budget" in opts:
        budgets = {s: int(opts["budget"]) for s in seeds}
    else:
        missing = [s for s in seeds if s not in aco]
        if missing:
            sys.exit("no ACO log for seeds %s (expected %s); run the campaign first or pass budget=N"
                     % (missing, RC._seed_log(kw.get("log_path"), seeds[0])))
        budgets = {s: aco[s]["n_fem"] for s in seeds}
    case = os.path.splitext(os.path.basename(script))[0]
    n_proc = min(int(opts.get("nproc", os.cpu_count() or 1)), len(seeds) * len(controls))
    print("baseline_search: %s | controls %s | seeds %s | budgets (FEM) %s | %d processes"
          % (case, controls, seeds, [budgets[s] for s in seeds], n_proc), flush=True)

    jobs = [(script, c, s, budgets[s]) for c in controls for s in seeds]
    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=n_proc) as pool:
        results = pool.map(_run_control, jobs)
    print("baseline_search: wall time %.0f s" % (time.time() - t0))

    # ------------------------------------------------------------- table
    print("\n  %-7s %5s %8s %14s %14s %8s %9s %8s %7s" % ("control", "seed", "FEM", "J_control", "J_ACO", "ratio", "admiss.", "restarts", "time[s]"))
    rows = []
    for r in results:
        Ja = aco.get(r["seed"], {}).get("J")
        ratio = (r["J"] / Ja) if (r["J"] is not None and Ja) else float("nan")
        print("  %-7s %5d %8d %14s %14s %8.3f %9d %8d %7.0f"
              % (r["control"], r["seed"], r["n_fem"], ("%.4e" % r["J"]) if r["J"] is not None else "none",
                 ("%.4e" % Ja) if Ja else "n/a", ratio, r["n_admissible"], r["n_restart"], r["wall_s"]))
        rows.append((r["control"], r["seed"], r["n_fem"], r["J"], Ja, ratio, r["n_admissible"], r["n_tried"], r["wall_s"]))
    print("\n  best of seeds (equal total budget):")
    if aco:
        Ja_best = min(v["J"] for v in aco.values() if v["J"] is not None)
        print("    %-7s J = %.4e" % ("ACO", Ja_best))
    for c in controls:
        Js = [r["J"] for r in results if r["control"] == c and r["J"] is not None]
        if Js:
            line = "    %-7s J = %.4e" % (c, min(Js))
            if aco:
                line += "   (ratio to ACO %.2f; >1 means the ACO is better)" % (min(Js) / Ja_best)
            print(line)

    with open("baseline_search_%s.csv" % case, "w") as f:
        f.write("control,seed,fem,J_control,J_aco,ratio_control_over_aco,n_admissible,n_tried,wall_s\n")
        for row in rows:
            f.write(",".join("" if v is None else ("%.6e" % v if isinstance(v, float) else str(v)) for v in row) + "\n")

    # ------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import plot_style  # noqa: F401
    except Exception:
        pass
    COL = {"ACO": "#b3122c", "random": "#7f8c8d", "hill": "#2a4d69"}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (s, v) in enumerate(sorted(aco.items())):
        ax.step(v["fem_hist"], v["best_hist"], where="post", color=COL["ACO"], lw=1.6, alpha=0.85,
                label="ACO (seeds %s)" % seeds if i == 0 else None)
    for c in controls:
        first = True
        for r in results:
            if r["control"] == c and len(r["fem_hist"]):
                ax.step(r["fem_hist"], r["best_hist"], where="post", color=COL.get(c, "0.4"), lw=1.4,
                        alpha=0.85, label=("%s control" % c) if first else None)
                first = False
    ax.set_yscale("log"); ax.set_xlabel("finite-element evaluations"); ax.set_ylabel(r"best-so-far objective  [W$\cdot$Hz]")
    # no in-figure title: the manuscript caption describes the panel
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("baseline_search_%s.png" % case, dpi=150); fig.savefig("baseline_search_%s.pdf" % case)
    print("\nSaved: baseline_search_%s.csv, baseline_search_%s.png/.pdf" % (case, case))


if __name__ == "__main__":
    main()
