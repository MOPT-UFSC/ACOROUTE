"""
run_campaign.py
Run the multi-start seeds of a run_*.py script in PARALLEL PROCESSES, including on
Windows (no fork needed), then hand the combined result to the script's own
post-processing. The run_*.py scripts are not modified.

    python run_campaign.py run_spline.py                     # seeds and settings from the script
    python run_campaign.py run_staircase.py nproc=4          # at most 4 worker processes
    python run_campaign.py run_sphere.py seeds=3,5,7,11,13   # explicit seeds
    python run_campaign.py run_spline.py every=1             # progress line of every seed at every iteration

Every worker prints its progress to the shared console, each line prefixed by
"[seed N]" (every `every` iterations, default 10), and writes its own log file.

How it works
------------
1. The script is split at its `<name> = A.optimize_multistart(...)` call. The part
   before it (the problem definition: lattice, constraints, objective, auto-scaled ACO
   parameters) is executed here and, independently, inside every worker process. The
   keyword arguments of the call (params, geom_fn, seed_paths, ...) are captured by
   evaluating the call text with optimize_multistart replaced by a recorder.
2. Each worker process runs aco.optimize() for ONE seed with its own route cache and
   its own log file (<log>_seed<s>.txt), exactly as optimize_multistart would do in
   its serial path, and returns the result dict (picklable: lists, dicts, arrays).
3. The parent combines the per-seed results with aco.combine_multistart() (the same
   function optimize_multistart uses), binds the result to the script's variable name
   and executes the rest of the script: geometry report, per-seed table, mesh
   convergence, figures, baseline, reactions.

The worker function is defined at module level and receives only strings and ints,
which is what Python's "spawn" start method requires; the campaign itself runs under
the `if __name__ == "__main__"` guard, also required by spawn. BLAS threading is
pinned to one thread per process before the workers start, so N processes use N
cores instead of N x (BLAS threads).

Results are identical to the serial run of the script with the same SEEDS: the seeds
are independent and every seed keeps its own cache in both cases.
"""
import os
import re
import sys
import io
import time
import contextlib

# one BLAS thread per process: must be set before numpy is imported in the workers
# (they inherit the environment) and is harmless for the parent
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _split_script(script):
    """(head, var_name, call_text, tail): source before the optimize_multistart call,
    the variable it is assigned to, the call expression, and the source after it."""
    src = open(script, encoding="utf-8").read()
    m = re.search(r"^(\w+)\s*=\s*A\.optimize_multistart\(", src, re.M)
    if m is None:
        raise RuntimeError("%s: no '<var> = A.optimize_multistart(' call found" % script)
    i_call = m.start()
    # find the closing parenthesis of the call by balancing brackets
    j = m.end(); depth = 1
    while depth:
        c = src[j]
        if c == "(": depth += 1
        elif c == ")": depth -= 1
        j += 1
    call_text = src[m.end() - len("A.optimize_multistart("):j]
    return src[:i_call], m.group(1), call_text, src[j:]


def _exec_head(script, quiet):
    head, var, call_text, tail = _split_script(script)
    if quiet:
        head = head.replace("SHOW = True", "SHOW = False")
    ns = {"__name__": "campaign_head", "__file__": os.path.abspath(script)}
    out = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(out):
        exec(compile(head, script + "[head]", "exec"), ns)
    # capture the arguments of the optimize_multistart call
    captured = {}
    def _recorder(*args, **kwargs):
        captured["args"] = args; captured["kwargs"] = kwargs
        return None
    A = ns["A"]
    real = A.optimize_multistart
    A.optimize_multistart = _recorder
    try:
        with contextlib.redirect_stdout(out):
            eval(compile(call_text, script + "[call]", "eval"), ns)
    finally:
        A.optimize_multistart = real
    return ns, var, captured, tail


def _seed_log(log_path, seed):
    if log_path is None:
        return None
    root, dot, ext = log_path.rpartition(".")
    return ("%s_seed%d.%s" % (root, seed, ext)) if dot else ("%s_seed%d" % (log_path, seed))


class _Prefixed:
    """stdout wrapper that prefixes every line with the seed tag and flushes at once,
    so the progress lines of concurrent workers stay readable on one console."""
    def __init__(self, stream, tag):
        self.stream, self.tag, self._nl = stream, tag, True
    def write(self, s):
        for piece in s.splitlines(True):
            if self._nl and piece.strip():
                self.stream.write(self.tag)
            self.stream.write(piece)
            self._nl = piece.endswith("\n")
        self.stream.flush()
    def flush(self):
        self.stream.flush()


def _worker(job):
    """One seed in one process. `job` = (script, seed, print_every).
    Returns (seed, result, wall_s)."""
    script, seed, every = job
    t0 = time.time()
    ns, _var, cap, _tail = _exec_head(script, quiet=True)
    A = ns["A"]
    args, kw = cap["args"], dict(cap["kwargs"])
    params = dict(kw.get("params") or {})
    params["n_eval_workers"] = None          # the cores are taken by the seed processes
    params["print_every"] = int(every)
    sys.stdout = _Prefixed(sys.__stdout__, "[seed %d] " % seed)
    try:
        r = A.optimize(*args[:6], params=params, seed=seed, geom_fn=kw.get("geom_fn"),
                       seed_paths=kw.get("seed_paths"), verbose=True, cache=None,
                       log_path=_seed_log(kw.get("log_path"), seed))
    finally:
        sys.stdout = sys.__stdout__
    return seed, r, time.time() - t0


def _save_campaign(script, seeds, results, combined, ns):
    """Persist what the re-ranking and the reproducibility appendix need: every
    distinct route each seed evaluated (lattice node ids) with its search objective,
    the per-seed summaries, the winning route and the search model settings. Written
    as campaign_<script stem>.json.gz next to the script; route_best_ids_<stem>.npy
    holds the winning route alone."""
    import gzip, json
    import numpy as np
    stem = os.path.splitext(os.path.basename(script))[0]
    keys = ("ELEM", "SMOOTH", "HARMONIC_SOLVER", "MODAL_FCUT", "BANDS", "BAND_WEIGHTS",
            "R_MIN", "MAX_TURN", "NODES", "BOX_SIZE", "LMAX_FACTOR", "EXIT_DIR_A",
            "ARRIVE_DIR_B", "EXIT_CONE_DEG", "ARRIVE_CONE_DEG", "TANGENT_CLAMP",
            "EXCLUDE_CENTER", "SELF_CLEARANCE", "RAY", "PRESCRIBED_DISP", "PRESCRIBED_MASK")
    def _py(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [_py(x) for x in v]
        return v
    settings = {k: _py(ns[k]) for k in keys if k in ns}
    settings["Lmax"] = float(ns["Lmax"]); settings["sid"] = int(ns["sid"]); settings["tid"] = int(ns["tid"])
    per_seed = {}
    for s in seeds:
        r = results[s]
        per_seed[str(s)] = {
            "best_power": r["best_power"], "best_length": r["best_length"],
            "best_path": [int(i) for i in r["best_path"]] if r["best_path"] is not None else None,
            "n_fem_calls": int(r["n_fem_calls"]), "n_iter": int(len(r["best_hist"])),
            "n_stagnation_events": int(r["n_stagnation_events"]), "wall_s": float(r.get("wall_s", float("nan"))),
            "best_hist": [float(x) for x in r["best_hist"]],
            "cache": [[[int(i) for i in p], float(J)] for p, J in r["cache"].items()],
        }
    ms = combined["multistart"]
    doc = {"script": os.path.basename(script), "seeds": [int(s) for s in seeds],
           "winning_seed": int(ms["winning_seed"]), "best_power": combined["best_power"],
           "best_path": [int(i) for i in combined["best_path"]],
           "params": {k: _py(v) for k, v in combined["params"].items()
                      if isinstance(v, (int, float, str, bool, list, tuple, type(None), np.ndarray))},
           "settings": settings, "per_seed": per_seed,
           "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    fname = "campaign_%s.json.gz" % stem
    with gzip.open(fname, "wt", encoding="utf-8") as f:
        json.dump(doc, f)
    np.save("route_best_ids_%s.npy" % stem, np.asarray(combined["best_path"], int))
    n_routes = sum(len(v["cache"]) for v in per_seed.values())
    print("campaign: saved %s (%d evaluated routes over %d seeds) and route_best_ids_%s.npy"
          % (fname, n_routes, len(seeds), stem), flush=True)


def _save_template(script, ns):
    """After the script's own post-processing: persist the deterministic template route
    it selected (helix / serpentine / shortest admissible route) so that it can be
    re-evaluated on the reference model by rerank_campaign.py."""
    import json
    import numpy as np
    stem = os.path.splitext(os.path.basename(script))[0]
    pts = ns.get("base_pts")
    if pts is None:
        return
    doc = {"base_label": ns.get("base_label", "baseline"),
           "base_pts": np.asarray(pts, float).tolist(),
           "W_base_search_mesh": (float(ns["W_base"]) if "W_base" in ns else None),
           "base_info": {k: (v.item() if hasattr(v, "item") else v) for k, v in (ns.get("base_info") or {}).items()}}
    with open("template_%s.json" % stem, "w") as f:
        json.dump(doc, f)
    print("campaign: saved template_%s.json (%s)" % (stem, doc["base_label"]), flush=True)



def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    script = sys.argv[1]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    n_proc = int(opts["nproc"]) if "nproc" in opts else None
    every = int(opts.get("every", 10))
    import multiprocessing as mp
    import numpy as np

    print("campaign: %s" % script)
    ns, var, cap, tail = _exec_head(script, quiet=False)
    A = ns["A"]
    seeds = cap["kwargs"].get("seeds", ns.get("SEEDS", [3]))
    if "seeds" in opts:
        seeds = [int(s) for s in opts["seeds"].split(",")]
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    n_proc = min(n_proc or (os.cpu_count() or 1), len(seeds))
    print("campaign: seeds %s in %d parallel processes (start method: spawn)" % (seeds, n_proc), flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_proc) as pool:
        results = {}
        for seed, r, wall in pool.imap_unordered(_worker, [(script, s, every) for s in seeds]):
            results[seed] = r
            p = r["best_power"] if r["best_power"] is not None else float("inf")
            print("  seed %d done: J=%.4e | FEM %d | iters %d | %.0f s"
                  % (seed, p, r["n_fem_calls"], len(r["best_hist"]), wall), flush=True)
    print("campaign: wall time %.0f s for %d seeds" % (time.time() - t0, len(seeds)))

    combined = A.combine_multistart([results[s] for s in seeds], seeds, verbose=True)
    _save_campaign(script, seeds, results, combined, ns)
    ns[var] = combined
    # the rest of the script: report, per-seed table, convergence study, figures, ...
    exec(compile(tail, script + "[tail]", "exec"), ns)
    _save_template(script, ns)


if __name__ == "__main__":
    main()
