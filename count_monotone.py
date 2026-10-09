"""
count_monotone.py
Search-space figures of the paper (Section 'Search-space size and random-search
benchmark') for the case defined in a run_*.py script, without any finite-element
analysis:

    python count_monotone.py run_spline.py [n_sample]

1. Exact count, by dynamic programming over (node, type of the arriving step), of the
   MONOTONE lattice routes A -> B (the seven forward increments only) that satisfy the
   construction-level filters of the script: retained nodes (keep-out), turn-angle
   pre-filter and exit/arrival cones. Reference counts without the filters are printed
   as well. Requires A and B at opposite corners of a cubic lattice (monotone = every
   index non-decreasing from A to B).
2. Uniform sample of n_sample routes from that set (the DP counts are the sampling
   weights at every step), passed through the polyline length limit and the
   spline-level checks of the script (geom_fn: developed length, R_min, keep-out,
   containment, with the end-tangent clamp when active). Reports the admissible
   fraction with its Wilson interval, the resulting estimate of admissible monotone
   routes, the length/radius range of the admissible samples and the fraction of that
   estimate covered by a given number of FEM evaluations.
"""
import sys, io, contextlib, math, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCRIPT = sys.argv[1] if len(sys.argv) > 1 else "run_spline.py"
N_SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

src = open(SCRIPT, encoding="utf-8").read()
i = src.index("A.optimize_multistart"); head = src[:src.rfind("\n", 0, i)]
ns = {"__name__": "mono_head"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, SCRIPT + "[head]", "exec"), ns)
grid, Lmax, geo, geom = ns["grid"], ns["Lmax"], ns["geo"], ns["geom_fn"]
NODES, SPACING, R_MIN, MAX_TURN = ns["NODES"], ns["SPACING"], ns["R_MIN"], ns["MAX_TURN"]
EXIT, ARR, CONE_A, CONE_B = ns["EXIT_DIR_A"], ns["ARRIVE_DIR_B"], ns["EXIT_CONE_DEG"], ns["ARRIVE_CONE_DEG"]
if ns["TANGENT_CLAMP"] and (EXIT is not None or ARR is not None):
    geo.END_TANGENTS = (EXIT, ARR)
h = SPACING[0]; m = NODES[0] - 1
id_of = grid["id_of"]
print("lattice %d^3 | pitch %.2f mm | Lmax %.1f mm | theta_max %.2f deg | cones %.0f/%.0f deg | retained nodes %d"
      % (NODES[0], h * 1e3, Lmax * 1e3, MAX_TURN, CONE_A, CONE_B, len(grid["coords"])))

STEPS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
U = [np.array(d, float) / np.linalg.norm(d) for d in STEPS]
cos_turn = math.cos(math.radians(MAX_TURN)) - 1e-9
cosA = math.cos(math.radians(CONE_A)) - 1e-9; cosB = math.cos(math.radians(CONE_B)) - 1e-9
uA = np.asarray(EXIT, float) / np.linalg.norm(EXIT) if EXIT is not None else None
uB = np.asarray(ARR, float) / np.linalg.norm(ARR) if ARR is not None else None
turn_ok = [[float(U[a] @ U[b]) >= cos_turn for b in range(7)] for a in range(7)]
first_ok = [uA is None or float(U[b] @ uA) >= cosA for b in range(7)]
last_ok = [uB is None or float(U[b] @ uB) >= cosB for b in range(7)]
B = (m, m, m)

# completions C[(node, last_type)] = number of valid monotone continuations to B
C = {}
nodes_sorted = sorted(((ix, iy, iz) for ix in range(m + 1) for iy in range(m + 1) for iz in range(m + 1)),
                      key=lambda t: -(t[0] + t[1] + t[2]))
t0 = time.perf_counter()
for v in nodes_sorted:
    if v not in id_of:
        continue
    for a in range(7):              # type of the step that arrived at v (or -1 at A handled below)
        if v == B:
            C[(v, a)] = 1 if last_ok[a] else 0
            continue
        tot = 0
        for b, d in enumerate(STEPS):
            if not turn_ok[a][b]:
                continue
            w = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
            if w[0] > m or w[1] > m or w[2] > m or w not in id_of:
                continue
            tot += C.get((w, b), 0)
        C[(v, a)] = tot
# from A: first step filtered by the exit cone, no turn test
A0 = (0, 0, 0)
first = {}
for b, d in enumerate(STEPS):
    if not first_ok[b]:
        continue
    w = (d[0], d[1], d[2])
    if w in id_of:
        first[b] = C.get((w, b), 0)
N_filt = sum(first.values())
print("DP done in %.1f s | N_mono(construction filters) = %d = %.3e" % (time.perf_counter() - t0, N_filt, N_filt))

# reference counts without filters, for the text
def count_plain(turn=False, cones=False, keepout=False):
    Cp = {}
    for v in nodes_sorted:
        if keepout and v not in id_of:
            continue
        for a in range(7):
            if v == B:
                Cp[(v, a)] = 1 if (not cones or last_ok[a]) else 0; continue
            tot = 0
            for b, d in enumerate(STEPS):
                if turn and not turn_ok[a][b]:
                    continue
                w = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
                if w[0] > m or w[1] > m or w[2] > m or (keepout and w not in id_of):
                    continue
                tot += Cp.get((w, b), 0)
            Cp[(v, a)] = tot
    return sum(Cp.get(((d[0], d[1], d[2]), b), 0) for b, d in enumerate(STEPS) if (not cones or first_ok[b]))
print("  all monotone routes                : %.3e" % count_plain())
print("  + keep-out nodes                   : %.3e" % count_plain(keepout=True))
print("  + keep-out + turn filter           : %.3e" % count_plain(keepout=True, turn=True))
print("  + keep-out + turn + end cones      : %.3e" % count_plain(keepout=True, turn=True, cones=True))

# uniform sampling from the filtered set via the completion counts
rng = np.random.default_rng(2026)
def sample():
    ks = list(first); ws = np.array([first[k] for k in ks], float); ws /= ws.sum()
    b = ks[rng.choice(len(ks), p=ws)]
    v = STEPS[b]; path = [A0, v]; a = b
    while v != B:
        opts, ws = [], []
        for b2, d in enumerate(STEPS):
            if not turn_ok[a][b2]:
                continue
            w = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
            c = C.get((w, b2), 0)
            if c:
                opts.append((w, b2)); ws.append(c)
        ws = np.array(ws, float); ws /= ws.sum()
        w, b2 = opts[rng.choice(len(opts), p=ws)]
        path.append(w); v, a = w, b2
    return path

stats = dict(poly_len=0, spline_len=0, spline_R=0, spline_keepout_contain=0, ok=0)
Ls, Rs = [], []
t0 = time.perf_counter()
for s in range(N_SAMPLE):
    idx = sample()
    coords = np.array([grid["coords"][id_of[t]] for t in idx])
    Lpoly = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
    if Lpoly > Lmax + 1e-9:
        stats["poly_len"] += 1; continue
    L, R, ok = geom(coords)
    if L > Lmax:
        stats["spline_len"] += 1; continue
    if R < R_MIN:
        stats["spline_R"] += 1; continue
    if not ok:
        stats["spline_keepout_contain"] += 1; continue
    stats["ok"] += 1; Ls.append(L); Rs.append(R)
print("sampled %d filtered monotone routes in %.1f s" % (N_SAMPLE, time.perf_counter() - t0))
for k, v in stats.items():
    print("  %-22s %6d  (%.4f)" % (k, v, v / N_SAMPLE))
p = stats["ok"] / N_SAMPLE
z = 1.959964; den = 1 + z * z / N_SAMPLE
c = (p + z * z / (2 * N_SAMPLE)) / den
hw = z * math.sqrt(p * (1 - p) / N_SAMPLE + z * z / (4 * N_SAMPLE * N_SAMPLE)) / den
print("p_feas = %.4f  Wilson 95%% [%.4f, %.4f]" % (p, c - hw, c + hw))
print("N_mono,feas ~ %.3e  [%.3e, %.3e]" % (p * N_filt, (c - hw) * N_filt, (c + hw) * N_filt))
if Ls:
    print("admissible samples: L %.0f-%.0f mm (mean %.0f), R %.1f-%.1f mm (mean %.1f)"
          % (min(Ls) * 1e3, max(Ls) * 1e3, np.mean(Ls) * 1e3, min(Rs) * 1e3, max(Rs) * 1e3, np.mean(Rs) * 1e3))
for nfem in (5000, 25000):
    print("  %d distinct evaluations = %.2e of the feasible estimate (scale to the campaign's totals)" % (nfem, nfem / (p * N_filt) if p > 0 else float("nan")))
print("exhaustive at 0.1 s/route: %.2e years (feasible monotone estimate)" % (p * N_filt * 0.1 / 3.15576e7))
