"""
validate_fixes.py
Regression tests for the issues raised in the code audit. Each test fails loudly, so this
script can be run after any change to the numerics:

    python validate_fixes.py

Covered:
  1. frequency quadrature: the band integral must track a converged reference in EVERY
     band, not only the high one it was calibrated on (this was the finding that could
     invert the ranking of two routes);
  2. spline endpoints: the curve must start at A and end at B for any smoothing value;
  3. Cowper shear coefficient: the solid-circle limit must return the tabulated value;
  4. element size: the requested size is a MAXIMUM;
  5. straight two-waypoint route: infinite bend radius, not a crash;
  6. evaporation: every admissible edge evaporates, not only the ones already deposited;
  7. warm start: the BEST feasible corridor becomes the incumbent, not the first one;
  8. input power: a passive structure delivers non-negative power (no silent abs()).
"""

import numpy as np

import aco as A
import geometry as geo
import grid as G
import fem_beam3d as fem
import baseline_routes as BL

# physical configuration of the paper
E, nu, rho = 210e9, 0.3, 7850.0
MAT = {"E": E, "G": E / (2.0 * (1.0 + nu)), "rho": rho}
SEC = fem.pipe_section(outer_radius=0.001575, thickness=0.0005, nu=nu)
RAY = (0.0, 2e-7)
PD = (1e-6, 1e-6, 1e-6)
ELEM = 0.005
BOX = np.array([0.300, 0.300, 0.300])

_fails = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    if not ok:
        _fails.append(name)


def _mesh(turns, radius):
    pts = BL.helix_points(np.zeros(3), BOX, turns, radius, n=400, taper=True)
    return geo.resample_by_length(geo.fit_spline(pts, smooth=0.0), ELEM)


def _spectrum(nodes, elems, freqs):
    return np.abs(np.asarray(fem.input_power_spectrum_prescribed(
        nodes, elems, MAT, SEC, 0, len(nodes) - 1, freqs,
        prescribed_disp=PD, rayleigh=RAY)))


# --------------------------------------------------------------------------- #
def test_quadrature(tol_pct=8.0, spread_pct=3.0):
    """The adaptive grid must integrate the band to a few percent in EVERY band, and,
    more importantly, the residual error must be nearly the same for every route: it is
    the SPREAD of the error across routes, not its mean, that reorders two candidates."""
    print("\n1) frequency quadrature vs a converged reference")
    for band in [(1000.0, 3000.0), (6000.0, 8000.0)]:
        errs = []
        for turns, radius in [(2, 0.075), (1.2, 0.112), (3, 0.060)]:
            nodes, elems, _ = _mesh(turns, radius)
            f_ref = np.arange(band[0], band[1] + 0.25, 0.25)     # converged reference
            J_ref = np.trapezoid(_spectrum(nodes, elems, f_ref), f_ref)
            peaks = fem.natural_frequencies(nodes, elems, MAT, SEC, len(nodes) - 1, band,
                                            extra_fixed_dofs=[0, 1, 2])
            f_ad = fem._adaptive_freq_grid(band, peaks, RAY)
            J_ad = np.trapezoid(_spectrum(nodes, elems, f_ad), f_ad)
            errs.append(100.0 * (J_ad - J_ref) / J_ref)
        errs = np.array(errs)
        check("band %d-%d Hz: |err| <= %.0f%%" % (band[0], band[1], tol_pct),
              np.abs(errs).max() <= tol_pct,
              "errors %s" % np.array2string(errs, precision=1))
        check("band %d-%d Hz: error spread <= %.0f%% (rank fidelity)"
              % (band[0], band[1], spread_pct),
              (errs.max() - errs.min()) <= spread_pct,
              "spread %.1f%%" % (errs.max() - errs.min()))


def test_endpoints():
    """A smoothing spline must still start at A and end at B: they are the physical
    connections, not design freedoms."""
    print("\n2) spline endpoints pinned for any smoothing value")
    g = G.build_grid(6, 6, 6, (0.06, 0.06, 0.06))
    c = g["coords"]
    adj = {i: list(v) for i, v in G.build_neighbours(g, connectivity=26).items()}
    sid, tid = g["id_of"][(0, 0, 0)], g["id_of"][(5, 5, 5)]
    Lmax = 1.7 * float(np.linalg.norm(c[tid] - c[sid]))
    p = {**A.default_params(), "max_turn_deg": 90.0, "max_ops": 4000}
    rng = np.random.default_rng(7)
    paths = []
    while len(paths) < 6:
        path, _ = A.build_one_ant(g, adj, {}, sid, tid, Lmax, p, rng, q0=0.3)
        if path is not None:
            paths.append(path)
    for s in (0.0, 5e-4, 1e-3, 5e-3):
        d = 0.0
        for path in paths:
            w = c[np.asarray(path)]
            sp = geo.eval_spline(geo.fit_spline(w, smooth=s), 400)
            d = max(d, np.linalg.norm(sp[0] - w[0]), np.linalg.norm(sp[-1] - w[-1]))
        check("smooth=%.0e: endpoint drift < 1 um" % s, d < 1e-6, "max %.2e m" % d)


def test_cowper():
    print("\n3) Cowper shear coefficient")
    k_solid = fem.pipe_section(0.01, 0.01, 0.3)["kappa"]      # thickness = radius -> solid
    check("solid circle, nu=0.3 -> 0.886", abs(k_solid - 0.8864) < 2e-3,
          "kappa = %.4f" % k_solid)


def test_element_size():
    print("\n4) requested element size is a maximum")
    tck = geo.fit_spline(BL.helix_points(np.zeros(3), BOX, 2, 0.075, n=300, taper=True))
    for h in (0.005, 0.010, 0.020):
        nodes, _e, _L = geo.resample_by_length(tck, h)
        seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1).max()
        check("elem <= %.0f mm" % (h * 1e3), seg <= h + 1e-9, "max segment %.2f mm" % (seg * 1e3))


def test_straight_route():
    print("\n5) straight two-waypoint route")
    tck = geo.fit_spline(np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    R, _ = geo.min_radius(tck)
    check("infinite bend radius", not np.isfinite(R) or R > 1e6, "R = %s" % R)


def test_evaporation():
    """Every admissible edge must evaporate at the same relative rate. Before, only edges
    that already carried a deposit were stored and evaporated, so a never-visited edge
    kept its initial value and ended up more attractive than a tried-and-abandoned one."""
    print("\n6) evaporation reaches every admissible edge")
    g = G.build_grid(4, 4, 4, (0.1, 0.1, 0.1))
    adj = {i: list(v) for i, v in G.build_neighbours(g, connectivity=26).items()}
    n_edges = sum(len(v) for v in adj.values()) // 2
    sid, tid = g["id_of"][(0, 0, 0)], g["id_of"][(3, 3, 3)]
    p = {**A.default_params(), "max_turn_deg": 90.0, "max_ops": 2000, "rho": 0.2}
    r = A.optimize(g, adj, sid, tid, 1.7 * float(np.linalg.norm(
        g["coords"][tid] - g["coords"][sid])),
        lambda c: float(geo.total_length(geo.fit_spline(c))),
        params={**p, "max_iter": 5, "n_ants": 6}, seed=0, verbose=False)
    tau = r["tau"]
    check("all admissible edges carry pheromone",
          len(tau) >= n_edges, "%d stored / %d admissible" % (len(tau), n_edges))
    unreinforced = [v for v in tau.values() if v < p["tau0"] - 1e-12]
    check("unreinforced edges have decayed below tau0",
          len(unreinforced) > 0, "%d edges below tau0" % len(unreinforced))


def test_warm_start():
    """The BEST feasible corridor must become the incumbent, not the first one that
    happens to be feasible."""
    print("\n7) warm start keeps the best feasible corridor")
    g = G.build_grid(5, 5, 5, (0.075, 0.075, 0.075))
    c = g["coords"]
    adj = {i: list(v) for i, v in G.build_neighbours(g, connectivity=26).items()}
    sid, tid = g["id_of"][(0, 0, 0)], g["id_of"][(4, 4, 4)]
    Lmax = 3.0 * float(np.linalg.norm(c[tid] - c[sid]))
    seeds = G.diverse_seed_paths(adj, c, sid, tid, k=3)
    if len(seeds) < 2:
        check("at least two corridors to compare", False, "got %d" % len(seeds))
        return
    # a contrived objective: the LAST corridor is by far the best, so a run that only
    # evaluates the first feasible corridor would start from the worse one
    best_key = tuple(seeds[-1])
    def obj(coords):
        for s in seeds:
            if len(s) == len(coords) and np.allclose(c[np.asarray(s)], coords):
                return 1.0 if tuple(s) == best_key else 100.0
        return 50.0
    geom = lambda co: (geo.total_length(geo.fit_spline(co)), 1.0, True)
    p = {**A.default_params(), **A.auto_params(g, overrides={"n_ants": 2, "max_iter": 1},
                                               verbose=False)}
    p["max_turn_deg"] = 90.0; p["R_min"] = 0.0; p["max_ops"] = 2000
    r = A.optimize(g, adj, sid, tid, Lmax, obj, params=p, seed=0, verbose=False,
                   geom_fn=geom, seed_paths=seeds)
    hist = np.asarray(r["best_hist"], float)
    check("incumbent starts from the BEST corridor (1.0, not 100.0)",
          np.nanmin(hist) <= 1.0 + 1e-9, "best_hist[0] = %.1f" % hist[0])


def test_power_sign():
    print("\n8) input power of a passive structure is non-negative")
    nodes, elems, _ = _mesh(2, 0.075)
    J = fem.evaluate_mesh_power(nodes, elems, MAT, SEC, (6000.0, 8000.0),
                                freq_mode="adaptive", rayleigh=RAY,
                                excitation="prescribed", prescribed_disp=PD)
    check("objective positive and finite", np.isfinite(J) and J > 0.0, "J = %.4e" % J)


if __name__ == "__main__":
    print("=" * 72)
    print("REGRESSION TESTS FOR THE AUDITED FIXES")
    print("=" * 72)
    test_quadrature()
    test_endpoints()
    test_cowper()
    test_element_size()
    test_straight_route()
    test_evaporation()
    test_warm_start()
    test_power_sign()
    print("\n" + "=" * 72)
    if _fails:
        print("FAILED: %d test(s): %s" % (len(_fails), ", ".join(_fails)))
        raise SystemExit(1)
    print("ALL TESTS PASSED")
