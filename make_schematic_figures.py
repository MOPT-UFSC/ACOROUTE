"""
make_schematic_figures.py
Legacy process-schematic generator retained for reproducibility of earlier figures.
The numerical defaults in this file predate the final paper configuration; synchronize
its target band, excitation and spline settings before regenerating publication figures.

  fig_adaptive_grid.png    the eigenfrequency-guided frequency sampling of Section
                           "Adaptive evaluation of the band objective": the input-power
                           spectrum of a representative tube (the 2-turn helix baseline
                           of the base case), the undamped natural frequencies that
                           place the samples, the adaptive grid, and a uniform grid of
                           the same size that misses the peaks.

  fig_containment_tol.png  the containment tolerance of the geometry filter: an
                           interpolating spline through waypoints on the boundary row
                           bulges outside the box between them, by a distance that
                           scales with the lattice pitch; excursions up to half the
                           smallest pitch are accepted, larger ones rejected.

Run from the repository root:  python make_schematic_figures.py
Figures are written into paper/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style  # noqa: F401  (sets Times New Roman for all figures)

import geometry as geo
import fem_beam3d as fem
import baseline_routes as BL

FIGDIR = "paper"

# Legacy configuration shared with make_paper_figures.py; it is not guaranteed to match
# the final paper case. See docs/PAPER_REPRODUCTION.md before regenerating figures.
BOX = np.array([0.300, 0.300, 0.300])
E, nu, rho = 210e9, 0.3, 7850.0
OUTER_RADIUS, THICKNESS = 0.001575, 0.0005
BAND = (6000.0, 8000.0)
RAY = (0.0, 2e-7)
PRESCRIBED_DISP = (1e-6, 1e-6, 1e-6)
ELEM = 0.005

mat = {"E": E, "G": E / (2.0 * (1.0 + nu)), "rho": rho}
sec = fem.pipe_section(outer_radius=OUTER_RADIUS, thickness=THICKNESS, nu=nu)


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path)


# ============================================================================ #
# 1) adaptive frequency sampling
# ============================================================================ #

def fig_adaptive_grid(band=(1000.0, 3000.0), n_uniform=None, ref_step_hz=0.25):
    """Eigenfrequency-guided sampling versus a converged uniform reference.

    `band` defaults to the lowest reported band, where the modal bandwidths are
    narrowest and the rule matters most. `n_uniform` defaults to the base-grid size of
    the production rule (fem_beam3d._adaptive_freq_grid), so the "base grid alone"
    curve is exactly the backbone of the adaptive grid. The reference uses a uniform
    step of `ref_step_hz`, well below the narrowest modal bandwidth in the band.
    """
    BAND = tuple(band)
    if n_uniform is None:
        n_uniform = fem._adaptive_freq_grid.__defaults__[0]      # n_base of the production rule
    # representative tube: the 2-turn helix baseline of the base case
    A = np.zeros(3); B = BOX.copy()
    pts = BL.helix_points(A, B, turns=2, radius=0.075, n=400, taper=True)
    tck = geo.fit_spline(pts, smooth=0.0)
    nodes, elems, _ = geo.resample_by_length(tck, ELEM)
    drive, fixed = 0, len(nodes) - 1
    extra = [6 * drive + k for k in range(3)]          # prescribed translations at A

    def W(f):
        return np.abs(np.asarray(fem.input_power_spectrum_prescribed(
            nodes, elems, mat, sec, drive, fixed, f,
            prescribed_disp=PRESCRIBED_DISP, rayleigh=RAY)))

    f_ref = np.arange(BAND[0], BAND[1] + 0.5 * ref_step_hz, ref_step_hz)   # converged reference
    W_ref = W(f_ref); J_ref = np.trapezoid(W_ref, f_ref)

    # the grids exactly as the objective evaluation builds them: the adaptive grid is
    # the n_base-point uniform base grid plus clusters at the in-band natural frequencies
    peaks = fem.natural_frequencies(nodes, elems, mat, sec, fixed, BAND,
                                    extra_fixed_dofs=extra)
    f_ad = fem._adaptive_freq_grid(BAND, peaks, RAY)
    f_b = np.linspace(BAND[0], BAND[1], n_uniform)      # coarse uniform comparison
    W_ad = W(f_ad); W_b = W(f_b)
    J_ad = np.trapezoid(W_ad, f_ad); J_b = np.trapezoid(W_b, f_b)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.semilogy(f_ref, W_ref, "-", color="0.55", lw=1.1,
                label="reference (%d points, %.2f Hz step)" % (len(f_ref), ref_step_hz))
    first = True
    for p in peaks:
        if BAND[0] <= p <= BAND[1]:
            ax.axvline(p, color="#2a4d69", ls=":", lw=0.9,
                       label=r"natural frequencies $\omega_r$" if first else None)
            first = False
    base_tag = ("base grid alone" if n_uniform == fem._adaptive_freq_grid.__defaults__[0]
                else "uniform grid")
    ax.semilogy(f_b, W_b, "x--", color="#b3122c", lw=0.9, ms=7,
                label="%s (%d points): err %+.0f%%"
                      % (base_tag, len(f_b), 100 * (J_b - J_ref) / J_ref))
    ax.semilogy(f_ad, W_ad, "o", color="#1b7837", ms=4,
                label="adaptive grid (%d points): err %+.0f%%"
                      % (len(f_ad), 100 * (J_ad - J_ref) / J_ref))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$\bar{W}(\omega)$  [W]")   # no modulus: the active input power is non-negative for a passive structure
    ax.set_xlim(BAND)
    # stretch the lower y-limit to open an empty band under the spectrum, so the
    # lower-right legend sits on empty space instead of on the curves
    lo = min(W_ref.min(), W_ad.min(), W_b.min())
    hi = max(W_ref.max(), W_ad.max(), W_b.max())
    ax.set_ylim(lo / 7.0, hi * 1.5)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    # legend inside the axes, lower right, on a light box so the curves underneath stay readable
    ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="none", facecolor="white")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_adaptive_grid.pdf"), bbox_inches="tight")
    _save(fig, "fig_adaptive_grid.png")
    print("    J_ref=%.4e  adaptive %+.1f%%  base-only %+.1f%%"
          % (J_ref, 100 * (J_ad - J_ref) / J_ref, 100 * (J_b - J_ref) / J_ref))


# ============================================================================ #
# 2) containment tolerance of the geometry filter (real routes, measured)
# ============================================================================ #

def fig_containment_tol(n_routes=200, seed=0):
    import grid as G
    import aco as ACO
    NODES = (6, 6, 6)
    SP = tuple(BOX[k] / (NODES[k] - 1) for k in range(3))
    g = G.build_grid(*NODES, SP)
    coords = g["coords"]
    adj = {i: list(v) for i, v in G.build_neighbours(g, connectivity=26).items()}
    sid = g["id_of"][(0, 0, 0)]; tid = g["id_of"][(5, 5, 5)]
    Lmax = 1.7 * float(np.linalg.norm(coords[tid] - coords[sid]))
    gmin = coords.min(0); gmax = coords.max(0)
    tol = 0.5 * min(SP)
    params = {**ACO.default_params(),
              **ACO.auto_params(g, overrides={"n_ants": 10, "max_iter": 10},
                                verbose=False)}
    params["max_turn_deg"] = 90.0; params["max_ops"] = 4000
    rng = np.random.default_rng(seed)

    def excursion_profile(route, smooth):
        tck = geo.fit_spline(coords[np.asarray(route)], smooth=smooth)
        sp = geo.eval_spline(tck, 500)
        e = np.maximum(np.maximum(gmin - sp, sp - gmax), 0.0).max(axis=1)
        return e

    exc0, exc1 = [], []
    worst = (0.0, None)
    for _ in range(n_routes):
        path, _L = ACO.build_one_ant(g, adj, {}, sid, tid, Lmax, params, rng,
                                     q0=0.4)
        if path is None:
            continue
        e0 = excursion_profile(path, 0.0)
        e1 = excursion_profile(path, 1e-3)
        exc0.append(e0.max()); exc1.append(e1.max())
        if e0.max() > worst[0]:
            worst = (e0.max(), path)
    exc0 = np.array(exc0); exc1 = np.array(exc1)

    fig, (axp, axh) = plt.subplots(1, 2, figsize=(8.6, 3.7),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    # left: excursion profile of the worst interpolated route along its arc length
    e = excursion_profile(worst[1], 0.0)
    s = np.linspace(0.0, 1.0, len(e))
    axp.fill_between(s, 0.0, e * 1e3, color="#b3122c", alpha=0.25)
    axp.plot(s, e * 1e3, "-", color="#b3122c", lw=1.6,
             label="worst route: %.0f mm" % (worst[0] * 1e3))
    axp.axhline(tol * 1e3, color="#c98a00", ls="--", lw=1.3,
                label="tolerance $h/2$ = %.0f mm" % (tol * 1e3))
    axp.axhline(0.0, color="k", lw=1.6)
    axp.text(0.01, 1.2, "design box wall", fontsize=8)
    axp.set_xlabel("normalized arc length")
    axp.set_ylabel("excursion outside the box [mm]")
    axp.set_ylim(0, max(tol * 1e3 * 1.35, e.max() * 1e3 * 1.25))
    axp.legend(frameon=False, fontsize=8)
    axp.grid(True, ls=":", alpha=0.4)
    # right: distribution of maximum excursions over the sampled routes
    bins = np.linspace(0, max(exc1.max(), tol * 1.3) * 1e3, 26)
    axh.hist(exc0 * 1e3, bins=bins, color="#2a4d69", alpha=0.75,
             label="interpolating spline (%d routes)" % len(exc0))
    axh.hist(exc1 * 1e3, bins=bins, color="#9bb4c9", alpha=0.60,
             label="smoothed spline")
    axh.axvline(tol * 1e3, color="#c98a00", ls="--", lw=1.3, label="tolerance $h/2$")
    axh.set_xlabel("maximum excursion of the route [mm]")
    axh.set_ylabel("routes")
    axh.legend(frameon=False, fontsize=8)
    axh.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_containment_tol.png")
    print("    excursions [mm]: interp mean %.1f max %.1f | smoothed mean %.1f max %.1f"
          " | tol %.0f | rejected: interp %d, smoothed %d of %d"
          % (exc0.mean() * 1e3, exc0.max() * 1e3, exc1.mean() * 1e3, exc1.max() * 1e3,
             tol * 1e3, int((exc0 > tol).sum()), int((exc1 > tol).sum()), len(exc0)))



# ============================================================================ #
# 3) overview of the three example domains (Section 2 opening figure)
# ============================================================================ #

def fig_overview_domains(seed=4):
    """Three-panel 3D overview: (a) open box, (b) central keep-out, (c) serpentine
    channel, each with the lattice, one example route as a polyline, and the smooth
    spline through the same waypoints. No beam mesh: the figure carries the idea of
    the paper (route on a lattice, tube as a smooth curve), not the discretization."""
    import grid as G
    import aco as ACO
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    NAVY, CRIM = "#2a4d69", "#b3122c"

    def draw_nodes(ax, coords):
        ax.scatter(coords[:, 0]*1e3, coords[:, 1]*1e3, coords[:, 2]*1e3,
                   s=16, c="0.45", alpha=0.35, depthshade=False)

    def draw_route(ax, coords, path, smooth, label=False):
        w = coords[np.asarray(path)]
        ax.plot(w[:, 0]*1e3, w[:, 1]*1e3, w[:, 2]*1e3, "--o", color=NAVY,
                lw=2.2, ms=5.5, zorder=6,
                label="lattice route (polyline)" if label else None)
        sp = geo.eval_spline(geo.fit_spline(w, smooth=smooth), 400)
        ax.plot(sp[:, 0]*1e3, sp[:, 1]*1e3, sp[:, 2]*1e3, "-", color=CRIM,
                lw=2.8, zorder=5, label="smooth centreline (spline)" if label else None)
        ax.scatter(*(w[0]*1e3), c="gold", s=90, edgecolor="k", zorder=8,
                   depthshade=False, label="source $A$" if label else None)
        ax.scatter(*(w[-1]*1e3), c="green", s=90, edgecolor="k", zorder=8,
                   depthshade=False, label="support $B$" if label else None)

    def box_edges(ax, lo, hi, color, lw=1.2):
        l, h = np.asarray(lo)*1e3, np.asarray(hi)*1e3
        for s, e in [((0,0,0),(1,0,0)),((0,0,0),(0,1,0)),((0,0,0),(0,0,1)),
                     ((1,1,1),(0,1,1)),((1,1,1),(1,0,1)),((1,1,1),(1,1,0)),
                     ((1,0,0),(1,1,0)),((1,0,0),(1,0,1)),((0,1,0),(1,1,0)),
                     ((0,1,0),(0,1,1)),((0,0,1),(1,0,1)),((0,0,1),(0,1,1))]:
            p = [l[k] if s[k]==0 else h[k] for k in range(3)]
            q = [l[k] if e[k]==0 else h[k] for k in range(3)]
            ax.plot(*zip(p, q), color=color, lw=lw, zorder=4)

    def slab(ax, lo, hi, color, alpha):
        l, h = np.asarray(lo)*1e3, np.asarray(hi)*1e3
        v = np.array([[l[0],l[1],l[2]],[h[0],l[1],l[2]],[h[0],h[1],l[2]],[l[0],h[1],l[2]],
                      [l[0],l[1],h[2]],[h[0],l[1],h[2]],[h[0],h[1],h[2]],[l[0],h[1],h[2]]])
        faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
        pc = Poly3DCollection([v[f] for f in faces], facecolor=color, alpha=alpha,
                              edgecolor=color, linewidths=0.6)
        ax.add_collection3d(pc)

    def ant_route(grid_, adj_, sid, tid, Lmax, rng, smooth, extra_ok=None,
                  hops=(7, 13), tries=400):
        p = {**ACO.default_params(), "max_turn_deg": 90.0, "max_ops": 4000}
        for _ in range(tries):
            path, _L = ACO.build_one_ant(grid_, adj_, {}, sid, tid, Lmax, p, rng,
                                         q0=0.3)
            if path is None or not (hops[0] <= len(path)-1 <= hops[1]):
                continue
            w = grid_["coords"][np.asarray(path)]
            tck = geo.fit_spline(w, smooth=smooth)
            R, _ = geo.min_radius(tck); L = geo.total_length(tck)
            if L > Lmax or R < 0.008:
                continue
            sp = geo.eval_spline(tck, 400)
            if extra_ok is not None and not extra_ok(sp):
                continue
            return path
        return None

    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(13.2, 4.9), layout="constrained")

    # --- (a) open box ------------------------------------------------------
    B1 = np.array([0.300, 0.300, 0.300]); N1 = (6, 6, 6)
    SP1 = tuple(B1[k]/(N1[k]-1) for k in range(3))
    g1 = G.build_grid(*N1, SP1); c1 = g1["coords"]
    adj1 = {i: list(v) for i, v in G.build_neighbours(g1, connectivity=26).items()}
    s1 = g1["id_of"][(0, 0, 0)]; t1 = g1["id_of"][(5, 5, 5)]
    Lmax1 = 1.7*float(np.linalg.norm(c1[t1]-c1[s1]))
    gmin1, gmax1 = c1.min(0), c1.max(0); tol1 = 0.5*min(SP1)
    ok1 = lambda sp: max(np.maximum(gmin1-sp, 0).max(), np.maximum(sp-gmax1, 0).max()) <= tol1
    p1 = ant_route(g1, adj1, s1, t1, Lmax1, rng, 0.0, extra_ok=ok1)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw_nodes(ax, c1); box_edges(ax, gmin1, gmax1, "0.75", 0.8)
    draw_route(ax, c1, p1, 0.0, label=True)
    ax.set_title("open region", fontsize=11)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)

    # --- (b) central keep-out ---------------------------------------------
    f = 0.4
    lo2 = 0.5*(1-f)*B1; hi2 = 0.5*(1+f)*B1
    def mask2(i, j, k):
        x = np.array([i*SP1[0], j*SP1[1], k*SP1[2]])
        return not np.all((x > lo2) & (x < hi2))
    g2 = G.build_grid(*N1, SP1, mask=mask2); c2 = g2["coords"]
    adjr = G.build_neighbours(g2, connectivity=26)
    def seg_free(pq):
        p, q = pq
        for t in np.linspace(0, 1, 7)[1:-1]:
            r = p + t*(q-p)
            if np.all((r > lo2) & (r < hi2)):
                return False
        return True
    adj2 = {i: [(j, L) for (j, L) in v if seg_free((c2[i], c2[j]))]
            for i, v in adjr.items()}
    s2 = g2["id_of"][(0, 0, 0)]; t2 = g2["id_of"][(5, 5, 5)]
    Lmax2 = 1.7*float(np.linalg.norm(c2[t2]-c2[s2]))
    gmin2, gmax2 = c2.min(0), c2.max(0); tol2 = 0.5*min(SP1)
    def ok2(sp):
        if np.any(np.all((sp > lo2) & (sp < hi2), axis=1)):
            return False
        return max(np.maximum(gmin2-sp, 0).max(), np.maximum(sp-gmax2, 0).max()) <= tol2
    p2 = ant_route(g2, adj2, s2, t2, Lmax2, rng, 0.0, extra_ok=ok2)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    draw_nodes(ax, c2); box_edges(ax, gmin2, gmax2, "0.75", 0.8)
    slab(ax, lo2, hi2, CRIM, 0.10); box_edges(ax, lo2, hi2, CRIM, 1.2)
    draw_route(ax, c2, p2, 0.0)
    ax.set_title("single obstacle", fontsize=11)

    # --- (c) serpentine channel (staircase) --------------------------------
    B3 = np.array([0.300, 0.160, 0.300]); N3 = (7, 3, 7)
    SP3 = tuple(B3[k]/(N3[k]-1) for k in range(3))
    sx, sy, sz = SP3
    LO_IZ, LO_IX = 2, (0, 1, 2, 3)
    UP_IZ, UP_IX = 4, (3, 4, 5, 6)
    def mask3(i, j, k):
        if k == LO_IZ and i in LO_IX: return False
        if k == UP_IZ and i in UP_IX: return False
        return True
    g3 = G.build_grid(*N3, SP3, mask=mask3); c3 = g3["coords"]
    def wall_box(cols, iz):
        xmin = 0.0 if min(cols) == 0 else (min(cols)-0.5)*sx
        xmax = B3[0] if max(cols) == N3[0]-1 else (max(cols)+0.5)*sx
        zc = iz*sz
        return (np.array([xmin, 0.0, zc-0.2*sz]), np.array([xmax, B3[1], zc+0.2*sz]))
    walls = [wall_box(LO_IX, LO_IZ), wall_box(UP_IX, UP_IZ)]
    adjr3 = G.build_neighbours(g3, connectivity=26)
    def seg_free3(p, q):
        for t in np.linspace(0, 1, 7)[1:-1]:
            r = p + t*(q-p)
            for lo, hi in walls:
                if np.all((r > lo) & (r < hi)):
                    return False
        return True
    adj3 = {i: [(j, L) for (j, L) in v if seg_free3(c3[i], c3[j])]
            for i, v in adjr3.items()}
    s3 = g3["id_of"][(0, 0, 0)]; t3 = g3["id_of"][(N3[0]-1, N3[1]-1, N3[2]-1)]
    p3 = G.shortest_path(adj3, s3, t3)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    draw_nodes(ax, c3); box_edges(ax, c3.min(0), c3.max(0), "0.75", 0.8)
    for lo, hi in walls:
        slab(ax, lo, hi, "#7a8ca3", 0.35)
        box_edges(ax, lo, hi, "#41546e", 1.4)   # strong edges, as on the keep-out cube
    draw_route(ax, c3, p3, 1e-3)
    ax.set_title("double obstacle", fontsize=11)

    for k, a in enumerate(fig.axes):
        a.set_xlabel("x [mm]", fontsize=8, labelpad=-2)
        a.set_ylabel("y [mm]", fontsize=8, labelpad=-2)
        a.set_zlabel("z [mm]", fontsize=8, labelpad=-2)
        a.tick_params(labelsize=6.5, pad=-1)
        if k < 2:
            a.set_box_aspect((1, 1, 1)); a.view_init(elev=18, azim=-62)
        else:
            a.set_box_aspect(tuple(B3/B3.max())); a.view_init(elev=16, azim=-70)
    _save(fig, "fig_overview_domains.png")
    print("    routes: a=%d hops, b=%d hops, c=%d hops"
          % (len(p1)-1, len(p2)-1, len(p3)-1))


if __name__ == "__main__":
    print("schematic figures:")
    fig_adaptive_grid()
    fig_containment_tol()
    fig_overview_domains()