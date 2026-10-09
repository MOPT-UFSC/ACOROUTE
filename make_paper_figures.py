"""
make_paper_figures.py
=====================
Legacy figure-generation script retained for reproducibility of an earlier manuscript
version. Its default lattice, target band, smoothing value and plot labels do not
necessarily match the final paper configuration. Consult docs/PAPER_REPRODUCTION.md
before regenerating figures for publication.

Produced here:
  fig_lattice.png              - candidate-node lattice, example route, 6/18/26 connectivity
  fig_turnangle_radius.png     - bend radius vs turn angle (single corner, geometric)
  fig_turnangle_definition.png - turn angle (polyline) vs bend radius (spline) at one corner
  fig_turnangle_proxy.png      - same turn angle, different bend radius (segment length sets R)
  fig_curvature_profile.png    - bend radius along a route spline; R = min over the spline
  fig_pipeline.png             - polyline -> interpolating spline -> beam mesh
  fig_sensitivity_frf.png      - geometry sensitivity: polyline optimum as polyline vs as spline
  fig_optimised_frf.png        - input-power spectrum of the optimised tube
  fig_optimised_geometry.png   - optimised manufacturable tube (3D)
  fig_convergence_baseline.png - convergence history + Monte Carlo baseline
  fig_cost.png                 - per-iteration full-tube FEM solves and cumulative cost funnel

NOT produced here: the process schematics fig_adaptive_grid and fig_containment_tol
come from make_schematic_figures.py (they need the FEM model and the ACO construction);
the block diagram fig_run_spline_blockdiagram and the two hand-drawn schematics
fig_problem_schematic and fig_aco_construction are standalone TikZ, compiled with
  pdflatex fig_run_spline_blockdiagram.tex   (likewise the other two .tex files).
Compile it separately:  pdflatex fig_run_spline_blockdiagram.tex

Runtime note: the result figures (sensitivity, optimised FRF, pipeline, geometry,
baseline) run two ACO optimisations plus a Monte Carlo sweep and take a few minutes,
more in the prescribed-displacement mode. The lattice and turn-angle figures are
instantaneous. Comment out calls in main() to regenerate a subset.

Before using any generated figure in a publication, explicitly synchronize the CONFIG
block with run_spline.py and with the configuration reported in the manuscript.
"""
import os
import numpy as np
import matplotlib
import plot_style  # noqa: F401  (sets Times New Roman for all figures)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from itertools import product
from scipy.interpolate import splev
from scipy.signal import find_peaks

import grid as G
import geometry as geo
import fem_beam3d as fem
import aco as A
import cost_figure

# ============================== CONFIG ==============================
FIGDIR = "."                       # where to write the PNGs
SEED   = 3                         # ACO seed (deterministic)

# geometry / lattice
BOX     = (0.300, 0.300, 0.300)    # bounding box [m]
NODES   = (6, 6, 6)                # nodes per axis
CONNECT = 26                       # 6 / 18 / 26
A_NODE  = (0, 0, 0)                # source index
# B_NODE defaults to the opposite corner

# tube section / material
OUTER_RADIUS = 0.001575
THICKNESS    = 0.0005
E, NU, RHO   = 210e9, 0.3, 7850.0

# excitation (keep consistent with run_spline.py)
BAND            = (6000.0, 8000.0)
RAY             = (0.0, 2e-7)
EXCITATION      = "prescribed"          # "prescribed" or "force"
PRESCRIBED_DISP = (1e-6, 1e-6, 1e-6)    # amplitudes (x, y, z) at A [m]
PRESCRIBED_MASK = (True, True, True)    # which translations at A are prescribed (False = free)
DRIVE           = (0.0, 1.0, 0.0)       # force direction, used only if EXCITATION == "force"

# routing / manufacturability
LMAX_FACTOR = 1.7
MAX_TURN    = 90.0
R_MIN       = 0.008
ELEM        = 0.005
SMOOTH      = 0.0

# Monte Carlo baseline
N_MC = 100
# ====================================================================

# colours
C_SRC, C_SUP, C_POLY, C_SPL = "#f0a800", "#1b7837", "#2a4d69", "#b3122c"
YL = r"$W_\text{in}(f)$ [W]"

mat = {"E": E, "G": E / (2 * (1 + NU)), "rho": RHO}
sec = fem.pipe_section(OUTER_RADIUS, THICKNESS, NU)
SP  = tuple(BOX[k] / (NODES[k] - 1) for k in range(3))
grid = G.build_grid(NODES[0], NODES[1], NODES[2], SP)
GMIN = grid["coords"].min(axis=0)
GMAX = grid["coords"].max(axis=0)
DOMAIN_TOL = 0.5 * min(SP)
adj  = G.build_neighbours(grid, CONNECT)
SID  = grid["id_of"][tuple(A_NODE)]
TID  = grid["id_of"][tuple(NODES[k] - 1 for k in range(3))]
LMAX = LMAX_FACTOR * float(np.linalg.norm(grid["coords"][TID] - grid["coords"][SID]))


def _power_mesh(nodes, elems):
    return fem.evaluate_mesh_power(nodes, elems, mat, sec, BAND, freq_mode="adaptive",
                                   drive_dir=DRIVE, rayleigh=RAY, excitation=EXCITATION,
                                   prescribed_disp=PRESCRIBED_DISP, prescribed_mask=PRESCRIBED_MASK)

def obj_poly(coords):
    nn, ee = fem.mesh_path(coords, 0.035)
    return _power_mesh(nn, ee)

def obj_spline(coords):
    nn, ee, _ = geo.mesh_spline(coords, ELEM, smooth=SMOOTH)
    return _power_mesh(nn, ee)

def geom_fn(coords):
    tck = geo.fit_spline(coords, smooth=SMOOTH)
    R, _ = geo.min_radius(tck)
    L = geo.total_length(tck)
    # domain containment of the as-built spline, same rule as the drivers: excursions
    # beyond half the smallest lattice pitch mean the route leaves the design volume
    sp = geo.eval_spline(tck, 400)
    exc = max(float(np.maximum(GMIN - sp, 0.0).max()),
              float(np.maximum(sp - GMAX, 0.0).max()))
    return L, R, exc <= DOMAIN_TOL

def frf(nodes, elems, freqs):
    if EXCITATION == "prescribed":
        return fem.input_power_spectrum_prescribed(nodes, elems, mat, sec, 0, len(nodes) - 1,
                                                   freqs, prescribed_disp=PRESCRIBED_DISP,
                                                   rayleigh=RAY, prescribed_mask=PRESCRIBED_MASK)
    return fem.input_power_spectrum(nodes, elems, mat, sec, 0, DRIVE, len(nodes) - 1,
                                    freqs, rayleigh=RAY)

def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path)


# ------------------------------------------------------------------ model-independent
def fig_lattice():
    gc = grid["coords"] * 1e3
    # a simple winding example route (adjacent 26-connected steps)
    seq = [(0,0,0),(1,1,0),(2,2,1),(2,3,2),(3,4,2),(4,4,3),(5,5,4),(5,5,5)]
    seq = [s for s in seq if s in grid["id_of"]]
    rc = np.array([grid["coords"][grid["id_of"][s]] for s in seq]) * 1e3
    a0, b0 = gc[SID], gc[TID]
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(gc[:, 0], gc[:, 1], gc[:, 2], s=20, c="0.5", alpha=0.9, depthshade=False)
    ax.plot(rc[:, 0], rc[:, 1], rc[:, 2], "-o", color=C_POLY, lw=2.0, ms=4, label="example route")
    ax.scatter(*a0, s=90, c=C_SRC, edgecolor="k", zorder=6, label="source A")
    ax.scatter(*b0, s=90, c=C_SUP, edgecolor="k", zorder=6, label="support B")
    ax.set_title("Candidate-node lattice and a route", fontsize=11)
    ax.set_xlabel("x [mm]", fontsize=9); ax.set_ylabel("y [mm]", fontsize=9); ax.set_zlabel("z [mm]", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.view_init(elev=20, azim=-60); ax.set_box_aspect((1, 1, 1))
    axc = fig.add_subplot(1, 2, 2, projection="3d")
    cols = {"face (6)": "#2a4d69", "edge (+12 -> 18)": "#e08a00", "body (+8 -> 26)": "#1b7837"}; seen = set()
    for d in product([-1, 0, 1], repeat=3):
        if d == (0, 0, 0):
            continue
        nz = sum(1 for v in d if v != 0)
        typ = "face (6)" if nz == 1 else ("edge (+12 -> 18)" if nz == 2 else "body (+8 -> 26)")
        lbl = typ if typ not in seen else None; seen.add(typ)
        axc.plot([0, d[0]], [0, d[1]], [0, d[2]], "-", color=cols[typ], lw=1.5, alpha=0.9, label=lbl)
        axc.scatter(*d, s=26, c=cols[typ])
    axc.scatter(0, 0, 0, s=80, c="k")
    axc.set_title("Neighbour connectivity (6 / 18 / 26)", fontsize=11)
    axc.set_xlabel("i", fontsize=9); axc.set_ylabel("j", fontsize=9); axc.set_zlabel("k", fontsize=9)
    axc.legend(frameon=False, fontsize=8, loc="upper left"); axc.view_init(elev=22, azim=-55); axc.set_box_aspect((1, 1, 1))
    axc.set_xticks([-1, 0, 1]); axc.set_yticks([-1, 0, 1]); axc.set_zticks([-1, 0, 1])
    fig.tight_layout(); _save(fig, "fig_lattice.png")


def fig_turnangle():
    h = float(np.mean(SP)); OD = 2 * OUTER_RADIUS
    def Rmin_at(theta_deg):
        th = np.deg2rad(theta_deg); dvec = np.array([np.cos(th), np.sin(th), 0.0])
        p = np.array([[-2*h, 0, 0], [-h, 0, 0], [0, 0, 0], h*dvec, 2*h*dvec])
        R, _ = geo.min_radius(geo.fit_spline(p, smooth=0.0)); return R
    ths = np.linspace(10, 170, 60); Rs = np.array([Rmin_at(t) * 1e3 for t in ths])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ths, np.minimum(Rs, 200), color=C_SPL, lw=2.2)
    ax.axhline(OD * 1e3, color="0.5", ls=":", lw=1.2, label="1x OD = %.0f mm" % (OD * 1e3))
    ax.axhline(2 * OD * 1e3, color=C_POLY, ls="--", lw=1.3, label="2x OD = %.0f mm" % (2 * OD * 1e3))
    for tt in (60, 90, 120):
        rr = min(Rmin_at(tt) * 1e3, 200); ax.plot(tt, rr, "ko", ms=5)
        ax.annotate("%.0f mm" % rr, (tt, rr), textcoords="offset points", xytext=(6, 7), fontsize=9)
    ax.set_xlabel(r"turn angle $\theta$ between consecutive segments [deg]")
    ax.set_ylabel(r"bend radius at the corner [mm]")
    ax.set_title(r"Bend radius produced by a single corner (lattice spacing $h=%.0f$ mm)" % (h * 1e3))
    ax.set_ylim(0, 200); ax.grid(True, ls=":", alpha=0.5); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); _save(fig, "fig_turnangle_radius.png")


# ------------------------------------------------------------------ model-dependent
def fig_sensitivity():
    """PART 1: optimise the polyline objective, then compare its FRF as polyline vs as spline."""
    print("[sensitivity] optimising polyline objective ...")
    r1 = A.optimize(grid, adj, SID, TID, LMAX, obj_poly, params=A.auto_params(grid), seed=SEED, verbose=False)
    pc = G.path_coords(grid, r1["best_path"])
    freqs = np.linspace(50, 12000, 500)
    nn_p, ee_p = fem.mesh_path(pc, ELEM); W_poly = frf(nn_p, ee_p, freqs)
    nn_s, ee_s, _ = geo.mesh_spline(pc, ELEM, smooth=SMOOTH); W_spl = frf(nn_s, ee_s, freqs)
    Jp = obj_poly(pc); Js = _power_mesh(nn_s, ee_s)
    print("  polyline winner J: polyline %.3e, spline %.3e  (factor %.1fx)" % (Jp, Js, Js / Jp))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(BAND[0] / 1e3, BAND[1] / 1e3, color="#f0c000", alpha=0.18, label="target band")
    ax.semilogy(freqs / 1e3, np.abs(W_poly) + 1e-30, color=C_POLY, lw=1.8, label="polyline")
    ax.semilogy(freqs / 1e3, np.abs(W_spl) + 1e-30, color=C_SPL, lw=1.8, label="same waypoints, as built spline")
    ax.set_xlabel("frequency [kHz]"); ax.set_ylabel(YL)
    ax.set_title("Geometry sensitivity: smoothing the polyline")
    ax.set_xlim(0, 12); ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); _save(fig, "fig_sensitivity_frf.pdf")


def optimise_tube():
    """PART 2: optimise the spline objective. Returns (route, tck, nodes, best_hist, J, R, L)."""
    print("[optimise] optimising spline objective (manufacturable) ...")
    pars = A.auto_params(grid, overrides={"max_turn_deg": MAX_TURN, "R_min": R_MIN})
    r2 = A.optimize(grid, adj, SID, TID, LMAX, obj_spline, params=pars, seed=SEED, verbose=False, geom_fn=geom_fn)
    tc = G.path_coords(grid, r2["best_path"]); tck = geo.fit_spline(tc, smooth=SMOOTH)
    nn, ee, _ = geo.mesh_spline(tc, ELEM, smooth=SMOOTH)
    R, _ = geo.min_radius(tck); L = geo.total_length(tck)
    print("  optimised tube J %.3e, R_min %.0f mm, length %.0f mm" % (r2["best_power"], R * 1e3, L * 1e3))
    return tc, tck, nn, r2["best_hist"], r2["best_power"], R, L, r2


def _result_figures(tc, tck, nn):
    """fig_optimised_frf, fig_pipeline, fig_optimised_geometry from the optimised tube."""
    # rebuild elements for the FRF
    nodes, elems, _ = geo.mesh_spline(tc, ELEM, smooth=SMOOTH)
    freqs = np.linspace(50, 12000, 500); W = frf(nodes, elems, freqs)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(BAND[0] / 1e3, BAND[1] / 1e3, color="#f0c000", alpha=0.18, label="target band")
    ax.semilogy(freqs / 1e3, np.abs(W) + 1e-30, color=C_SUP, lw=1.9)
    ax.set_xlabel("frequency [kHz]"); ax.set_ylabel(YL); ax.set_title("Input-power spectrum of the optimised tube")
    ax.set_xlim(0, 12); ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); _save(fig, "fig_optimised_frf.png")

    # reaction force/torque spectra at the clamped end B (the blocked transmitted load),
    # on the same frequency axis as the input-power FRF, for the verification section
    if EXCITATION == "prescribed":
        import reaction_figure as RF
        RF.compute_and_plot(nodes, elems, mat, sec, BAND,
                            forces_path=os.path.join(FIGDIR, "fig_optimised_reaction_forces.png"),
                            torques_path=os.path.join(FIGDIR, "fig_optimised_reaction_torques.png"),
                            freq_range=(0.0, 12000.0), n_freq=500, rayleigh=RAY,
                            prescribed_disp=PRESCRIBED_DISP, prescribed_mask=PRESCRIBED_MASK)
        print("  saved fig_optimised_reaction_forces.png, fig_optimised_reaction_torques.png")

    gc = grid["coords"] * 1e3; wp = tc * 1e3
    cv = np.column_stack(splev(np.linspace(0, 1, 400), tck)) * 1e3; mpts = nodes * 1e3
    a0, b0 = gc[SID], gc[TID]
    fig = plt.figure(figsize=(14, 4.8)); titles = ["1. lattice route (polyline)", "2. interpolating spline", "3. beam mesh on the curve"]
    for k in range(3):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d"); ax.scatter(gc[:, 0], gc[:, 1], gc[:, 2], s=8, c="0.55", alpha=0.65, depthshade=False)
        if k == 0:
            ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], "-o", color=C_POLY, lw=1.6, ms=4)
        if k >= 1:
            ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], "--o", color=C_POLY, lw=0.9, ms=3, alpha=0.6)
            ax.plot(cv[:, 0], cv[:, 1], cv[:, 2], "-", color=C_SPL, lw=2.4)
        if k == 2:
            ax.scatter(mpts[:, 0], mpts[:, 1], mpts[:, 2], s=9, c="k", depthshade=False)
        ax.scatter(*a0, s=55, c=C_SRC, edgecolor="k", zorder=6); ax.scatter(*b0, s=55, c=C_SUP, edgecolor="k", zorder=6)
        ax.set_title(titles[k], fontsize=10); ax.view_init(elev=20, azim=-60); ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8); ax.set_zlabel("z", fontsize=8); ax.tick_params(labelsize=6)
    fig.tight_layout(); _save(fig, "fig_pipeline.png")

    fig = plt.figure(figsize=(6.6, 6.0)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gc[:, 0], gc[:, 1], gc[:, 2], s=12, c="0.55", alpha=0.65, depthshade=False)
    ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], "--o", color=C_POLY, lw=1.0, ms=3, alpha=0.7, label="waypoints")
    ax.plot(cv[:, 0], cv[:, 1], cv[:, 2], "-", color=C_SPL, lw=2.8, label="manufacturable centreline")
    ax.scatter(*a0, s=80, c=C_SRC, edgecolor="k", zorder=6, label="source A"); ax.scatter(*b0, s=80, c=C_SUP, edgecolor="k", zorder=6, label="support B")
    ax.set_title("Optimised manufacturable tube", fontsize=11)
    ax.set_xlabel("x [mm]", fontsize=9); ax.set_ylabel("y [mm]", fontsize=9); ax.set_zlabel("z [mm]", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.view_init(elev=20, azim=-60); ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(); _save(fig, "fig_optimised_geometry.png")


def fig_baseline(best_hist, opt_val):
    """fig_convergence_baseline: convergence history + Monte Carlo distribution of feasible routes."""
    print("[baseline] Monte Carlo sampling of %d feasible routes ..." % N_MC)
    sc = np.array([grid["coords"][SID], grid["coords"][TID]]); nn, ee = fem.mesh_path(sc, ELEM)
    straight = _power_mesh(nn, ee)
    pars = {**A.default_params(), **A.auto_params(grid, overrides={"max_turn_deg": MAX_TURN, "R_min": R_MIN})}
    rng = np.random.default_rng(11); samples = []; att = 0
    while len(samples) < N_MC and att < 8 * N_MC:
        att += 1
        path, _ = A.build_one_ant(grid, adj, {}, SID, TID, LMAX, pars, rng, q0=0.0)
        if path is None:
            continue
        c = G.path_coords(grid, path); Lspl, R, ok = geom_fn(c)
        if not ok or Lspl > LMAX or R < R_MIN:
            continue
        samples.append(obj_spline(c))
    samples = np.array(samples)
    print("  straight J=%.3e | MC n=%d, min=%.3e median=%.3e max=%.3e | below opt %.1f%%"
          % (straight, len(samples), samples.min(), np.median(samples), samples.max(), 100 * np.mean(samples < opt_val)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    it = np.arange(1, len(best_hist) + 1)
    a1.plot(it, best_hist, "-o", color=C_SPL, lw=1.8, ms=3)
    a1.axhline(straight, color=C_POLY, ls="--", lw=1.3, label="straight route")
    a1.set_xlabel("iteration"); a1.set_ylabel(r"best objective $J$ [W$\cdot$rad/s]")
    a1.set_title("Convergence of the best route"); a1.set_yscale("log"); a1.grid(True, which="both", ls=":", alpha=0.4); a1.legend(frameon=False, fontsize=9)
    lo, hi = samples.min() * 0.7, samples.max() * 1.4; bins = np.logspace(np.log10(lo), np.log10(hi), 24)
    a2.hist(samples, bins=bins, color="0.72", edgecolor="0.4", label="feasible routes (Monte Carlo, n=%d)" % len(samples))
    a2.axvline(opt_val, color=C_SUP, lw=2.2, label="optimised tube")
    a2.axvline(straight, color=C_POLY, ls="--", lw=1.8, label="straight route")
    a2.set_xscale("log"); a2.set_xlabel(r"band-integrated input power $J$ [W$\cdot$rad/s]"); a2.set_ylabel("number of routes")
    a2.set_title("Objective over the admissible search space"); a2.grid(True, which="both", ls=":", alpha=0.4); a2.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); _save(fig, "fig_convergence_baseline.png")


def main():
    print("Excitation: %s  disp=%s  mask=%s" % (EXCITATION, PRESCRIBED_DISP, PRESCRIBED_MASK))
    # --- model-independent (instantaneous) ---
    fig_lattice()
    fig_turnangle()
    fig_turnangle_definition()
    fig_turnangle_proxy()
    fig_curvature_profile()
    # --- model-dependent (runs optimisations; minutes) ---
    fig_sensitivity()                              # PART 1
    tc, tck, nn, best_hist, J, R, L, r2 = optimise_tube()  # PART 2
    _result_figures(tc, tck, nn)                   # optimised FRF, pipeline, geometry
    fig_baseline(best_hist, J)                     # convergence + Monte Carlo
    fig_cost(r2)                                   # per-iteration + cumulative cost
    print("\nAll data figures written to %s/" % FIGDIR)
    print("Block diagram is separate: pdflatex fig_run_spline_blockdiagram.tex")



# ------------------------------------------------------------------ geometry detail figures
def _spline_corner(theta_deg, h_mm):
    """One planar corner (segment length h_mm, deviation angle theta). Returns
    polyline pts, spline samples, min bend radius R, osculating-circle centre and
    min-radius point, all in mm. Interior search avoids end-point artefacts."""
    from scipy.interpolate import splev
    th = np.deg2rad(theta_deg); dv = np.array([np.cos(th), np.sin(th)])
    P = np.array([[s*h_mm, 0.0] for s in (-4, -3, -2, -1, 0)]
                 + [(s*h_mm*dv[0], s*h_mm*dv[1]) for s in (1, 2, 3, 4)])
    tck = geo.fit_spline(np.column_stack([P, np.zeros(len(P))]), smooth=0.0)
    u = np.linspace(0, 1, 1400); xyz = np.array(splev(u, tck)); x, y = xyz[0], xyz[1]
    d1 = np.array(splev(u, tck, der=1)); d2 = np.array(splev(u, tck, der=2))
    xp, yp, xpp, ypp = d1[0], d1[1], d2[0], d2[1]; sp = np.hypot(xp, yp)
    ks = (xp*ypp - yp*xpp) / sp**3
    idx = np.where((u > 0.3) & (u < 0.7))[0]; i = idx[np.argmax(np.abs(ks[idx]))]
    R = 1.0/abs(ks[i]); nx, ny = -yp[i]/sp[i], xp[i]/sp[i]
    cen = np.array([x[i] + (1.0/ks[i])*nx, y[i] + (1.0/ks[i])*ny])
    return P[:, :2], np.column_stack([x, y]), R, cen, np.array([x[i], y[i]])


def fig_turnangle_definition():
    from matplotlib.patches import Arc
    h = float(np.mean(SP)) * 1e3; th = 70.0
    poly, spl, R, cen, pmin = _spline_corner(th, h)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    a = np.linspace(0, 2*np.pi, 240)
    ax.plot(cen[0]+R*np.cos(a), cen[1]+R*np.sin(a), color=C_SUP, lw=1.2, ls=":")
    ax.plot([cen[0], pmin[0]], [cen[1], pmin[1]], color=C_SUP, lw=1.5)
    ax.plot(*cen, "o", color=C_SUP, ms=5)
    ax.annotate(r"$R$", ((cen[0]+pmin[0])/2-6, (cen[1]+pmin[1])/2), color=C_SUP, fontsize=15)
    ax.plot(poly[:, 0], poly[:, 1], "--o", color=C_POLY, lw=1.4, ms=5, label="lattice polyline")
    ax.plot(spl[:, 0], spl[:, 1], "-", color=C_SPL, lw=2.7, label="manufacturable spline")
    ax.plot(0, 0, "o", color="k", ms=6, zorder=6)
    ax.plot([0, 0.7*h], [0, 0], color="0.6", lw=1.0, ls=(0, (4, 3)))
    ax.add_patch(Arc((0, 0), 0.62*h, 0.62*h, theta1=0, theta2=th, color=C_POLY, lw=1.7))
    ax.annotate(r"$\theta$", (0.4*h*np.cos(np.deg2rad(th/2)), 0.4*h*np.sin(np.deg2rad(th/2))),
                color=C_POLY, fontsize=16, ha="center", va="center")
    ax.annotate("turn angle\n(polyline)", (0.55*h, -0.22*h), color=C_POLY, fontsize=9.5)
    ax.annotate("bend radius\n(spline)", (cen[0]-0.5*h, cen[1]+0.4*h), color=C_SUP, fontsize=9.5, ha="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.7*h, 1.5*h); ax.set_ylim(-0.45*h, 1.7*h)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(r"One corner: polyline turn angle $\theta$, spline bend radius $R$")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.grid(True, ls=":", alpha=0.35)
    fig.tight_layout(); _save(fig, "fig_turnangle_definition.png")


def fig_turnangle_proxy():
    from matplotlib.patches import Arc
    hf = float(np.mean(SP)) * 1e3; hb = hf*np.sqrt(3.0); th = 70.0
    XL = (-2.75*hf, 2.5*hf); YL = (-0.75*hf, 3.0*hf)
    labs = ["face edge ($h=%.0f$ mm)" % hf, "body diagonal ($h=%.0f$ mm)" % hb]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
    for k, (ax, hh, lab) in enumerate(zip(axs, (hf, hb), labs)):
        poly, spl, R, cen, pmin = _spline_corner(th, hh)
        a = np.linspace(0, 2*np.pi, 240)
        ax.plot(cen[0]+R*np.cos(a), cen[1]+R*np.sin(a), color=C_SUP, lw=1.3, ls=":")
        ax.plot([cen[0], pmin[0]], [cen[1], pmin[1]], color=C_SUP, lw=1.4)
        ax.plot(*cen, "o", color=C_SUP, ms=4)
        ax.plot(poly[:, 0], poly[:, 1], "--o", color=C_POLY, lw=1.3, ms=4,
                label="lattice polyline" if k == 0 else None)
        ax.plot(spl[:, 0], spl[:, 1], "-", color=C_SPL, lw=2.7,
                label="manufacturable spline" if k == 0 else None)
        ax.plot(0, 0, "o", color="k", ms=6, zorder=6)
        ax.plot([0, 0.85*hh], [0, 0], color="0.6", lw=1.0, ls=(0, (4, 3)))
        ax.add_patch(Arc((0, 0), 0.62*hh, 0.62*hh, theta1=0, theta2=th, color=C_POLY, lw=1.7))
        ax.annotate(r"$\theta=70^\circ$",
                    (0.42*hh*np.cos(np.deg2rad(th/2))+8, 0.42*hh*np.sin(np.deg2rad(th/2))),
                    color=C_POLY, fontsize=12, ha="left", va="center")
        ax.annotate(r"$R=%.0f$ mm" % R, (cen[0], cen[1]), textcoords="offset points",
                    xytext=(-2, -17), color=C_SUP, fontsize=13, ha="center", fontweight="bold")
        ax.set_aspect("equal"); ax.set_xlim(*XL); ax.set_ylim(*YL)
        ax.set_title(lab, fontsize=12); ax.set_xlabel("x [mm]"); ax.grid(True, ls=":", alpha=0.35)
    axs[0].set_ylabel("y [mm]"); axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(r"Same turn angle $\theta=70^\circ$, different bend radius $R$: the segment length sets $R$",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(); _save(fig, "fig_turnangle_proxy.png")


def fig_curvature_profile():
    from scipy.interpolate import splev
    h = float(np.mean(SP)) * 1e3
    idx = np.array([(0, 0, 0), (1, 1, 0), (2, 1, 1), (3, 2, 1), (3, 3, 2),
                    (4, 4, 3), (5, 4, 4), (5, 5, 5)], float)
    wp = idx * h
    tck = geo.fit_spline(wp, smooth=0.0)
    u = np.linspace(0, 1, 3000); P = np.column_stack(splev(u, tck))
    d1 = np.array(splev(u, tck, der=1)).T; d2 = np.array(splev(u, tck, der=2)).T
    kap = np.linalg.norm(np.cross(d1, d2), axis=1) / np.linalg.norm(d1, axis=1)**3
    R = 1.0/np.maximum(kap, 1e-12)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; L = s[-1]
    i = int(np.argmin(R)); RMIN = R_MIN * 1e3
    fig = plt.figure(figsize=(12.5, 5.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], "--o", color=C_POLY, lw=1.0, ms=4, alpha=0.7, label="waypoints")
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", color=C_SPL, lw=2.6, label="spline centreline")
    ax.scatter(*P[i], s=130, marker="*", color=C_SUP, edgecolor="k", zorder=6, label="tightest bend")
    ax.scatter(*wp[0], s=70, c=C_SRC, edgecolor="k", zorder=6)
    ax.scatter(*wp[-1], s=70, c=C_SUP, edgecolor="k", zorder=6)
    ax.set_title("Spline centreline of a route", fontsize=11)
    ax.set_xlabel("x [mm]", fontsize=9); ax.set_ylabel("y [mm]", fontsize=9); ax.set_zlabel("z [mm]", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.view_init(elev=20, azim=-62); ax.set_box_aspect((1, 1, 1))
    a2 = fig.add_subplot(1, 2, 2)
    a2.plot(s, np.minimum(R, 140), "-", color=C_SPL, lw=2.0)
    a2.axhline(RMIN, color="0.5", ls="--", lw=1.5); a2.fill_between(s, 0, RMIN, color="0.85", alpha=0.7)
    a2.annotate(r"$R_{\min}=%.0f$ mm" % RMIN, (L*0.62, RMIN+4), color="0.3", fontsize=10)
    a2.plot(s[i], R[i], "*", color=C_SUP, ms=16, markeredgecolor="k", zorder=6)
    a2.annotate("tightest bend\n$R(\\gamma)=%.0f$ mm" % R[i], (s[i], R[i]), textcoords="offset points",
                xytext=(14, 16), color=C_SUP, fontsize=10)
    a2.set_xlabel(r"arc length $\ell$ [mm]"); a2.set_ylabel(r"bend radius $\mathcal{R}_\gamma(\ell)$ [mm]  (clipped at 140)")
    a2.set_title(r"Bend radius along the spline; the route value is $R(\gamma)=\min_\ell \mathcal{R}_\gamma(\ell)$", fontsize=11)
    a2.set_ylim(0, 150); a2.set_xlim(0, L); a2.grid(True, ls=":", alpha=0.4)
    fig.tight_layout(); _save(fig, "fig_curvature_profile.png")


def fig_cost(result):
    """Per-iteration full-tube FEM solves and the cumulative cost funnel.

    Delegates to the shared cost_figure module so run_spline.py and this script
    produce the figure identically, and persists the cost arrays so the figure can
    be regenerated later without re-running the optimisation."""
    cost_figure.save_cost_data(result, os.path.join(FIGDIR, "fig_cost_data.npz"))
    cost_figure.make_cost_figure(result, os.path.join(FIGDIR, "fig_cost.png"))
    print("  saved", os.path.join(FIGDIR, "fig_cost.png"))



if __name__ == "__main__":
    main()
