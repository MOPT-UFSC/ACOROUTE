"""
plotting.py
Visualisation helpers (Matplotlib). Procedural style.
"""

import os
import sys
import numpy as np
import matplotlib
# Headless Linux (no display) must use the non-interactive Agg backend so figures
# can still be saved. On a desktop, actively pick an interactive backend so the 3D
# geometry can be rotated on screen, instead of relying on Matplotlib's default
# (which silently falls back to Agg when no GUI backend is pre-selected). Falls
# back quietly if none of the candidates are importable.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
else:
    for _backend, _probe in (("TkAgg", "tkinter"), ("QtAgg", "PyQt5"), ("Qt5Agg", "PyQt5")):
        try:
            __import__(_probe)
            matplotlib.use(_backend)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt

import plot_style  # noqa: F401  (sets Times New Roman for all figures)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from grid import path_coords


# --- figure typography for the manuscript -------------------------------------
# Centralised sizes keep every figure readable after reduction in the article.
# These constants affect typography only; colours, lines, markers and layouts are
# intentionally left unchanged.
AXIS_FS = 14       # nomes dos eixos nos gráficos 2D
TITLE_FS = 14
TICK_FS = 12       # valores dos eixos nos gráficos 2D

AXIS_3D_FS = 11    # nomes dos eixos x, y e z nos gráficos 3D
TICK_3D_FS = 9    # valores numéricos dos eixos 3D

LEGEND_FS = 11
ANNOTATION_FS = 11

# --- 3D layout for elongated domains ------------------------------------------
# Aspect ratio (longest side / shortest side) above which a domain is treated as
# elongated. Below it, every 3D figure keeps exactly the framing used so far, so the
# cubic (1.0) and staircase (2.0) cases are unchanged; above it the slender cases
# (300x60x60 is 5.0, a 900x140x140 domain is 6.4) get the adapted framing, which is
# what stops the z-axis label from falling outside the canvas.
ELONGATED_RATIO = 2.5

# Box zoom passed to set_box_aspect. MUST stay at 1.0 for any figure that will be
# rotated on screen. Matplotlib sizes the 3D box at zoom = 1 so that it still fits
# the axes at EVERY rotation, i.e. it reserves room for the space diagonal. On an
# elongated domain the diagonal is barely longer than the long side, so there is
# almost no slack: a zoom above one makes the box overflow the axes as soon as the
# long axis is turned towards the viewer, and Matplotlib then clips the artists at
# the axes boundary. Measured on a 900 x 140 x 140 domain, zoom = 1.6 silently
# dropped 39 of 171 lattice nodes from the view. Raise this only for a figure whose
# camera is fixed with ax.view_init, never for an interactive one.
BOX_ZOOM = 1.0


def _elongation(ext_m):
    """Longest/shortest side ratio of the domain extents."""
    e = np.asarray(ext_m, float)
    return float(e.max() / max(e.min(), 1e-12))


def _fig3d(ext_m, base=(7.2, 6.4)):
    """Create the 3D figure. Returns (fig, ax, zoom).

    The canvas keeps the base size for every domain shape. Widening it does NOT
    enlarge the drawing: Axes3D fits the box into a region derived from the axes and
    simply adds side margins, so on a 6.4:1:1 domain the plot used 87% of the width
    in a 7.2 in canvas but only 48% in a 13 in one. The empty band that an elongated
    domain leaves is removed at save time by the tight bounding box instead, which
    costs no data. See BOX_ZOOM for why the box is not enlarged either.
    """
    fig = plt.figure(figsize=base)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax, BOX_ZOOM


def _frame_3d(ax, cmin, cmax, ext_m, zoom=1.0, pad=0.06):
    """Axis limits, box aspect, projection and tick density of a 3D domain view.

    The true aspect ratio is always preserved, since compressing the long axis would
    misrepresent the domain; the framing is adapted instead. On an elongated domain
    two extra measures apply: an orthographic projection, so the far end of a long
    box does not taper away, and fewer ticks on the short axes, because the default
    locator crowds eight labels into a projected length of a few millimetres, which
    is what makes the y and z tick labels overlap.
    """
    cmn, cmx = np.asarray(cmin) * 1e3, np.asarray(cmax) * 1e3
    ext_m = np.asarray(ext_m, float)
    p = pad * ext_m * 1e3
    ax.set_xlim(cmn[0] - p[0], cmx[0] + p[0])   # anchor the view to the domain, not to
    ax.set_ylim(cmn[1] - p[1], cmx[1] + p[1])   # any residual spline excursion
    ax.set_zlim(cmn[2] - p[2], cmx[2] + p[2])
    try:
        ax.set_box_aspect(tuple(ext_m), zoom=zoom)   # zoom kwarg needs matplotlib >= 3.6
    except TypeError:
        ax.set_box_aspect(tuple(ext_m))
    if _elongation(ext_m) > ELONGATED_RATIO:
        ax.set_proj_type("ortho")
        for name, rel in zip("xyz", ext_m / ext_m.max()):
            ax.locator_params(axis=name, nbins=5 if rel > 0.5 else 3)


def _finish_3d(fig, ext_m):
    """Final layout pass: tight_layout as before, explicit margins when elongated.

    tight_layout overrides subplots_adjust, so the two cannot be combined; on an
    elongated domain the explicit margins are what keep the outermost axis label
    inside the canvas.
    """
    if _elongation(ext_m) > ELONGATED_RATIO:
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.07, top=0.93)
    else:
        fig.tight_layout()


def _save_ink_tight(fig, fname, dpi=150, pad=0.12):
    """Save the figure cropped to the pixels actually drawn.

    bbox_inches="tight" asks every artist for its bounding box, and a 3D axes reports
    the whole square region it reserves so the box still fits when rotated, not the
    band it actually draws. On an elongated domain most of that reserved region is
    empty, which is the white band under the plot: measured on a 900 x 140 x 140 case
    the saved file used only 74% of its height. Rendering once and measuring the
    non-white pixels gives the real extent, which is then handed to savefig as an
    explicit bounding box in inches (95% of the height, with nothing lost). Passing a
    Bbox rather than "tight" also works for vector output.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.transforms import Bbox
    old_canvas = fig.canvas
    try:
        FigureCanvasAgg(fig).draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    except Exception:
        fig.savefig(fname, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        return fname
    finally:
        fig.canvas = old_canvas          # keep any interactive window usable
    ink = buf.min(axis=2) < 250          # the light 3D panes count as content
    if not ink.any():
        fig.savefig(fname, dpi=dpi)
        return fname
    rows, cols = np.where(ink)
    h, w = ink.shape
    d = float(fig.get_dpi())
    bbox = Bbox([[max(cols.min() / d - pad, 0.0),
                  max((h - 1 - rows.max()) / d - pad, 0.0)],        # buffer rows run
                 [min((cols.max() + 1) / d + pad, w / d),           # top-down, figure
                  min((h - rows.min()) / d + pad, h / d)]])         # inches bottom-up
    fig.savefig(fname, dpi=dpi, bbox_inches=bbox)
    return fname


def _apply_manuscript_typography(ax):
    """Apply the manuscript font sizes to 2D and 3D axes."""
    is_3d = hasattr(ax, "zaxis")

    axis_fs = AXIS_3D_FS if is_3d else AXIS_FS
    tick_fs = TICK_3D_FS if is_3d else TICK_FS

    ax.title.set_fontsize(TITLE_FS)

    ax.xaxis.label.set_fontsize(axis_fs)
    ax.yaxis.label.set_fontsize(axis_fs)

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=tick_fs,
    )

    if is_3d:
        ax.zaxis.label.set_fontsize(axis_fs)
        ax.tick_params(
            axis="z",
            which="both",
            labelsize=tick_fs,
        )

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_FS)


def plot_convergence(result, fname, show=False):
    """Input-power evolution over the optimization (winning seed).

    Two panels sharing the iteration axis. The main panel has three curves: the
    best-so-far objective (running minimum, the envelope that only ratchets down), the
    iteration-best (lowest objective among that iteration's feasible ants, the real
    convergence signal), and the iteration mean (how much the colony still explores).
    All three carry NaN on iterations with no feasible ants; markers keep isolated
    feasible iterations visible where a plain line would vanish. Vertical dashed lines
    mark stagnation restarts (the jumps in the iteration-best right after them are
    deliberate re-exploration, not instability). When the warm-start seed sets the
    initial best, a diamond labels it: a best-so-far line pinned well below the colony
    from iteration 1 then reads as "the seeded route was never beaten", which is a
    statement about the seed, not a failure of the search. The bottom panel shows the
    feasible-ant count per iteration, so sparse-feasibility stretches on refined grids
    explain the gaps above. The objective axis is logarithmic only when the plotted
    values span more than one decade; on a narrower spread a log axis compresses the
    curves into a thin band with almost no ticks, so a linear axis is used instead.
    Iterations are 1-based; the values are the band-integrated objective the driver
    prints."""
    import numpy as np
    best = np.asarray(result["best_hist"], float)
    ibest = np.asarray(result.get("iter_best_hist", []), float)
    mean = np.asarray(result.get("mean_hist", []), float)
    feas = np.asarray(result.get("feasible_hist", []), float)
    resets = [int(k) + 1 for k in result.get("reset_iters", [])]   # to 1-based
    seeded = bool(result.get("initial_best_from_seed", False))
    q0h = np.array([rec["q0"] for rec in result.get("iter_log", [])], float)
    n = len(best)
    it = np.arange(1, n + 1)

    have_bars = feas.size == n and feas.size
    if have_bars:
        fig, (ax, axf) = plt.subplots(2, 1, figsize=(7.5, 4.8), sharex=True,
                                      gridspec_kw={"height_ratios": [4, 1]},
                                      layout="constrained")
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
        axf = None

    # log only if the finite plotted values genuinely span more than a decade
    finite = np.concatenate([v[np.isfinite(v)] for v in (best, ibest, mean)]) if n else np.array([])
    finite = finite[finite > 0]
    use_log = finite.size > 1 and (finite.max() / finite.min() > 10.0)
    if use_log:
        ax.set_yscale("log")

    for k in resets:
        ax.axvline(k, color="0.55", ls="--", lw=0.9, zorder=2)
        if axf is not None:
            axf.axvline(k, color="0.55", ls="--", lw=0.9, zorder=2)
    if resets:
        ax.plot([], [], color="0.55", ls="--", lw=0.9, label="stagnation restart")

    if mean.size == n and np.isfinite(mean).any():
        ax.plot(it, mean, "-", color="#9bb4c9", lw=1.0, marker=".", ms=3,
                label="iteration mean", zorder=3)
    if ibest.size == n and np.isfinite(ibest).any():
        ax.plot(it, ibest, "-o", color="#2a7d2a", lw=1.3, ms=3,
                label="iteration best", zorder=4)
    if np.isfinite(best).any():
        ax.plot(it, best, "-", color="#b3122c", lw=2.2, label="best so far", zorder=5)
        k0 = int(np.where(np.isfinite(best))[0][0])
        k1 = int(np.where(np.isfinite(best))[0][-1])
        if seeded:
            ax.plot(it[k0], best[k0], "D", color="#b3122c", ms=6, mfc="white",
                    zorder=6, label="warm start")
        else:
            ax.plot(it[k0], best[k0], "o", color="#b3122c", ms=6, zorder=6)
        # initial objective: labelled at the first feasible point. It is placed to the
        # RIGHT of the marker (the point sits on the left edge, so a left-anchored label
        # would run off the axes) and BELOW the curve, which drops from here, so the
        # text falls into empty space instead of onto the line.
        ax.annotate("%.3e" % best[k0], (it[k0], best[k0]),
                    textcoords="offset points", xytext=(8, -12), ha="left", va="top",
                    fontsize=ANNOTATION_FS, color="#b3122c")
        ax.plot(it[k1], best[k1], "o", color="#b3122c", ms=6, zorder=6)
        ax.annotate("%.3e" % best[k1], (it[k1], best[k1]),
                    textcoords="offset points", xytext=(-6, 9), ha="right",
                    fontsize=ANNOTATION_FS, color="#b3122c")
    else:
        ax.text(0.5, 0.55, "no feasible route found", transform=ax.transAxes,
                ha="center", fontsize=ANNOTATION_FS, color="0.35")

    ax.set_ylabel(r"band-integrated input power  [W$\cdot$Hz]")
    # no in-figure title: the manuscript caption describes the panel
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, columnspacing=1.6,
              handlelength=2.2, borderaxespad=0.0)
    ax.set_xlim(0.5, max(2, n) + 0.5)

    if axf is not None:
        axf.bar(it, feas, width=0.85, color="0.80", edgecolor="none", zorder=3)
        axf.set_ylim(0, max(1.0, feas.max() * 1.15))
        axf.set_ylabel("feasible\nants", fontsize=AXIS_FS, color="0.35")
        axf.tick_params(labelsize=TICK_FS)
        axf.grid(True, axis="y", ls=":", alpha=0.4)
        if q0h.size == n:
            axq = axf.twinx()
            axq.plot(it, q0h, "-", color="#b3122c", lw=1.2, alpha=0.8, zorder=4)
            axq.set_ylim(0.0, 1.0)
            axq.set_yticks([0.0, 0.5, 1.0])
            axq.set_ylabel(r"$q_0$", fontsize=AXIS_FS, color="#b3122c",
                           rotation=0, labelpad=8)
            axq.tick_params(labelsize=TICK_FS, colors="#b3122c")
            axq.spines["right"].set_color("#b3122c")
            axf.plot([], [], "-", color="#b3122c", lw=1.2, label=r"$q_0(t)$")
            axf.legend(frameon=False, fontsize=LEGEND_FS, loc="upper left")
        axf.set_xlabel("iteration")
    else:
        ax.set_xlabel("iteration")

    _apply_manuscript_typography(ax)
    if axf is not None:
        _apply_manuscript_typography(axf)
    fig.savefig(fname, dpi=140)
    if not show:
        plt.close(fig)


def _draw_grid(ax, grid, alpha=0.30):
    # lattice nodes: large and dark enough to survive print scaling in the paper,
    # gray so the colored routes still dominate visually
    c = grid["coords"]
    ax.scatter(c[:, 0]*1e3, c[:, 1]*1e3, c[:, 2]*1e3,
               s=14, c="0.45", alpha=alpha, depthshade=False)


def plot_best_path(grid, result, fname, start_id, target_id):
    """3D view of the best tube path on the lattice."""
    cmin = grid["coords"].min(0)
    cmax = grid["coords"].max(0)
    ext_m = np.where((cmax - cmin) > 0, cmax - cmin, 1.0)
    fig, ax, _zoom = _fig3d(ext_m, base=(6.2, 5.6))
    _draw_grid(ax, grid, alpha=0.38)
    pc = path_coords(grid, result["best_path"]) * 1e3
    ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], "-", color="#b3122c", lw=3.5)
    A = grid["coords"][start_id]*1e3
    B = grid["coords"][target_id]*1e3
    ax.scatter(*A, s=90, c="#f0a800", edgecolor="k", zorder=5, label="A (drive)")
    ax.scatter(*B, s=90, c="#1b7837", edgecolor="k", zorder=5, label="B (support)")
    ax.set_title(f"Best path | power={result['best_power']:.3e} | "
                 f"L={result['best_length']*1e3:.0f} mm")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.legend(frameon=False, loc="upper left")
    _apply_manuscript_typography(ax)
    _frame_3d(ax, cmin, cmax, ext_m, zoom=_zoom)
    _finish_3d(fig, ext_m)
    _save_ink_tight(fig, fname, dpi=140)
    plt.close(fig)


def plot_snapshots(grid, result, fname, start_id, target_id, n_panels=4):
    """Panels of all ant paths at increasing iterations (many paths -> one)."""
    snaps = result["snapshots"]
    if not snaps:
        return
    idx = np.linspace(0, len(snaps) - 1, n_panels).astype(int)
    fig = plt.figure(figsize=(4.2 * n_panels, 4.4))
    A = grid["coords"][start_id]*1e3
    B = grid["coords"][target_id]*1e3
    for p, si in enumerate(idx):
        snap = snaps[si]
        ax = fig.add_subplot(1, n_panels, p + 1, projection="3d")
        _draw_grid(ax, grid, alpha=0.25)
        cmap = plt.cm.turbo
        for q, path in enumerate(snap["paths"]):
            pc = path_coords(grid, path) * 1e3
            ax.plot(pc[:, 0], pc[:, 1], pc[:, 2],
                    color=cmap((q + 1) / (len(snap["paths"]) + 1)),
                    lw=1.0, alpha=0.55)
        if snap["best"]:
            pc = path_coords(grid, snap["best"]) * 1e3
            ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], color="k", lw=2.6)
        ax.scatter(*A, s=45, c="#f0a800", edgecolor="k", zorder=5)
        ax.scatter(*B, s=45, c="#1b7837", edgecolor="k", zorder=5)
        ax.set_title(f"iter {snap['iter']}  ({len(snap['paths'])} ants)")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        _apply_manuscript_typography(ax)
    fig.suptitle("Ant paths over iterations (colony converges to one path)",
                 y=1.02, fontsize=TITLE_FS)
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_baseline(mc_powers, aco_power, diagonal_power, fname):
    """Histogram of random feasible paths, with the ACO best and the straight
    diagonal marked, to show how good the optimized result really is."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(mc_powers, bins=30, color="#9bb8d3", edgecolor="white",
            label=f"random feasible paths (n={len(mc_powers)})")
    ax.axvline(diagonal_power, color="#2a4d69", lw=2.0, ls="--",
               label=f"straight diagonal = {diagonal_power:.2e}")
    ax.axvline(aco_power, color="#b3122c", lw=2.6,
               label=f"ACO best = {aco_power:.2e}")
    ax.set_xlabel("band-integrated input power  [W·Hz]")
    ax.set_ylabel("count")
    ax.set_title("How good is the optimized pipe? (vs random and diagonal)")
    ax.legend(frameon=False, fontsize=LEGEND_FS)
    _apply_manuscript_typography(ax)
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def plot_spline(grid, waypoint_ids, tck, nodes, fname, start_id, target_id,
                r_min=None, r_min_pt=None, show=False, keepout=None,
                baseline_pts=None, baseline_label="baseline",
                title=None, legend=True, show_nodes=False):
    """3D view of the optimized route: lattice polyline, as-built spline and
    (optionally) the beam nodes and the baseline route."""
    import geometry as geo
    import matplotlib.patheffects as pe
    cmin = grid["coords"].min(0)
    cmax = grid["coords"].max(0)
    ext_m = np.where((cmax - cmin) > 0, cmax - cmin, 1.0)
    fig, ax, _zoom = _fig3d(ext_m, base=(6.4, 5.8))
    ax.computed_zorder = False          # honour the explicit zorder of each artist below

    # lattice as a faint background: small, nearly transparent nodes
    c = grid["coords"] * 1e3
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=5, c="0.55", alpha=0.12,
               depthshade=False, zorder=1)

    if keepout is not None:
        kb = keepout if isinstance(keepout, list) else [keepout]
        for bx in kb:
            clo = np.maximum(np.asarray(bx[0]), cmin)
            chi = np.minimum(np.asarray(bx[1]), cmax)
            _add_box(ax, (clo, chi))
        ax.plot([], [], [], color=OBSTACLE_EDGE, lw=OBSTACLE_EDGE_LW, label="keep-out")

    if baseline_pts is not None:
        bl = geo.eval_spline(geo.fit_spline(np.asarray(baseline_pts), smooth=0.0,
                                            end_tangents=None), 300) * 1e3
        ax.plot(bl[:, 0], bl[:, 1], bl[:, 2], "--", color="#f0a800", lw=1.4,
                alpha=0.9, label=baseline_label, zorder=3)

    wp = path_coords(grid, waypoint_ids) * 1e3
    ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], "--", color="#2a4d69", lw=1.0,
            marker="o", ms=3.5, label="ACO route (waypoints)", zorder=4)

    curve = geo.eval_spline(tck, 400) * 1e3
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "-", color="#b3122c", lw=3.0,
            label="smooth spline", zorder=6,
            path_effects=[pe.withStroke(linewidth=5.5, foreground="white")])

    if show_nodes:
        nm = nodes * 1e3
        ax.scatter(nm[:, 0], nm[:, 1], nm[:, 2], s=4, c="#1b7837", alpha=0.9,
                   zorder=7, label=f"beam nodes (n={len(nodes)})")
    # waypoints on top of the spline: the lattice vertices the centreline passes through
    ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], s=14, c="#2a4d69",
               edgecolor="white", linewidth=0.4, zorder=7)

    A = grid["coords"][start_id] * 1e3
    B = grid["coords"][target_id] * 1e3
    ax.scatter(*A, s=80, c="#f0a800", edgecolor="k", zorder=8)
    ax.scatter(*B, s=80, c="#1b7837", edgecolor="k", zorder=8)
    if r_min_pt is not None:
        p = np.asarray(r_min_pt) * 1e3
        ax.scatter(*p, s=70, marker="x", c="k", zorder=9,
                   label=f"tightest bend R={r_min*1e3:.0f} mm" if r_min else None)

    if title:
        ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    if legend:
        ax.legend(frameon=False, loc="upper left", fontsize=LEGEND_FS)
    _apply_manuscript_typography(ax)
    _frame_3d(ax, cmin, cmax, ext_m, zoom=_zoom)
    _finish_3d(fig, ext_m)
    _save_ink_tight(fig, fname, dpi=140)
    if not show:
        plt.close(fig)


OBSTACLE_EDGE = "#7b1fa2"      # single visual identity for obstacle outlines, used by
OBSTACLE_EDGE_LW = 1.8         # _add_box and the legend proxies


def _add_box(ax, box, color="#8a8a8a", edge=OBSTACLE_EDGE, alpha=0.20,
             edge_lw=OBSTACLE_EDGE_LW):
    """Draw a translucent rectangular keep-out region. box = (lo, hi) in metres.

    Two version-portability notes, learned the hard way. (1) The face alpha is baked
    into the RGBA facecolors instead of being passed as a separate ``alpha`` kwarg:
    Poly3DCollection re-derives its colours during the 3D projection and, on some
    matplotlib versions, a separate alpha is dropped there, which rendered the faces
    fully opaque. (2) The edges are drawn as a second Poly3DCollection with empty
    faces rather than as individual ax.plot lines: modern Axes3D defaults to
    computed_zorder=True, which orders artists by mean projected depth and ignores an
    explicit zorder, so on a thin slab the translucent faces tied with, and were
    painted over, the edge lines. The edge collection shares the exact face geometry,
    hence the exact depth statistics, and being added after the faces it wins the
    stable-sort tie in every view."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgba
    lo = np.asarray(box[0]) * 1e3
    hi = np.asarray(box[1]) * 1e3
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
             [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
             [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=to_rgba(color, alpha),
                                         edgecolors="none", zsort="average"))
    ax.add_collection3d(Poly3DCollection(faces, facecolors=(0, 0, 0, 0),
                                         edgecolors=to_rgba(edge, 1.0),
                                         linewidths=edge_lw, zsort="average"))


def plot_all_splines(grid, evaluated_paths, best_path, fname, start_id, target_id,
                     smooth=0.0, boxes=None, max_curves=None, baseline_pts=None,
                     baseline_label="baseline", show=False):
    """3D plot of every tested route as a faint spline, with the best one solid.

    ``evaluated_paths`` is the list of routes the search actually evaluated (the
    feasible, FEM-evaluated ones, e.g. result["evaluated_paths"]); each is splined the
    same way as the winner and drawn with low opacity, so the figure shows the
    geometric envelope explored and where the best route (solid) sits within it.
    Pass ``smooth`` equal to the driver's SMOOTH so the curves match what was tested,
    and ``boxes`` (a list of (lo, hi) keep-out boxes) to shade forbidden regions.
    ``max_curves`` optionally subsamples when there are very many routes.
    """
    import geometry as geo
    cmin = grid["coords"].min(0)
    cmax = grid["coords"].max(0)
    ext_m = np.where((cmax - cmin) > 0, cmax - cmin, 1.0)
    fig, ax, _zoom = _fig3d(ext_m)
    _draw_grid(ax, grid, alpha=0.08)
    if boxes:
        for bx in boxes:                               # clip each keep-out box to the domain so
            clo = np.maximum(np.asarray(bx[0]), cmin)  # it does not stick out past the lattice
            chi = np.minimum(np.asarray(bx[1]), cmax)
            _add_box(ax, (clo, chi))
        ax.plot([], [], [], color=OBSTACLE_EDGE, lw=OBSTACLE_EDGE_LW, label="blocked region")

    bp = tuple(best_path)
    paths = [tuple(p) for p in evaluated_paths if tuple(p) != bp]
    if max_curves and len(paths) > max_curves:
        idx = np.linspace(0, len(paths) - 1, max_curves).astype(int)
        paths = [paths[i] for i in idx]

    lo_box = cmin - 0.35 * ext_m          # a little overshoot outside the lattice is fine
    hi_box = cmax + 0.35 * ext_m          # but a spline that swings far out is an artifact

    _alpha = float(np.clip(12.0 / max(len(paths), 1), 0.09, 0.35))  # fainter when there are many
    n_drawn = 0
    n_skip = 0
    for p in paths:
        try:
            curve_m = geo.eval_spline(geo.fit_spline(path_coords(grid, list(p)), smooth=smooth), 60)
        except Exception:
            continue
        if (curve_m < lo_box).any() or (curve_m > hi_box).any():
            n_skip += 1                   # interpolating-spline overshoot far outside the domain
            continue
        c = curve_m * 1e3
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color="#2a4d69", alpha=_alpha, lw=0.8)
        n_drawn += 1

    curveb = geo.eval_spline(geo.fit_spline(path_coords(grid, list(best_path)), smooth=smooth), 200) * 1e3
    ax.plot(curveb[:, 0], curveb[:, 1], curveb[:, 2], color="#b3122c", lw=2.8,
            label="best route", zorder=6)
    ax.plot([], [], [], color="#2a4d69", alpha=0.6, lw=1.4, label="tested routes (n=%d)" % n_drawn)

    if baseline_pts is not None:
        bl = geo.eval_spline(geo.fit_spline(np.asarray(baseline_pts), smooth=smooth, end_tangents=None), 300) * 1e3
        ax.plot(bl[:, 0], bl[:, 1], bl[:, 2], color="#f0a800", lw=2.2, ls="--",
                label=baseline_label, zorder=5)

    A = grid["coords"][start_id] * 1e3
    B = grid["coords"][target_id] * 1e3
    ax.scatter(*A, s=80, c="#f0a800", edgecolor="k", zorder=7, label="A (drive)")
    ax.scatter(*B, s=80, c="#1b7837", edgecolor="k", zorder=7, label="B (clamped)")

    ax.set_title("All tested routes (faint) and the best (solid)")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    _frame_3d(ax, cmin, cmax, ext_m, zoom=_zoom)
    ax.legend(frameon=False, loc="upper left", fontsize=LEGEND_FS)
    _apply_manuscript_typography(ax)
    _finish_3d(fig, ext_m)
    _save_ink_tight(fig, fname, dpi=150)
    if not show:
        plt.close(fig)
    if n_skip:
        print("  (%d tested route(s) had large spline overshoot, left out of the view)" % n_skip)
    return fname, n_drawn


def plot_domain(grid, fname, start_id, target_id, boxes=None,
                baseline_pts=None, baseline_label="baseline",
                smooth=0.0, title=None, show=False):
    """3D view of the design domain itself: lattice nodes, endpoints and obstacles.

    With ``baseline_pts=None`` the figure shows the bare domain (plus any keep-out
    boxes clipped to the lattice), so each case can be presented before any route is
    drawn. Passing ``baseline_pts`` adds only the deterministic reference curve (the
    open helix or the serpentine from baseline_routes), splined with the same
    ``smooth`` used in the optimization, so the naive space-filling route is shown on
    the otherwise empty domain, without the cloud of tested ACO splines. The framing
    (axis limits, padding and aspect) matches plot_all_splines so the domain-only,
    baseline-only and all-splines figures of one case overlay consistently.
    """
    import geometry as geo
    cmin = grid["coords"].min(0)
    cmax = grid["coords"].max(0)
    ext_m = np.where((cmax - cmin) > 0, cmax - cmin, 1.0)
    fig, ax, _zoom = _fig3d(ext_m)
    _draw_grid(ax, grid, alpha=0.35)          # nothing else competes here, so darker
    if boxes:
        for bx in boxes:                               # clip each keep-out box to the domain so
            clo = np.maximum(np.asarray(bx[0]), cmin)  # it does not stick out past the lattice
            chi = np.minimum(np.asarray(bx[1]), cmax)
            _add_box(ax, (clo, chi))
        ax.plot([], [], [], color=OBSTACLE_EDGE, lw=OBSTACLE_EDGE_LW, label="blocked region")

    if baseline_pts is not None:
        bl = geo.eval_spline(geo.fit_spline(np.asarray(baseline_pts), smooth=smooth, end_tangents=None), 300) * 1e3
        ax.plot(bl[:, 0], bl[:, 1], bl[:, 2], color="#f0a800", lw=2.4, ls="--",
                label=baseline_label, zorder=5)   # same identity as in plot_all_splines

    A = grid["coords"][start_id] * 1e3
    B = grid["coords"][target_id] * 1e3
    ax.scatter(*A, s=80, c="#f0a800", edgecolor="k", zorder=7, label="A (drive)")
    ax.scatter(*B, s=80, c="#1b7837", edgecolor="k", zorder=7, label="B (clamped)")

    if title is None:
        title = ("Design domain and lattice" if baseline_pts is None
                 else "Design domain with the reference baseline")
    ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    _frame_3d(ax, cmin, cmax, ext_m, zoom=_zoom)
    ax.legend(frameon=False, loc="upper left", fontsize=LEGEND_FS)
    _apply_manuscript_typography(ax)
    _finish_3d(fig, ext_m)
    _save_ink_tight(fig, fname, dpi=150)
    if not show:
        plt.close(fig)
    return fname


def show_interactive():
    """Open any figures kept open (e.g. plot_spline with show=True) in a window.

    On a desktop it blocks until the window is closed and the 3D view can be
    rotated and zoomed with the mouse. If no interactive backend is available
    (Matplotlib fell back to Agg), it prints how to enable one instead of failing.

    While a 3D window is open, releasing the mouse prints the current view angles as
    a ready-to-paste ax.view_init(...) call: rotating by hand and saving from the
    toolbar produces a figure that cannot be regenerated, so the angle that looks
    right should be recorded and fixed in the plotting function.
    """
    if matplotlib.get_backend().lower() == "agg":
        print("  [No interactive Matplotlib backend was found, so the window cannot open.")
        print("   The figure was still saved to PNG. To rotate it on screen, install a GUI")
        print("   backend, e.g.  pip install PyQt5  (or ensure Python's tkinter is present),")
        print("   then run again.]")
        return
    for num in plt.get_fignums():
        f = plt.figure(num)
        for a in f.axes:
            if hasattr(a, "zaxis"):
                f.canvas.mpl_connect(
                    "button_release_event",
                    lambda ev, a=a: print("  ax.view_init(elev=%.1f, azim=%.1f, roll=%.1f)"
                                          % (a.elev, a.azim, getattr(a, "roll", 0.0))))
                break
    def _save_current(ev):
        if ev.key != "i":
            return
        fname = "view_%d.pdf" % ev.canvas.figure.number
        _save_ink_tight(ev.canvas.figure, fname, dpi=300)
        print("  saved current view to %s (cropped to the drawn content)" % fname)
    for num in plt.get_fignums():
        plt.figure(num).canvas.mpl_connect("key_press_event", _save_current)
    print("  [rotate with the mouse; press 'i' to save the current view cropped]")
    try:
        plt.show()
    except Exception as exc:
        print(f"  [Could not open an interactive window: {exc}.")
        print("   The figure was saved to PNG; install a GUI backend (pip install PyQt5) to rotate it.]")


def plot_frf(grid, result, fname, mat, sec, band_hz, evaluate_spectrum,
             freq_range=(0.0, 10000.0), baseline_coords=None,
             baseline_label="baseline", show=False, baseline_spectrum=None):
    """Input-power FRF of the best path, with a FIXED frequency axis.

    freq_range : (f_lo, f_hi) in Hz for the x-axis. Defaults to (0, 10000) so the
    plot always spans the same range regardless of the target band. The sweep starts
    just above 0 because the active input power is exactly zero at f = 0 (the omega
    prefactor), which cannot be shown on the log y-axis.

    baseline_coords : optional (M,3) route (the reference baseline centreline). When
    given, its spectrum is computed through the same callback and drawn as a dashed
    reference line, so the optimized tube can be read against the naive route across
    the whole band and not only through the single band-integrated number.
    """
    import numpy as np
    f_lo, f_hi = freq_range
    f_start = max(f_lo, 1.0)
    freqs = np.linspace(f_start, f_hi, 400)
    W = evaluate_spectrum(path_coords(grid, result["best_path"]), freqs)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogy(freqs, np.abs(W), color="#b3122c", lw=2.0, label="optimized route", zorder=4)
    if baseline_coords is not None:
        # the baseline may need its own evaluator (e.g. a spline fit with free end
        # tangents when the optimized route has prescribed exit/arrival directions)
        Wb = (baseline_spectrum or evaluate_spectrum)(np.asarray(baseline_coords), freqs)
        ax.semilogy(freqs, np.abs(Wb), color="#f0a800", lw=1.9, ls="--",
                    label=baseline_label, zorder=3)
    _bands = [band_hz] if np.isscalar(band_hz[0]) else list(band_hz)
    for _i, _b in enumerate(_bands):
        ax.axvspan(_b[0], _b[1], color="0.85", label="target band" if _i == 0 else None)
    ax.set_xlim(f_lo, f_hi)                      # always 0 to 10000 Hz
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$W_\text{in}(f)$ [W]")
    # no in-figure title: the manuscript caption describes the panel
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(frameon=False)
    _apply_manuscript_typography(ax)
    fig.savefig(fname, dpi=140)
    if not show:
        plt.close(fig)


def plot_radius_profile(tck, fname, r_min=None, show=False):
    """Radius of curvature along the tube arc length, with the manufacturability floor.

    The dashed line is the minimum bend radius R_min; the tube is buildable when the
    radius stays ABOVE it everywhere. The tightest bend (smallest radius) is marked.
    Nearly straight segments have a very large radius, so the y-axis is logarithmic and
    capped at a few times R_min: anything straighter than the cap is left off the top.
    """
    import numpy as np
    from scipy.interpolate import splev
    import geometry as geo
    kappa, u = geo.curvature(tck, n=800)
    pts = np.array(splev(u, tck)).T
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]) * 1e3
    with np.errstate(divide="ignore"):
        R_mm = np.where(kappa > 1e-9, 1e3 / kappa, np.inf)   # mm; inf on straight parts
    finite = R_mm[np.isfinite(R_mm)]
    rmn = (r_min * 1e3) if r_min else (float(np.min(finite)) if finite.size else 1.0)
    top = 20.0 * rmn                                          # display cap: a few x R_min
    R_disp = np.where(R_mm <= top, R_mm, np.nan)              # gap out the near-straight parts
    ip = int(np.nanargmin(R_mm))                             # tightest bend = smallest radius
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(s, R_disp, color="#b3122c", lw=2.0, label="radius of curvature")
    if r_min:
        ax.axhline(rmn, color="0.4", ls="--", lw=1.3,
                   label=r"$R_{\min}$ = %.0f mm (manufacturing limit)" % rmn)
    ax.plot(s[ip], R_mm[ip], "kx", ms=9, label="tightest: R = %.0f mm" % R_mm[ip])
    ax.set_yscale("log")
    ax.set_ylim(0.7 * rmn, top)
    ax.set_xlim(s[0], s[-1])
    ax.set_xlabel("arc length [mm]")
    ax.set_ylabel("radius of curvature [mm]")
    ax.set_title("Bend-radius profile along the pipe")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=LEGEND_FS)
    _apply_manuscript_typography(ax)
    fig.savefig(fname, dpi=140)
    if not show:
        plt.close(fig)
    return fname