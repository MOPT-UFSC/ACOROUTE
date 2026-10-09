"""Computational-cost figure for one ACO optimization.

Shared by run_spline.py (per-case runs) and make_paper_figures.py so the figure is
produced the same way everywhere. The cost data comes straight from the optimize()
result dict (feasible_hist, fem_hist, params); save_cost_data persists just those
fields so the figure can be regenerated later without re-running the search.

Typical use after an optimization:
    import cost_figure
    cost_figure.save_cost_data(result, "fig_cost_data.npz")   # persist the cost arrays
    cost_figure.make_cost_figure(result, "fig_cost.pdf")      # write the figure

Regenerate the figure later from the saved data only (no optimization):
    cost_figure.make_cost_figure("fig_cost_data.npz", "fig_cost.pdf")
"""
import numpy as np
import os
import sys
import matplotlib
# Only force the non-interactive Agg backend on a headless Linux box, so this
# module does not override an interactive backend chosen on a desktop.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style  # noqa: F401  (sets Times New Roman for all figures)

C_FEM, C_FEA = "#b3122c", "#2a4d69"

# Same typography used by the other manuscript figures.
AXIS_FS = 14
TITLE_FS = 14
TICK_FS = 12
LEGEND_FS = 11
ANNOTATION_FS = 11


def save_cost_data(result, path="fig_cost_data.npz"):
    """Persist the per-iteration cost arrays from an optimize() result dict."""
    np.savez(path,
             feasible_hist=np.asarray(result["feasible_hist"], float),
             fem_hist=np.asarray(result["fem_hist"], float),
             best_hist=np.asarray(result.get("best_hist", []), float),
             n_ants=int(result["params"]["n_ants"]))
    return path


def _arrays(source):
    """Accept either an optimize() result dict or a path to a saved npz."""
    if isinstance(source, str):
        d = np.load(source)
        return (np.asarray(d["feasible_hist"], float),
                np.asarray(d["fem_hist"], float), int(d["n_ants"]))
    return (np.asarray(source["feasible_hist"], float),
            np.asarray(source["fem_hist"], float),
            int(source["params"]["n_ants"]))


def make_cost_figure(source, path="fig_cost.pdf", show=False):
    """Per-iteration full-tube FEM solves and the cumulative cost funnel.

    `source` is an optimize() result dict or a path to a saved npz. Each iteration
    launches a fixed number of ants; some reach the target feasibly, and each
    distinct feasible route costs one full-tube FEM solve (repeats are cached).
    """
    feas, femc, n_ants = _arrays(source)
    T = len(feas)
    it = np.arange(1, T + 1)
    new_fem = np.diff(np.r_[0.0, femc])                 # new FEM solves per iteration
    launched = np.full(T, n_ants, float)
    cum_launch = np.cumsum(launched)
    cum_feas = np.cumsum(feas)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    # (a) per-iteration workload
    a1.bar(it, new_fem, color=C_FEM, alpha=0.85, width=0.8, label="full-pipe FEM solves", zorder=3)
    a1.plot(it, feas, "-o", color=C_FEA, lw=1.5, ms=3.5, label="feasible routes built", zorder=4)
    a1.plot(it, launched, "--", color="0.5", lw=1.4, label="ants launched", zorder=2)
    a1.set_xlabel("iteration", fontsize=AXIS_FS)
    a1.set_ylabel("routes per iteration", fontsize=AXIS_FS)
    a1.set_ylim(0, n_ants + 2.5)
    a1.set_title("Per-iteration cost: full-pipe evaluations",
                 fontsize=TITLE_FS, pad=42)
    a1.legend(frameon=False, fontsize=LEGEND_FS, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.01),
              borderaxespad=0.0, columnspacing=1.0, handletextpad=0.5)
    a1.tick_params(axis="both", which="both", labelsize=TICK_FS)
    a1.grid(True, axis="y", ls=":", alpha=0.4)
    a1.set_xlim(0.3, T + 0.7)
    # (b) cumulative funnel
    a2.plot(it, cum_launch, "--", color="0.5", lw=1.8, label="ants launched")
    a2.fill_between(it, cum_feas, cum_launch, color="0.5", alpha=0.12)
    a2.plot(it, cum_feas, "-", color=C_FEA, lw=2.0, label="feasible routes built")
    a2.fill_between(it, femc, cum_feas, color=C_FEA, alpha=0.10)
    a2.plot(it, femc, "-", color=C_FEM, lw=2.6, label="distinct FEM solves")
    for y, c in ((cum_launch[-1], "0.4"), (cum_feas[-1], C_FEA), (femc[-1], C_FEM)):
        a2.annotate("%d" % int(y), (T, y), textcoords="offset points", xytext=(6, 0),
                    va="center", color=c, fontsize=ANNOTATION_FS, fontweight="bold")
    a2.set_xlabel("iteration", fontsize=AXIS_FS)
    a2.set_ylabel("cumulative count", fontsize=AXIS_FS)
    a2.set_xlim(1, T + 1.5)
    a2.set_title("Cumulative cost over the run", fontsize=TITLE_FS, pad=42)
    a2.legend(frameon=False, fontsize=LEGEND_FS, ncol=3,
              loc="lower left", bbox_to_anchor=(0.0, 1.01),
              borderaxespad=0.0, columnspacing=1.0, handletextpad=0.5)
    a2.tick_params(axis="both", which="both", labelsize=TICK_FS)
    a2.grid(True, ls=":", alpha=0.4)
    fig.suptitle("Computational cost of one optimization "
                 "(%d ants, %d iterations, %d full-pipe FEM solves)"
                 % (n_ants, T, int(femc[-1])), fontsize=TITLE_FS, y=1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return path
