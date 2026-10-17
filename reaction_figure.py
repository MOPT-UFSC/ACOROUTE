"""Far-end reaction spectra for a final tube design, drawn like the input-power FRF.

Node B is clamped, so its reactions are the blocked force and moment transmitted to
the downstream structure. This module evaluates the six reaction components at B over
frequency and draws two figures in the same style as the input-power FRF (fixed 0-10
kHz axis, the target analysis band shaded): one with the three reaction forces
Fx, Fy, Fz, and one with the three reaction moments Mx, My, Mz. Plotting these next to
the input power shows whether minimising the input power also lowers what reaches B.

The reactions at B are already formed inside the prescribed-displacement solver (they
are part of the reaction vector R); fem.input_power_spectrum_prescribed with
return_reactions=True simply hands them back.

Typical use on a final design (nodes, elements from geometry.mesh_spline /
geometry.resample_by_length):
    import reaction_figure as RF
    RF.compute_and_plot(nodes, elements, mat, sec, BAND,
                        rayleigh=RAY, prescribed_disp=PRESCRIBED_DISP,
                        prescribed_mask=PRESCRIBED_MASK)
"""
import os
import sys
import numpy as np
import matplotlib
# Only force the non-interactive Agg backend on a headless Linux box, so this
# module does not override an interactive backend chosen on a desktop.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

import plot_style  # noqa: F401  (sets Times New Roman for all figures)
import fem_beam3d as fem

# colours for the x, y, z components (shared by the force and moment figures)
COMP_COLORS = ("#b3122c", "#2a4d69", "#3a7d44")
# grey shades for the same x, y, z components of the BASELINE design (dashed overlay)
GRAY_COLORS = ("0.15", "0.45", "0.70")


def spectra(nodes, elements, mat, sec, band_hz, drive_node=0, fixed_node=None,
            n_freq=400, freqs=None, rayleigh=(0.0, 1e-6),
            prescribed_disp=(1.0, 1.0, 1.0), prescribed_mask=(True, True, True),
            constrain_drive_rot=False):
    """Return (freqs, Wbar, RB): the input-power spectrum and the 6-component reaction
    spectrum at the clamped node B. RB has shape (len(freqs), 6) ordered
    [Fx, Fy, Fz, Mx, My, Mz] in the global frame. A uniform grid of n_freq points over
    the band is used unless an explicit `freqs` array is given."""
    if fixed_node is None:
        fixed_node = len(nodes) - 1
    if freqs is None:
        freqs = np.linspace(band_hz[0], band_hz[1], n_freq)
    Wbar, RB = fem.input_power_spectrum_prescribed(
        nodes, elements, mat, sec, drive_node, fixed_node, freqs,
        prescribed_disp=prescribed_disp, rayleigh=rayleigh,
        constrain_drive_rot=constrain_drive_rot, prescribed_mask=prescribed_mask,
        return_reactions=True)
    return freqs, Wbar, RB


# --- figure typography for the manuscript -------------------------------------
# One knob per role, sized for a 7.5 x 4.2 in figure placed at ~\textwidth: axis
# labels and ticks must survive the reduction to column width.
AXIS_FS = 14      # axis-label font size
TITLE_FS = 14     # title font size (reserved for consistent future figures)
TICK_FS = 12      # tick-label font size
LEGEND_FS = 11    # legend font size


def _frf_style(ax, band_hz, freq_range):
    """Apply the input-power FRF styling: fixed Hz axis and shaded target band(s)."""
    f_lo, f_hi = freq_range
    bands = [band_hz] if np.isscalar(band_hz[0]) else list(band_hz)
    for i, b in enumerate(bands):
        ax.axvspan(b[0], b[1], color="0.85", label="target band" if i == 0 else None)
    ax.set_xlim(f_lo, f_hi)
    ax.set_xlabel("frequency [Hz]", fontsize=AXIS_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.grid(True, which="both", ls=":", alpha=0.5)


def _legend_above(ax):
    """Legend in a row above the axes, outside the data area, so no spectrum peak is
    ever covered. The in-figure title is omitted for the manuscript (the LaTeX caption
    plays that role), which frees exactly this space."""
    ax.legend(frameon=False, ncol=4, fontsize=LEGEND_FS,
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              columnspacing=1.4, handlelength=1.8, handletextpad=0.5)


def plot_reaction_forces(freqs, RB, band_hz, fname="fig_reaction_forces.pdf",
                         freq_range=(0.0, 10000.0), RB_base=None,
                         base_label="baseline", show=False):
    """The three reaction-force components at B vs frequency, in the FRF style.

    If RB_base is given (same shape as RB, the reaction spectrum of the baseline
    design), its three force components are overlaid as grey dashed lines, so the
    optimized reactions can be read against the naive route across the band. The
    optimized (coloured) lines are drawn on top."""
    labels = ("$F_{B,x}$", "$F_{B,y}$", "$F_{B,z}$")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i in range(3):
        ax.semilogy(freqs, np.abs(RB[:, i]), color=COMP_COLORS[i], lw=1.9,
                    zorder=4, label=labels[i])
    if RB_base is not None:
        for i in range(3):
            ax.semilogy(freqs, np.abs(RB_base[:, i]), color=GRAY_COLORS[i], lw=1.3,
                        ls="--", zorder=2, label="%s, %s" % (labels[i], base_label))
    _frf_style(ax, band_hz, freq_range)
    ax.set_ylabel(r"$|F_B|$  [N]", fontsize=AXIS_FS)
    _legend_above(ax)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fname


def plot_reaction_torques(freqs, RB, band_hz, fname="fig_reaction_moments.pdf",
                          freq_range=(0.0, 10000.0), RB_base=None,
                          base_label="baseline", show=False):
    """The three reaction-moments components at B vs frequency, in the FRF style.

    If RB_base is given, the baseline moment components are overlaid as grey dashed
    lines, with the optimized (coloured) lines drawn on top."""
    labels = ("$M_{B,x}$", "$M_{B,y}$", "$M_{B,z}$")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i in range(3):
        ax.semilogy(freqs, np.abs(RB[:, 3 + i]), color=COMP_COLORS[i], lw=1.9,
                    zorder=4, label=labels[i])
    if RB_base is not None:
        for i in range(3):
            ax.semilogy(freqs, np.abs(RB_base[:, 3 + i]), color=GRAY_COLORS[i], lw=1.3,
                        ls="--", zorder=2, label="%s, %s" % (labels[i], base_label))
    _frf_style(ax, band_hz, freq_range)
    ax.set_ylabel(r"$|M_B|$  [N m]", fontsize=AXIS_FS)
    _legend_above(ax)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fname


def compute_and_plot(nodes, elements, mat, sec, band_hz,
                     forces_path="fig_reaction_forces.pdf",
                     torques_path="fig_reaction_moments.pdf",
                     freq_range=(0.0, 10000.0), n_freq=400,
                     base_nodes=None, base_elements=None, base_label="baseline",
                     show=False, **kw):
    """Compute the reaction spectrum at B on the input-power FRF grid (fixed 0-10 kHz,
    target band shaded) and write the force and moment figures. If base_nodes and
    base_elements are given (the baseline design meshed at the same element size), the
    baseline reaction spectrum is computed on the same grid and overlaid in grey dashed
    lines. Extra keyword arguments go to spectra(): drive_node, rayleigh,
    prescribed_disp, prescribed_mask, fixed_node, constrain_drive_rot."""
    f_lo, f_hi = freq_range
    freqs = np.linspace(max(f_lo, 1.0), f_hi, n_freq)   # match the FRF grid
    _, _, RB = spectra(nodes, elements, mat, sec, band_hz, freqs=freqs, **kw)
    RB_base = None
    if base_nodes is not None and base_elements is not None:
        _, _, RB_base = spectra(base_nodes, base_elements, mat, sec, band_hz,
                                freqs=freqs, **kw)
    pf = plot_reaction_forces(freqs, RB, band_hz, forces_path, freq_range,
                              RB_base=RB_base, base_label=base_label, show=show)
    pt = plot_reaction_torques(freqs, RB, band_hz, torques_path, freq_range,
                               RB_base=RB_base, base_label=base_label, show=show)
    return pf, pt
