"""
compare_ansys.py
Comparison of the Ansys BEAM188 harmonic analysis with the ACOROUTE beam model for the
reported staircase route (6-10 kHz), same mesh, properties, damping and boundary conditions.

Inputs (same folder):
    stair_6_10_ansys_A.txt      f, Re/Im of FX FY FZ at the driven node A      (from Ansys)
    stair_6_10_ansys_B.txt      f, Re/Im of FX FY FZ MX MY MZ at the clamp B   (from Ansys)
    stair_6_10_ansys_modes.txt  natural frequencies                            (from Ansys, optional)
    stair_6_10_nodes.txt        beam nodes of the shared mesh
ACOROUTE side: fem_beam3d.py is taken from the parent folder (the repository root), from the
current folder or from ACOROUTE_DIR below, and the direct solver is run at the Ansys
frequencies. If it cannot be imported, the precomputed files acoroute_stair_6_10_direct.txt
and acoroute_stair_6_10_modes.txt (written by make_ansys_package.py) are used, which requires
the frequency list of the APDL input as generated.

    python compare_ansys.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ACOROUTE_DIR = None                       # e.g. r"C:\work\ACOROUTE"; None: current folder / PYTHONPATH

# model data (must match the APDL input)
E, NU, RHO = 210e9, 0.3, 7850.0
OUTER_RADIUS, THICKNESS = 0.001575, 0.0005
RAY = (0.0, 2e-7)
U_A = np.array([1e-6, 1e-6, 1e-6], dtype=complex)   # prescribed displacement phasors at A [m]
BAND = (6000.0, 10000.0)                  # design band of the case
SUB_BANDS = [(0.0, 1000.0), (1000.0, 3000.0), (3000.0, 6000.0), (6000.0, 10000.0), (0.0, 10000.0)]
F_MODAL_MAX = 12000.0


def load_ansys_A(path):
    d = np.loadtxt(path)
    f = d[:, 0]
    F = d[:, 1::2] + 1j * d[:, 2::2]
    return f, F


def load_ansys_B(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1::2] + 1j * d[:, 2::2]


def input_power_from_reactions(f, F, u):
    """Active input power 0.5*w*Re{i F^H u} for e^{iwt}, with F the reactions at the driven DOFs."""
    w = 2.0 * np.pi * f
    return 0.5 * w * np.real(1j * (np.conj(F) @ u))


def try_import_fem():
    for p in [ACOROUTE_DIR, os.path.dirname(HERE), os.getcwd(), HERE]:
        if p and os.path.isfile(os.path.join(p, "fem_beam3d.py")) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        import fem_beam3d as fem
        return fem
    except ImportError:
        return None


def acoroute_solution(fem, f):
    """Returns (W, RB, natural frequencies, source label)."""
    if fem is not None:
        nodes = np.loadtxt(os.path.join(HERE, "stair_6_10_nodes.txt"))
        n = len(nodes)
        elems = np.column_stack([np.arange(n - 1), np.arange(1, n)])
        mat = {"E": E, "G": E / (2.0 * (1.0 + NU)), "rho": RHO}
        sec = fem.pipe_section(outer_radius=OUTER_RADIUS, thickness=THICKNESS, nu=NU)
        KM = fem.assemble(nodes, elems, mat, sec)
        W, RB = fem.input_power_spectrum_prescribed(nodes, elems, mat, sec, 0, n - 1, f, prescribed_disp=tuple(U_A), rayleigh=RAY, return_reactions=True, KM=KM)
        fn = fem.natural_frequencies(nodes, elems, mat, sec, n - 1, extra_fixed_dofs=[0, 1, 2], KM=KM)
        return W, RB, fn[fn <= F_MODAL_MAX], "fem_beam3d direct solver at the Ansys frequencies"
    d = np.loadtxt(os.path.join(HERE, "acoroute_stair_6_10_direct.txt"))
    idx = np.clip(np.searchsorted(d[:, 0], f), 1, len(d) - 1)
    idx = np.where(np.abs(d[idx - 1, 0] - f) < np.abs(d[idx, 0] - f), idx - 1, idx)
    if np.max(np.abs(d[idx, 0] - f) / f) > 1e-8:
        raise SystemExit("The Ansys frequencies differ from the precomputed ACOROUTE file; run this script where fem_beam3d.py can be imported.")
    d = d[idx]
    RB = d[:, 2::2] + 1j * d[:, 3::2]
    fn = np.loadtxt(os.path.join(HERE, "acoroute_stair_6_10_modes.txt"))
    return d[:, 1], RB, fn, "precomputed acoroute_stair_6_10_direct.txt"


def band_integral(f, W, band):
    m = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(W[m], f[m]) if hasattr(np, "trapezoid") else np.trapz(W[m], f[m])


def match_frequencies(fa, fb):
    """Pairs each entry of fa with the nearest entry of fb. Returns (fa, fb_matched, relative difference)."""
    idx = np.array([np.argmin(np.abs(fb - x)) for x in fa])
    fbm = fb[idx]
    return fa, fbm, (fbm - fa) / fa


def main():
    out = []
    def say(s):
        print(s); out.append(s)

    f, FA = load_ansys_A(os.path.join(HERE, "stair_6_10_ansys_A.txt"))
    W_ans = input_power_from_reactions(f, FA, U_A)
    if np.sum(W_ans) < 0.0:
        W_ans = -W_ans
        say("note: the Ansys reactions at A gave negative power with the convention 0.5*w*Re{i F^H u}; sign reversed (reaction taken as the force of the structure on the support)")
    fem = try_import_fem()
    W_aco, RB_aco, fn_aco, label = acoroute_solution(fem, f)
    say("ACOROUTE side: %s" % label)
    say("frequencies: %d points from %.1f to %.1f Hz" % (len(f), f[0], f[-1]))

    for band in SUB_BANDS:
        J_ans, J_aco = band_integral(f, W_ans, band), band_integral(f, W_aco, band)
        m = (f >= band[0]) & (f <= band[1])
        ddb = np.abs(10.0 * np.log10(W_aco[m] / W_ans[m]))
        say("%5.0f-%5.0f Hz: integrated input power Ansys %.6e, ACOROUTE %.6e W Hz, difference %+.2f %%; level difference median %.2f dB, 95th percentile %.2f dB" % (band[0], band[1], J_ans, J_aco, 100.0 * (J_aco - J_ans) / J_ans, np.median(ddb), np.percentile(ddb, 95)))

    modes_path = os.path.join(HERE, "stair_6_10_ansys_modes.txt")
    if os.path.isfile(modes_path):
        fn_ans = np.atleast_1d(np.loadtxt(modes_path))
        fn_ans = fn_ans[(fn_ans > 0.0) & (fn_ans <= F_MODAL_MAX)]
        fa, fb, rel = match_frequencies(fn_ans[fn_ans <= 10000.0], fn_aco)
        say("natural frequencies below 10 kHz: Ansys %d, ACOROUTE %d; largest relative difference %.3f %% (at %.1f Hz), mean |difference| %.3f %%" % (len(fa), int((fn_aco <= 10000.0).sum()), 100.0 * np.max(np.abs(rel)), fa[np.argmax(np.abs(rel))], 100.0 * np.mean(np.abs(rel))))
        inb = (fa >= BAND[0]) & (fa <= BAND[1])
        say("natural frequencies inside the band: Ansys %d, ACOROUTE %d; largest relative difference %.3f %%" % (int(inb.sum()), int(((fn_aco >= BAND[0]) & (fn_aco <= BAND[1])).sum()), 100.0 * np.max(np.abs(rel[inb]))))
        np.savetxt(os.path.join(HERE, "comparison_modes.txt"), np.column_stack([fa, fb, 100.0 * rel]), fmt="%.6e", header="f Ansys [Hz], f ACOROUTE [Hz], relative difference [%]")

    fig, axs0 = plt.subplots(2, 1, figsize=(7.0, 6.0))
    for ax, (x0, x1) in zip(axs0, [(0.0, f[-1] * 1e-3), (0.0, 2.0)]):
        ax.axvspan(BAND[0] * 1e-3, BAND[1] * 1e-3, color="0.9", lw=0)
        ax.semilogy(f * 1e-3, W_ans, "k-", lw=1.4, label="Ansys BEAM188")
        ax.semilogy(f * 1e-3, W_aco, "r--", lw=1.0, label="Present model")
        ax.set_xlabel("Frequency [kHz]"); ax.set_ylabel("Active input power [W]")
        ax.set_xlim(x0, x1); ax.grid(True, which="both", lw=0.3, alpha=0.5)
    axs0[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ansys_input_power.pdf")); fig.savefig(os.path.join(HERE, "fig_ansys_input_power.png"), dpi=200)

    b_path = os.path.join(HERE, "stair_6_10_ansys_B.txt")
    if os.path.isfile(b_path):
        fB, RB_ans = load_ansys_B(b_path)
        Fa, Fo = np.linalg.norm(RB_ans[:, :3], axis=1), np.linalg.norm(RB_aco[:, :3], axis=1)
        Ma, Mo = np.linalg.norm(RB_ans[:, 3:], axis=1), np.linalg.norm(RB_aco[:, 3:], axis=1)
        mb = (fB >= BAND[0]) & (fB <= BAND[1])
        rmsF = lambda x: np.sqrt(np.mean(x[mb] ** 2))
        say("reactions at B inside the band (RMS over frequency of the resultant amplitude): force Ansys %.4e, ACOROUTE %.4e N (%+.2f %%); moment Ansys %.4e, ACOROUTE %.4e N m (%+.2f %%)" % (rmsF(Fa), rmsF(Fo), 100.0 * (rmsF(Fo) - rmsF(Fa)) / rmsF(Fa), rmsF(Ma), rmsF(Mo), 100.0 * (rmsF(Mo) - rmsF(Ma)) / rmsF(Ma)))
        fig2, axs = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
        for ax2, ya, yo, lab in [(axs[0], Fa, Fo, "Reaction force at $B$ [N]"), (axs[1], Ma, Mo, "Reaction moment at $B$ [N m]")]:
            ax2.axvspan(BAND[0] * 1e-3, BAND[1] * 1e-3, color="0.9", lw=0)
            ax2.semilogy(fB * 1e-3, ya, "k-", lw=1.4, label="Ansys BEAM188")
            ax2.semilogy(fB * 1e-3, yo, "r--", lw=1.0, label="Present model")
            ax2.set_ylabel(lab); ax2.grid(True, which="both", lw=0.3, alpha=0.5)
        axs[0].legend(); axs[1].set_xlabel("Frequency [kHz]"); axs[1].set_xlim(0.0, fB[-1] * 1e-3)
        fig2.tight_layout(); fig2.savefig(os.path.join(HERE, "fig_ansys_reactions_B.pdf")); fig2.savefig(os.path.join(HERE, "fig_ansys_reactions_B.png"), dpi=200)

    np.savetxt(os.path.join(HERE, "comparison_input_power.txt"), np.column_stack([f, W_ans, W_aco]), fmt="%.8e", header="f [Hz], input power Ansys [W], input power ACOROUTE [W]")
    with open(os.path.join(HERE, "comparison_summary.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
