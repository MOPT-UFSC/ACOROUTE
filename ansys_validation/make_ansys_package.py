"""
make_ansys_package.py
Builds the Ansys verification package for the reported staircase route (6-10 kHz):
the APDL input with the beam mesh embedded, the node file, and the ACOROUTE reference
(direct solver on the same mesh and at the same frequencies).

The route is rebuilt from the case script revision_results/stair_6_10/stair_6_10.py and from
the lattice indices of the reported route (route_final_ids_stair_6_10.npy, in this folder).
Run from anywhere:
    python ansys_validation/make_ansys_package.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import run_campaign as RC

CASE = "stair_6_10"
H_MESH = 0.002            # element size of the shared mesh [m] (reference model of the paper)
F_MAX = 11000.0           # upper limit of the harmonic sweep [Hz]
DF_BASE = 2.0             # uniform base step [Hz], from DF_BASE to F_MAX
REFINE_REL = 0.005        # half-width of the refined window around each natural frequency, relative
PTS_PER_BW = 2.0          # points per half-power bandwidth inside the refined windows
F_MODAL_MAX = 12000.0     # upper limit of the modal extraction [Hz]
OUT = HERE
CASE_SCRIPT = os.path.join(ROOT, "revision_results", CASE, CASE + ".py")


def rebuild_route():
    ns, _var, _cap, _tail = RC._exec_head(CASE_SCRIPT, quiet=True)
    geo, grid = ns["geo"], ns["grid"]
    ids = np.load(os.path.join(HERE, "route_final_ids_%s.npy" % CASE))
    nodes, elems, L = geo.mesh_spline(grid["coords"][ids], H_MESH, smooth=ns["SMOOTH"])
    return ns, nodes, elems, L


def sweep_frequencies(fn, rayleigh):
    """Frequency list from DF_BASE to F_MAX: uniform base grid plus refined windows around the
    natural frequencies whose half-power bandwidth is not resolved by the base grid."""
    alpha, beta = rayleigh
    base = np.arange(DF_BASE, F_MAX + 0.5 * DF_BASE, DF_BASE)
    parts = [base]
    for f0 in fn[fn < F_MAX]:
        bw = alpha / (2.0 * np.pi) + 2.0 * np.pi * beta * f0 ** 2
        if bw >= 4.0 * DF_BASE:
            continue
        hw = max(REFINE_REL * f0, 4.0 * bw)
        step = bw / PTS_PER_BW
        parts.append(f0 + np.arange(-hw, hw + 0.5 * step, step))
    f = np.unique(np.concatenate(parts))
    f = f[(f > 0.0) & (f <= F_MAX)]
    keep = np.r_[True, np.diff(f) > 1e-4]
    return f[keep]


def write_apdl(path, nodes, ns, n_modes_hint, freqs):
    sec = ns["sec"]; mat = ns["mat"]
    ro = ns["OUTER_RADIUS"]; ri = ro - ns["THICKNESS"]
    u = ns["PRESCRIBED_DISP"]; alpha, beta = ns["RAY"]
    kga = sec["kappa"] * sec["A"] * mat["G"]
    n = len(nodes)
    L = []
    a = L.append
    a("! ==========================================================================")
    a("! ACOROUTE verification against Ansys BEAM188")
    a("! Case: staircase domain (two baffles), 6-10 kHz, reported route (seed 5)")
    a("! Mesh: %d nodes, %d two-node elements of about %.1f mm on the smooth centreline" % (n, n - 1, H_MESH * 1e3))
    a("! Units: SI (m, kg, s, N, Hz)")
    a("! Boundary conditions: node A (first) with prescribed translations and free")
    a("! rotations; node B (last) clamped. Rayleigh damping C = ALPHA_R*M + BETA_R*K.")
    a("! Harmonic sweep from %.0f Hz to %.0f Hz with a user-defined frequency list (HARFRQ, FREQARR)." % (DF_BASE, F_MAX))
    a("! Recommended: run_ansys_smp.bat (batch, shared-memory parallel: one results file).")
    a("! Otherwise File > Read Input From (*VWRITE does not work when the commands are")
    a("! pasted in the command line).")
    a("! Outputs: stair_6_10_ansys_modes.txt, stair_6_10_ansys_A.txt, stair_6_10_ansys_B.txt")
    a("! ==========================================================================")
    a("FINISH")
    a("/CLEAR,NOSTART")
    a("/UIS,MSGPOP,3           ! interactive runs: no pop-up boxes for notes and warnings")
    a("/FILNAME,stair_6_10_modal,0")
    a("/TITLE,ACOROUTE staircase route - BEAM188 verification")
    a("")
    a("! ---- parameters -----------------------------------------------------------")
    a("E_MOD   = %.6e      ! Young's modulus [Pa]" % mat["E"])
    a("NU_P    = %.4f           ! Poisson's ratio" % ns["nu"])
    a("RHO_M   = %.4f        ! density [kg/m3]" % mat["rho"])
    a("R_OUT   = %.9e   ! outer radius [m]" % ro)
    a("R_IN    = %.9e   ! inner radius [m]" % ri)
    a("ALPHA_R = %.6e      ! Rayleigh mass coefficient [1/s]" % alpha)
    a("BETA_R  = %.6e      ! Rayleigh stiffness coefficient [s]" % beta)
    a("U_AX    = %.6e      ! prescribed displacement phasors at A [m] (real parts)" % np.real(u[0]))
    a("U_AY    = %.6e" % np.real(u[1]))
    a("U_AZ    = %.6e" % np.real(u[2]))
    a("U_AX_I  = %.6e      ! imaginary parts" % np.imag(u[0]))
    a("U_AY_I  = %.6e" % np.imag(u[1]))
    a("U_AZ_I  = %.6e" % np.imag(u[2]))
    a("F_MAX   = %.4f      ! upper limit of the sweep [Hz]; the frequency list FRQ is defined below" % F_MAX)
    a("N_FRQ   = %d" % len(freqs))
    a("DO_MODAL    = 1         ! 1: natural frequencies with the same boundary conditions")
    a("F_MODAL_MAX = %.4f" % F_MODAL_MAX)
    a("N_MODES_MAX = %d       ! upper bound on the number of modes below F_MODAL_MAX" % n_modes_hint)
    a("USE_COWPER  = 0         ! 0: shear stiffness computed by Ansys for the CTUBE section")
    a("                        ! 1: override with kappa*G*A of ACOROUTE (Cowper, kappa = %.6f)" % sec["kappa"])
    a("KGA     = %.9e   ! kappa*G*A [N], used only when USE_COWPER = 1" % kga)
    a("N_NODES = %d" % n)
    a("N_A     = 1             ! driven node")
    a("N_B     = N_NODES       ! clamped node")
    a("")
    a("! ---- model ----------------------------------------------------------------")
    a("/PREP7")
    a("ET,1,BEAM188")
    a("KEYOPT,1,1,0            ! six DOF per node, no warping DOF")
    a("KEYOPT,1,3,3            ! cubic shape functions")
    a("MP,EX,1,E_MOD")
    a("MP,PRXY,1,NU_P")
    a("MP,DENS,1,RHO_M")
    a("SECTYPE,1,BEAM,CTUBE")
    a("SECDATA,R_IN,R_OUT,64")
    a("*IF,USE_COWPER,EQ,1,THEN")
    a("SECCONTROL,KGA,,KGA")
    a("*ENDIF")
    for i, p in enumerate(nodes, start=1):
        a("N,%d,%.12e,%.12e,%.12e" % (i, p[0], p[1], p[2]))
    a("TYPE,1")
    a("MAT,1")
    a("SECNUM,1")
    a("*DO,I,1,N_NODES-1")
    a("E,I,I+1")
    a("*ENDDO")
    a("NSEL,S,NODE,,N_A")
    a("NSEL,A,NODE,,N_B")
    a("CM,ENDS,NODE")
    a("ALLSEL,ALL")
    a("FINISH")
    a("")
    a("! ---- frequency list of the harmonic analysis [Hz]: %.0f Hz base grid plus refined windows" % DF_BASE)
    a("! around the natural frequencies below about %.0f Hz (half-power bandwidth below %.0f Hz)" % (np.sqrt(4.0 * DF_BASE / (2.0 * np.pi * ns["RAY"][1])), 4.0 * DF_BASE))
    a("*DIM,FRQ,ARRAY,N_FRQ")
    for i in range(0, len(freqs), 8):
        a("FRQ(%d) = " % (i + 1) + ",".join("%.10e" % v for v in freqs[i:i + 8]))
    a("")
    a("! ---- natural frequencies with the boundary conditions of the harmonic analysis")
    a("*IF,DO_MODAL,EQ,1,THEN")
    a("/SOLU")
    a("ANTYPE,MODAL")
    a("MODOPT,LANB,N_MODES_MAX,0,F_MODAL_MAX")
    a("MXPAND,N_MODES_MAX")
    a("D,N_A,UX,0")
    a("D,N_A,UY,0")
    a("D,N_A,UZ,0")
    a("D,N_B,ALL,0")
    a("SOLVE")
    a("FINISH")
    a("/POST1")
    a("*GET,N_FOUND,ACTIVE,0,SET,NSET")
    a("*DIM,FMOD,ARRAY,N_FOUND")
    a("*DO,I,1,N_FOUND")
    a("*GET,FMOD(I),MODE,I,FREQ")
    a("*ENDDO")
    a("*CFOPEN,stair_6_10_ansys_modes,txt")
    a("*VWRITE,FMOD(1)")
    a("(E22.13)")
    a("*CFCLOS")
    a("FINISH")
    a("*ENDIF")
    a("")
    a("! ---- full harmonic analysis ------------------------------------------------")
    a("/FILNAME,stair_6_10_harm,0")
    a("/SOLU")
    a("ANTYPE,HARMIC")
    a("HROPT,FULL")
    a("HROUT,ON                ! real and imaginary parts")
    a("EQSLV,SPARSE")
    a("ALPHAD,ALPHA_R")
    a("BETAD,BETA_R")
    a("DDELE,ALL,ALL")
    a("D,N_B,ALL,0")
    a("D,N_A,UX,U_AX,U_AX_I")
    a("D,N_A,UY,U_AY,U_AY_I")
    a("D,N_A,UZ,U_AZ,U_AZ_I")
    a("HARFRQ,0,F_MAX,,,%FRQ%,1E-9   ! user-defined frequencies; the single substep of the range falls on F_MAX")
    a("NSUBST,1")
    a("KBC,1")
    a("OUTRES,ERASE")
    a("OUTRES,ALL,NONE")
    a("OUTRES,NSOL,ALL,ENDS")
    a("OUTRES,RSOL,ALL")
    a("SOLVE")
    a("FINISH")
    a("")
    a("! ---- distributed runs: combine the local results files if the global one is missing")
    a("/INQUIRE,HAS_RST,EXIST,stair_6_10_harm,rst")
    a("*IF,HAS_RST,EQ,0,THEN")
    a("/AUX2")
    a("COMBINE,RST")
    a("FINISH")
    a("*ENDIF")
    a("")
    a("! ---- export: reactions at A and B, real and imaginary parts ------------------")
    a("/POST26")
    a("FILE,stair_6_10_harm,rst")
    a("NUMVAR,20")
    a("RFORCE,2,N_A,F,X,FAX")
    a("RFORCE,3,N_A,F,Y,FAY")
    a("RFORCE,4,N_A,F,Z,FAZ")
    a("RFORCE,5,N_B,F,X,FBX")
    a("RFORCE,6,N_B,F,Y,FBY")
    a("RFORCE,7,N_B,F,Z,FBZ")
    a("RFORCE,8,N_B,M,X,MBX")
    a("RFORCE,9,N_B,M,Y,MBY")
    a("RFORCE,10,N_B,M,Z,MBZ")
    a("STORE,MERGE")
    a("*GET,N_PTS,VARI,0,NSETS")
    a("*DIM,RA,ARRAY,N_PTS,7")
    a("*DIM,RB,ARRAY,N_PTS,13")
    a("VGET,RA(1,1),1")
    for k in range(3):
        a("VGET,RA(1,%d),%d,,0" % (2 + 2 * k, 2 + k))
        a("VGET,RA(1,%d),%d,,1" % (3 + 2 * k, 2 + k))
    a("VGET,RB(1,1),1")
    for k in range(6):
        a("VGET,RB(1,%d),%d,,0" % (2 + 2 * k, 5 + k))
        a("VGET,RB(1,%d),%d,,1" % (3 + 2 * k, 5 + k))
    a("! columns: f [Hz], then Re and Im of FX, FY, FZ at A [N]")
    a("*CFOPEN,stair_6_10_ansys_A,txt")
    a("*VWRITE," + ",".join("RA(1,%d)" % j for j in range(1, 8)))
    a("(7E22.13)")
    a("*CFCLOS")
    a("! columns: f [Hz], then Re and Im of FX, FY, FZ [N], MX, MY, MZ [N m] at B")
    a("*CFOPEN,stair_6_10_ansys_B,txt")
    a("*VWRITE," + ",".join("RB(1,%d)" % j for j in range(1, 14)))
    a("(13E22.13)")
    a("*CFCLOS")
    a("FINISH")
    a("*CFOPEN,stair_6_10_ansys_done,txt")
    a("*VWRITE,N_PTS")
    a("('export finished, frequency points: ',F8.0)")
    a("*CFCLOS")
    a("/COM, ===== EXPORT FINISHED: stair_6_10_ansys_A.txt and stair_6_10_ansys_B.txt written =====")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    ns, nodes, elems, L = rebuild_route()
    fem = ns["fem"]; mat, sec = ns["mat"], ns["sec"]
    n = len(nodes)
    fn = fem.natural_frequencies(nodes, elems, mat, sec, n - 1, extra_fixed_dofs=[0, 1, 2])
    fn = fn[fn <= F_MODAL_MAX]
    n_hint = int(np.ceil(1.3 * len(fn) / 10.0) * 10)
    fn_all = fem.natural_frequencies(nodes, elems, mat, sec, n - 1, extra_fixed_dofs=[0, 1, 2])
    freqs = sweep_frequencies(fn_all, ns["RAY"])
    W, RB = fem.input_power_spectrum_prescribed(nodes, elems, mat, sec, 0, n - 1, freqs, prescribed_disp=ns["PRESCRIBED_DISP"], rayleigh=ns["RAY"], prescribed_mask=ns["PRESCRIBED_MASK"], return_reactions=True)
    write_apdl(os.path.join(OUT, "stair_6_10_harmonic.inp"), nodes, ns, n_hint, freqs)
    np.savetxt(os.path.join(OUT, "stair_6_10_nodes.txt"), nodes, fmt="%.12e", header="x y z [m], node 1 = A (driven), last node = B (clamped), %d nodes, developed length %.6f m" % (n, L))
    cols = [freqs, W]
    for k in range(6):
        cols += [RB[:, k].real, RB[:, k].imag]
    np.savetxt(os.path.join(OUT, "acoroute_stair_6_10_direct.txt"), np.column_stack(cols), fmt="%.13e", header="f [Hz], active input power [W], then Re and Im of FX FY FZ [N] MX MY MZ [N m] at B; ACOROUTE direct solver, %.1f mm mesh" % (H_MESH * 1e3))
    np.savetxt(os.path.join(OUT, "acoroute_stair_6_10_modes.txt"), fn, fmt="%.10e", header="natural frequencies [Hz] below %.0f Hz, translations of A fixed, B clamped; ACOROUTE, %.1f mm mesh" % (F_MODAL_MAX, H_MESH * 1e3))
    lo, hi = ns["BANDS"][0]
    m = (freqs >= lo) & (freqs <= hi)
    print("nodes %d, length %.4f m, modes below %.0f Hz: %d (N_MODES_MAX %d), in band: %d" % (n, L, F_MODAL_MAX, len(fn), n_hint, int(((fn >= lo) & (fn <= hi)).sum())))
    print("frequency list: %d points, %.4f to %.1f Hz, smallest step %.2e Hz; lowest natural frequencies %s" % (len(freqs), freqs[0], freqs[-1], np.min(np.diff(freqs)), np.array2string(fn[:4], precision=2)))
    print("band integral on the sweep (trapezoid): %.6e W Hz" % np.trapezoid(W[m], freqs[m]))
    print("kappa %.6f, kappa*G*A %.6e N" % (sec["kappa"], sec["kappa"] * sec["A"] * mat["G"]))


if __name__ == "__main__":
    main()
