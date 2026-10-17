"""Validate the 3D Timoshenko beam FEM against analytical beam frequencies."""
import numpy as np
from scipy.linalg import eigh
import fem_beam3d as fem

# Steel circular tube
E = 210e9; nu = 0.3; rho = 7850.0
G = E / (2.0 * (1.0 + nu))
mat = {"E": E, "G": G, "rho": rho}
sec = fem.pipe_section(outer_radius=0.005, thickness=0.001, nu=nu)
L = 0.3

# Straight beam along x, finely meshed
path = [(0, 0, 0), (L, 0, 0)]
nodes, elems = fem.mesh_path(path, max_elem_len=L / 40)
K, M = fem.assemble(nodes, elems, mat, sec)
K = K.toarray(); M = M.toarray()
ndof = K.shape[0]


def freqs_with_fixed(fixed_dofs):
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    w2, _ = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
    w2 = np.clip(w2, 0, None)
    return np.sqrt(w2) / (2 * np.pi)


def unique_bending(f, tol=1e-2):
    """Collapse degenerate (xy/xz) bending pairs into single frequencies."""
    f = f[f > 1.0]
    out = []
    for v in f:
        if not out or abs(v - out[-1]) / max(v, 1.0) > tol:
            out.append(v)
    return np.array(out)


A = sec["A"]; I = sec["Iz"]
coeff = np.sqrt(E * I / (rho * A)) / L**2

# --- Cantilever (clamp node 0) ---
fixed = list(range(0, 6))
fc = freqs_with_fixed(fixed)
bl = np.array([1.875104, 4.694091, 7.854757])
fc_an = bl**2 / (2 * np.pi) * coeff
print("CANTILEVER bending (Hz)   FEM        analytical   err%")
fb = unique_bending(fc)
for k in range(3):
    print(f"  mode {k+1:>2d}   {fb[k]:12.2f} {fc_an[k]:12.2f}   {100*(fb[k]-fc_an[k])/fc_an[k]:6.2f}")

# --- Free-free ---
ff = freqs_with_fixed([])
bl = np.array([4.730041, 7.853205, 10.995608])
ff_an = bl**2 / (2 * np.pi) * coeff
print("\nFREE-FREE bending (Hz)    FEM        analytical   err%")
fb = unique_bending(ff)
for k in range(3):
    print(f"  mode {k+1:>2d}   {fb[k]:12.2f} {ff_an[k]:12.2f}   {100*(fb[k]-ff_an[k])/ff_an[k]:6.2f}")

# --- Axial (clamp node 0): f_n = (2n-1)/4 * c/L, c=sqrt(E/rho) ---
print("\nAXIAL clamped-free (Hz)   analytical check")
c = np.sqrt(E / rho)
print(f"  f1 analytical = {c/(4*L):.1f}")

# --- Rigid-body check (free-free should have ~6 zero modes) ---
print(f"\nFree-free near-zero modes (should be 6): "
      f"{int(np.sum(ff < 1.0))}")

# --- Prescribed-displacement input power: reaction power == dissipated power ---
# Active input power for unit displacements at node 0 must equal the power
# dissipated by the damping at every frequency (steady-state energy balance).
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

Kc, Mc = fem.assemble(nodes, elems, mat, sec)
ray = (0.0, 2e-7); alpha, beta = ray
Cc = alpha * Mc + beta * Kc
drive = 0; fixedN = len(nodes) - 1
pres, u_p, free = fem._prescribed_sets(drive, fixedN, (1.0, 1.0, 1.0), Kc.shape[0])

print("\nPRESCRIBED-DISPLACEMENT input power (energy balance)")
print("  f[Hz]       W_in(reactions)   W_diss(damping)    rel.err")
maxerr = 0.0
for f in [2000.0, 5500.0, 9000.0]:
    w = 2 * np.pi * f
    Z = (-w**2) * Mc + (1j * w) * Cc + Kc
    uf = splu(csc_matrix(Z[np.ix_(free, free)])).solve(-(Z[np.ix_(free, pres)].dot(u_p)))
    R = Z[np.ix_(pres, free)].dot(uf) + Z[np.ix_(pres, pres)].dot(u_p)
    W_in = 0.5 * w * np.real(1j * np.vdot(R, u_p))
    U = np.zeros(Kc.shape[0], complex); U[free] = uf; U[pres] = u_p
    W_diss = 0.5 * w**2 * np.real(np.vdot(U, Cc.dot(U)))
    err = abs(W_in - W_diss) / max(abs(W_diss), 1e-30); maxerr = max(maxerr, err)
    print(f"  {f:7.1f}    {W_in:14.6e}   {W_diss:14.6e}    {err:.1e}")
print(f"  max relative error: {maxerr:.1e}  (should be ~1e-12)")
Wfn = fem.input_power_spectrum_prescribed(nodes, elems, mat, sec, drive, fixedN,
                                          np.linspace(1000, 10000, 150), rayleigh=ray)
print(f"  input power non-negative over sweep (passive): {bool(Wfn.min() >= -1e-9 * abs(Wfn).max())}")
