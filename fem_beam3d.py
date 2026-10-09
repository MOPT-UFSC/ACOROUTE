"""
fem_beam3d.py
3D Timoshenko beam finite element model for harmonic analysis of metallic pipes.

Harmonic convention: u(t) = Re{u_hat exp(i omega t)}. Prescribed displacements and
forces are phasors and may therefore be complex.

This module is intentionally written in a procedural style (functions + dicts/arrays,
no classes) and is fully self-contained (NumPy + SciPy only). It plays the role that
OpenPulse plays in the original work: given a tube path (a polyline in 3D space), it
builds the mass/stiffness matrices, solves the forced-harmonic problem
    (-w^2 M + i w C + K) U = P
and returns the active input power
    Wbar(w) = (w/2) * Re( j * P^H U )
which is the positive-definite objective used by the optimizer.

The element is a two-node, 6-DOF/node Timoshenko beam (12 DOF). For a CIRCULAR pipe
the cross section is isotropic in bending (Iy == Iz, J == 2I), so the orientation of
the local cross-sectional axes is irrelevant: only the segment direction matters. This
removes the usual "up vector" ambiguity of 3D beams.

Local DOF order per node: [u_x, u_y, u_z, theta_x, theta_y, theta_z], x = beam axis.

To swap this for OpenPulse, replace `evaluate_path_power` only: keep the same signature
(path nodes -> band-integrated input power) and the ACO engine does not change.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import solve_banded

# NumPy < 2.0 compatibility: np.trapezoid was named np.trapz before NumPy 2.0
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz


# ----------------------------------------------------------------------------- #
# Section properties (circular hollow pipe)
# ----------------------------------------------------------------------------- #
def pipe_section(outer_radius, thickness, nu):
    """Geometric and shear properties of a thin/thick-walled circular pipe.

    Returns a dict with A, Iy, Iz, J, kappa (shear coefficient), As_y, As_z.
    Cowper's shear coefficient for a hollow circular section is used.
    """
    ro = float(outer_radius)
    ri = float(outer_radius - thickness)
    if ri < 0.0:
        raise ValueError("thickness larger than outer radius")

    A = np.pi * (ro**2 - ri**2)
    I = np.pi * (ro**4 - ri**4) / 4.0       # Iy == Iz for a circular section
    J = 2.0 * I                              # polar second moment

    # Cowper (1966), shear coefficient of a hollow circular cross section:
    #   kappa = 6(1+nu)(1+m^2)^2 / [ (7+6nu)(1+m^2)^2 + (20+12nu) m^2 ],  m = r_i/r_o.
    # Check of the formula: at m=0 (solid circle) it returns 6(1+nu)/(7+6nu) = 0.886 for
    # nu=0.3, the tabulated value. A previous version of this function used a variant
    # that returns 0.925 in that limit and is therefore not Cowper's coefficient.
    m = ri / ro if ro > 0.0 else 0.0
    m2 = m * m
    num = 6.0 * (1.0 + nu) * (1.0 + m2) ** 2
    den = (7.0 + 6.0 * nu) * (1.0 + m2) ** 2 + (20.0 + 12.0 * nu) * m2
    kappa = num / den

    return {
        "A": A, "Iy": I, "Iz": I, "J": J,
        "kappa": kappa, "As_y": kappa * A, "As_z": kappa * A,
    }


# ----------------------------------------------------------------------------- #
# Element matrices
# ----------------------------------------------------------------------------- #
def _local_stiffness(L, E, G, sec):
    """12x12 local Timoshenko stiffness matrix for a circular beam segment."""
    A, Iy, Iz, J = sec["A"], sec["Iy"], sec["Iz"], sec["J"]
    As_y, As_z = sec["As_y"], sec["As_z"]

    K = np.zeros((12, 12))

    # Axial (u_x): dofs 0, 6
    ka = E * A / L
    for (a, b, s) in [(0, 0, 1), (6, 6, 1), (0, 6, -1), (6, 0, -1)]:
        K[a, b] += s * ka

    # Torsion (theta_x): dofs 3, 9
    kt = G * J / L
    for (a, b, s) in [(3, 3, 1), (9, 9, 1), (3, 9, -1), (9, 3, -1)]:
        K[a, b] += s * kt

    # Bending in x-y plane uses Iz (dofs u_y=1, theta_z=5, u_y2=7, theta_z2=11)
    phi_y = 12.0 * E * Iz / (G * As_y * L**2)
    c = E * Iz / (L**3 * (1.0 + phi_y))
    idx = [1, 5, 7, 11]
    Kb = c * np.array([
        [ 12.0,        6.0 * L,      -12.0,        6.0 * L],
        [ 6.0 * L, (4.0 + phi_y) * L**2, -6.0 * L, (2.0 - phi_y) * L**2],
        [-12.0,       -6.0 * L,       12.0,       -6.0 * L],
        [ 6.0 * L, (2.0 - phi_y) * L**2, -6.0 * L, (4.0 + phi_y) * L**2],
    ])
    for a in range(4):
        for b in range(4):
            K[idx[a], idx[b]] += Kb[a, b]

    # Bending in x-z plane uses Iy (dofs u_z=2, theta_y=4, u_z2=8, theta_y2=10)
    # Sign convention of the translation-rotation coupling is flipped vs x-y plane.
    phi_z = 12.0 * E * Iy / (G * As_z * L**2)
    c = E * Iy / (L**3 * (1.0 + phi_z))
    idx = [2, 4, 8, 10]
    Kb = c * np.array([
        [ 12.0,       -6.0 * L,      -12.0,       -6.0 * L],
        [-6.0 * L, (4.0 + phi_z) * L**2,  6.0 * L, (2.0 - phi_z) * L**2],
        [-12.0,        6.0 * L,       12.0,        6.0 * L],
        [-6.0 * L, (2.0 - phi_z) * L**2,  6.0 * L, (4.0 + phi_z) * L**2],
    ])
    for a in range(4):
        for b in range(4):
            K[idx[a], idx[b]] += Kb[a, b]

    return K


def _local_mass(L, rho, sec):
    """12x12 consistent mass matrix (translational + rotary inertia)."""
    A, Iy, Iz, J = sec["A"], sec["Iy"], sec["Iz"], sec["J"]
    M = np.zeros((12, 12))

    # Axial (u_x): dofs 0, 6
    ma = rho * A * L / 6.0
    for (a, b, v) in [(0, 0, 2.0), (6, 6, 2.0), (0, 6, 1.0), (6, 0, 1.0)]:
        M[a, b] += v * ma

    # Torsion (theta_x): dofs 3, 9  (polar rotary inertia rho*J)
    mt = rho * J * L / 6.0
    for (a, b, v) in [(3, 3, 2.0), (9, 9, 2.0), (3, 9, 1.0), (9, 3, 1.0)]:
        M[a, b] += v * mt

    # Bending consistent mass: translational part (rho*A*L/420) + rotary part (rho*I/(30L))
    def bending_mass(I, idx, flip):
        mtb = rho * A * L / 420.0
        Mt = mtb * np.array([
            [156.0,  22.0 * L,   54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0,   13.0 * L,  156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ])
        mrb = rho * I / (30.0 * L)
        Mr = mrb * np.array([
            [36.0,   3.0 * L,  -36.0,   3.0 * L],
            [3.0 * L, 4.0 * L**2, -3.0 * L, -1.0 * L**2],
            [-36.0, -3.0 * L,   36.0,  -3.0 * L],
            [3.0 * L, -1.0 * L**2, -3.0 * L, 4.0 * L**2],
        ])
        Mb = Mt + Mr
        if flip:
            # flip sign of translation-rotation coupling terms for the x-z plane
            S = np.array([[1, -1, 1, -1],
                          [-1, 1, -1, 1],
                          [1, -1, 1, -1],
                          [-1, 1, -1, 1]], dtype=float)
            Mb = Mb * S
        for a in range(4):
            for b in range(4):
                M[idx[a], idx[b]] += Mb[a, b]

    bending_mass(Iz, [1, 5, 7, 11], flip=False)   # x-y plane
    bending_mass(Iy, [2, 4, 8, 10], flip=True)     # x-z plane
    return M


def _rotation_matrix(p_i, p_j):
    """3x3 rotation from global to local frame; local x along (p_j - p_i).

    For a circular section the choice of local y,z is arbitrary, so a robust
    orthonormal completion is used (handles the axis-aligned-with-Z case).
    """
    ex = np.asarray(p_j, float) - np.asarray(p_i, float)
    L = np.linalg.norm(ex)
    if L < 1e-12:
        raise ValueError("zero-length element")
    ex = ex / L
    ref = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ey = np.cross(ref, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return np.vstack([ex, ey, ez]), L


def _element_global(p_i, p_j, E, G, rho, sec):
    """Element K, M in global coordinates."""
    R, L = _rotation_matrix(p_i, p_j)
    T = np.zeros((12, 12))
    for b in range(4):
        T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
    Kl = _local_stiffness(L, E, G, sec)
    Ml = _local_mass(L, rho, sec)
    Kg = T.T @ Kl @ T
    Mg = T.T @ Ml @ T
    return Kg, Mg, L


# ----------------------------------------------------------------------------- #
# Mesh a polyline path into elements
# ----------------------------------------------------------------------------- #
def mesh_path(path_xyz, max_elem_len):
    """Subdivide a polyline (list of 3D points) so each element <= max_elem_len.

    Returns (nodes [n,3], elements [m,2] int).
    """
    nodes = [np.asarray(path_xyz[0], float)]
    elements = []
    for k in range(1, len(path_xyz)):
        a = np.asarray(path_xyz[k - 1], float)
        b = np.asarray(path_xyz[k], float)
        seg = np.linalg.norm(b - a)
        n_sub = max(1, int(np.ceil(seg / max_elem_len)))
        for s in range(1, n_sub + 1):
            nodes.append(a + (b - a) * (s / n_sub))
            elements.append((len(nodes) - 2, len(nodes) - 1))
    return np.array(nodes), np.array(elements, dtype=int)


# ----------------------------------------------------------------------------- #
# Assembly + harmonic solve + input power
# ----------------------------------------------------------------------------- #
def assemble(nodes, elements, mat, sec):
    """Assemble global sparse K and M. mat = {E, G, rho}.

    The element contributions are collected as COO triplets and scattered into CSC
    in one vectorized conversion (duplicate entries are summed by construction of
    the COO format). The previous per-entry lil_matrix accumulation (144 Python
    __setitem__ calls per element) dominated the entire FEM evaluation; the physics
    and the assembled matrices are identical.
    """
    ndof = 6 * len(nodes)
    nel = len(elements)
    rows = np.empty((nel, 144), dtype=np.int64)
    cols = np.empty((nel, 144), dtype=np.int64)
    dK = np.empty((nel, 144))
    dM = np.empty((nel, 144))
    for e, (ni, nj) in enumerate(elements):
        Kg, Mg, _ = _element_global(nodes[ni], nodes[nj],
                                    mat["E"], mat["G"], mat["rho"], sec)
        dofs = np.r_[6 * ni:6 * ni + 6, 6 * nj:6 * nj + 6]
        rows[e] = np.repeat(dofs, 12)
        cols[e] = np.tile(dofs, 12)
        dK[e] = Kg.ravel()
        dM[e] = Mg.ravel()
    shape = (ndof, ndof)
    K = coo_matrix((dK.ravel(), (rows.ravel(), cols.ravel())), shape=shape).tocsc()
    M = coo_matrix((dM.ravel(), (rows.ravel(), cols.ravel())), shape=shape).tocsc()
    return K, M


def _bandwidth(A):
    """Half-bandwidth max|i-j| of the nonzero structure of a sparse matrix."""
    A = A.tocoo()
    return int(np.max(np.abs(A.row - A.col))) if A.nnz else 0


def _sparse_to_banded(A, hb, dtype=complex):
    """LAPACK banded storage ab[hb + i - j, j] = A[i, j] (scipy solve_banded layout)."""
    A = A.tocoo()
    ab = np.zeros((2 * hb + 1, A.shape[0]), dtype=dtype)
    np.add.at(ab, (hb + A.row - A.col, A.col), A.data)
    return ab


def _apply_bc(K, M, fixed_dofs):
    """Penalty-free reduction: drop fixed DOFs and return free index map."""
    ndof = K.shape[0]
    free = np.setdiff1d(np.arange(ndof), np.asarray(fixed_dofs, int))
    return K[np.ix_(free, free)], M[np.ix_(free, free)], free


def input_power_spectrum(nodes, elements, mat, sec,
                         drive_node, drive_dir, fixed_node,
                         freqs_hz, rayleigh=(0.0, 1e-6), force=1.0, KM=None):
    """Active input power Wbar(w) at the drive DOF over a list of frequencies.

    drive_dir : unit 3-vector giving the direction of the harmonic point force
                (only translational excitation is used here).
    fixed_node: node index clamped in all 6 DOF (support / shell side).
    rayleigh  : (alpha, beta) for C = alpha*M + beta*K.
    KM        : optional pre-assembled (K, M); avoids re-assembling when the caller
                evaluates several quantities on the same mesh.
    Returns array Wbar [len(freqs_hz)].

    With Rayleigh damping the dynamic stiffness factorizes over the two assembled
    matrices, Z(w) = cK(w) K + cM(w) M with cK = 1 + i w beta and cM = i w alpha - w^2,
    and for a beam chain Z is banded (half-bandwidth <= 11), so each frequency costs
    one LAPACK banded LU (zgbsv) instead of a general sparse factorization.
    """
    K, M = assemble(nodes, elements, mat, sec) if KM is None else KM
    ndof = K.shape[0]

    fixed = list(range(6 * fixed_node, 6 * fixed_node + 6))
    Kf, Mf, free = _apply_bc(K, M, fixed)

    alpha, beta = rayleigh

    # Force vector: translational components on the drive node, projected to free set.
    P = np.zeros(ndof, dtype=complex)
    d = np.asarray(drive_dir, float)
    d = d / np.linalg.norm(d)
    P[6 * drive_node:6 * drive_node + 3] = force * d
    Pf = P[free]

    hb = max(_bandwidth(Kf), _bandwidth(Mf))
    Kb = _sparse_to_banded(Kf, hb)
    Mb = _sparse_to_banded(Mf, hb)

    Wbar = np.zeros(len(freqs_hz))
    Zb = np.empty_like(Kb)                 # reused banded buffer: no per-frequency allocation
    tmp = np.empty_like(Mb)
    for k, f in enumerate(freqs_hz):
        w = 2.0 * np.pi * f
        cK = 1.0 + 1j * w * beta
        cM = 1j * w * alpha - w**2
        np.multiply(Kb, cK, out=Zb)
        np.multiply(Mb, cM, out=tmp)
        Zb += tmp
        U = solve_banded((hb, hb), Zb, Pf, overwrite_ab=True, check_finite=False)
        # Wbar = (w/2) Re( j * P^H U )  ==  (1/2) Re( conj(P) . (j w U) )  [active input power]
        Wbar[k] = 0.5 * w * np.real(1j * np.vdot(Pf, U))
    return Wbar


def _prescribed_sets(drive_node, fixed_node, prescribed_disp, ndof,
                     constrain_drive_rot=False, prescribed_mask=(True, True, True)):
    """DOF bookkeeping for prescribed-displacement excitation.

    prescribed_mask selects which of the three translations at the drive node are
    prescribed; the unselected translations are left in the free set and respond
    freely. A prescribed component set to zero is held at zero (clamped in that
    direction); an unselected component is free, like a support that imposes some
    directions and allows sliding in the others.

    Returns (pres, u_p, free):
      pres : indices of the prescribed DOFs (the selected drive translations,
             optionally the drive rotations, and the six clamped DOFs at B)
      u_p  : complex prescribed displacement phasors on `pres`
      free : the remaining (unknown) DOFs, including the unselected drive translations
    """
    d = np.asarray(prescribed_disp, dtype=complex)
    m = [bool(x) for x in prescribed_mask]
    drive_tr = [6 * drive_node + i for i in range(3) if m[i]]
    drive_vals = [d[i] for i in range(3) if m[i]]
    drive_rot = list(range(6 * drive_node + 3, 6 * drive_node + 6))
    fixed = list(range(6 * fixed_node, 6 * fixed_node + 6))
    pres = list(drive_tr); vals = list(drive_vals)
    if constrain_drive_rot:
        pres += drive_rot; vals += [0.0, 0.0, 0.0]
    pres += fixed; vals += [0.0] * 6
    pres = np.asarray(pres, int)
    u_p = np.asarray(vals, complex)
    free = np.setdiff1d(np.arange(ndof), pres)
    return pres, u_p, free


def input_power_spectrum_prescribed(nodes, elements, mat, sec,
                                    drive_node, fixed_node, freqs_hz,
                                    prescribed_disp=(1.0, 1.0, 1.0),
                                    rayleigh=(0.0, 1e-6),
                                    constrain_drive_rot=False,
                                    prescribed_mask=(True, True, True),
                                    return_reactions=False, KM=None):
    """Active input power Wbar(w) for PRESCRIBED displacements at the drive node.

    Instead of a point force, the translations of `drive_node` selected by
    `prescribed_mask` are prescribed to the corresponding complex phasors in
    `prescribed_disp`, `fixed_node` is clamped, and the remaining DOFs (including any
    unselected drive translations) are free. A selected zero component is held at zero
    rather than left free. The active input power is the power
    delivered by the reaction forces at the prescribed DOFs,
        Wbar = (w/2) Re{ i R^H u_p },
    with R the reaction at the prescribed DOFs and u_p the prescribed displacement.
    The clamp at B and any zero-valued prescribed component deliver no power. The
    static condensation U_f = -Z_ff^{-1} Z_fp u_p gives the free response, and
    R = Z_pf U_f + Z_pp u_p the reactions. Returns array Wbar [len(freqs_hz)].

    KM: optional pre-assembled (K, M) for this mesh (avoids re-assembly).

    Solver: with Rayleigh damping every dynamic-stiffness block factorizes over the
    corresponding K and M blocks, Z_xx(w) = cK(w) K_xx + cM(w) M_xx with
    cK = 1 + i w beta, cM = i w alpha - w^2. The free-free block of a beam chain is
    banded (half-bandwidth <= 11, preserved under the removal of the prescribed rows
    and columns), so each frequency is one LAPACK banded LU (zgbsv) plus small sparse
    matvecs for the right-hand side and the reactions.
    """
    K, M = assemble(nodes, elements, mat, sec) if KM is None else KM
    ndof = K.shape[0]
    alpha, beta = rayleigh

    pres, u_p, free = _prescribed_sets(drive_node, fixed_node, prescribed_disp,
                                       ndof, constrain_drive_rot, prescribed_mask)

    # partition once; per frequency only the scalar combination cK*K + cM*M is formed
    Kff = K[np.ix_(free, free)]; Mff = M[np.ix_(free, free)]
    Kfp = K[np.ix_(free, pres)]; Mfp = M[np.ix_(free, pres)]
    Kpf = K[np.ix_(pres, free)]; Mpf = M[np.ix_(pres, free)]
    Kpp = K[np.ix_(pres, pres)]; Mpp = M[np.ix_(pres, pres)]

    hb = max(_bandwidth(Kff), _bandwidth(Mff))
    Kb = _sparse_to_banded(Kff, hb)
    Mb = _sparse_to_banded(Mff, hb)
    Kfp_up = Kfp.dot(u_p); Mfp_up = Mfp.dot(u_p)     # frequency-independent RHS parts
    Kpp_up = Kpp.dot(u_p); Mpp_up = Mpp.dot(u_p)

    Wbar = np.zeros(len(freqs_hz))
    if return_reactions:
        # node B is clamped, so its six DOFs are the last entries of `pres`; locate them
        # explicitly so the extraction does not rely on construction order
        posB = np.array([np.where(pres == 6 * fixed_node + i)[0][0] for i in range(6)])
        RB = np.zeros((len(freqs_hz), 6), complex)   # [Fx,Fy,Fz,Mx,My,Mz] at B per freq
    Zb = np.empty_like(Kb)                 # reused banded buffer: no per-frequency allocation
    tmp = np.empty_like(Mb)
    for k, f in enumerate(freqs_hz):
        w = 2.0 * np.pi * f
        cK = 1.0 + 1j * w * beta
        cM = 1j * w * alpha - w**2
        np.multiply(Kb, cK, out=Zb)
        np.multiply(Mb, cM, out=tmp)
        Zb += tmp
        Uf = solve_banded((hb, hb), Zb, -(cK * Kfp_up + cM * Mfp_up),
                          overwrite_ab=True, overwrite_b=True, check_finite=False)
        # reactions at prescribed DOFs: R = Z_pf U_f + Z_pp u_p
        R = cK * (Kpf.dot(Uf) + Kpp_up) + cM * (Mpf.dot(Uf) + Mpp_up)
        Wbar[k] = 0.5 * w * np.real(1j * np.vdot(R, u_p))
        if return_reactions:
            RB[k, :] = R[posB]
    return (Wbar, RB) if return_reactions else Wbar


def modal_basis_prescribed(K, M, pres, free, u_p, f_cut_hz=None):
    """Modal data of the free block for prescribed-displacement excitation.

    The free DOFs are those with A's prescribed translations and B's six DOFs
    blocked, so the modes of (K_ff, M_ff) diagonalize Z_ff(w) = cK K_ff + cM M_ff for
    every frequency (Rayleigh damping). f_cut_hz = None keeps the COMPLETE basis
    (dense generalized eigenproblem; the harmonic response is then the direct
    solution written in the modal basis, identical to round-off). A finite f_cut_hz
    keeps the modes below it (Sturm count for the exact number, shift-invert Lanczos
    or a range-restricted dense solver) and the modes above it are represented
    statically through the residual flexibility K_ff^-1 - Phi Lam^-1 Phi^T applied
    to the two forcing vectors (mode-acceleration correction). Without that term the
    quasi-static part of the response to a prescribed displacement, which is spread
    over the high modes, would be lost and the band power would be wrong by factors.
    """
    from scipy.linalg import eigh
    from scipy.sparse.linalg import eigsh, splu
    Kff = K[np.ix_(free, free)].tocsc(); Mff = M[np.ix_(free, free)].tocsc()
    fK = K[np.ix_(free, pres)].dot(u_p); fM = M[np.ix_(free, pres)].dot(u_p)
    Kpf = K[np.ix_(pres, free)]; Mpf = M[np.ix_(pres, free)]
    n = Kff.shape[0]
    if f_cut_hz is None:
        lam, Phi = eigh(Kff.toarray(), Mff.toarray())
        lam = np.clip(lam, 0.0, None)
        resK = resM = None
    else:
        w2c = (2.0 * np.pi * f_cut_hz) ** 2
        lu = splu((Kff - w2c * Mff).tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0)
        k = int(np.sum(lu.U.diagonal() < 0))             # exact number of modes below f_cut
        k = max(k, 1)
        if k < n // 4:
            lam, Phi = eigsh(Kff, k=k, M=Mff, sigma=0.5 * w2c, which="LM")
        else:
            lam, Phi = eigh(Kff.toarray(), Mff.toarray(), subset_by_value=(-np.inf, w2c))
        o = np.argsort(lam); lam, Phi = np.clip(lam[o], 0.0, None), Phi[:, o]
        luK = splu(Kff)
        def _solve(v):
            return luK.solve(np.real(v)) + 1j * luK.solve(np.imag(v))
        gK, gM = _solve(fK), _solve(fM)                     # static responses K_ff^-1 f
        a0 = Phi.T @ fK; b0 = Phi.T @ fM
        resK = gK - Phi @ (a0 / lam); resM = gM - Phi @ (b0 / lam)   # residual flexibility part
    basis = {
        "lam": lam, "n_modes": len(lam), "n_free": n, "f_cut_hz": f_cut_hz,
        "a": Phi.T @ fK, "b": Phi.T @ fM,                   # modal forcing (frequency-independent parts)
        "PK": Kpf.dot(Phi), "PM": Mpf.dot(Phi),             # p x k: reactions from modal amplitudes
        "Kpp_up": K[np.ix_(pres, pres)].dot(u_p), "Mpp_up": M[np.ix_(pres, pres)].dot(u_p),
    }
    if resK is not None:
        basis.update({"RKK": Kpf.dot(resK), "RKM": Kpf.dot(resM),
                      "RMK": Mpf.dot(resK), "RMM": Mpf.dot(resM)})
    return basis


def input_power_spectrum_modal(freqs_hz, u_p, basis, rayleigh=(0.0, 1e-6),
                               return_reactions=False, posB=None):
    """Active input power under prescribed displacement by modal superposition, using
    the data of modal_basis_prescribed. Vectorized over the frequencies. Returns
    Wbar [len(freqs_hz)] and, if requested, the reactions at the six DOFs of B."""
    alpha, beta = rayleigh
    w = 2.0 * np.pi * np.asarray(freqs_hz, float)
    cK = (1.0 + 1j * w * beta)[:, None]
    cM = (1j * w * alpha - w * w)[:, None]
    lam = basis["lam"][None, :]
    Q = -(cK * basis["a"][None, :] + cM * basis["b"][None, :]) / (cK * lam + cM)   # nf x k
    R = cK * (Q @ basis["PK"].T + basis["Kpp_up"][None, :]) \
        + cM * (Q @ basis["PM"].T + basis["Mpp_up"][None, :])                       # nf x p
    if "RKK" in basis:
        # residual (statically represented) modes: U_res = -(cK resK + cM resM)/cK
        UK = -(cK * basis["RKK"][None, :] + cM * basis["RKM"][None, :]) / cK
        UM = -(cK * basis["RMK"][None, :] + cM * basis["RMM"][None, :]) / cK
        R = R + cK * UK + cM * UM
    Wbar = 0.5 * w * np.real(1j * (np.conj(R) @ u_p))
    if return_reactions:
        return Wbar, R[:, posB]
    return Wbar


def natural_frequencies(nodes, elements, mat, sec, fixed_node, band_hz=None,
                        margin=0.15, extra_fixed_dofs=None, KM=None):
    """Undamped natural frequencies (Hz) of the clamped structure.

    If band_hz is given, only the frequencies inside [lo*(1-margin), hi*(1+margin)]
    are returned. Used to place frequency samples exactly on the resonances, which
    dominate the input power. Dense solver here (small models); for OpenPulse-scale
    models use a sparse shift-invert eigensolver targeting the band.
    KM: optional pre-assembled (K, M) for this mesh (avoids re-assembly).
    """
    from scipy.linalg import eigh
    K, M = assemble(nodes, elements, mat, sec) if KM is None else KM
    fixed = list(range(6 * fixed_node, 6 * fixed_node + 6))
    if extra_fixed_dofs is not None:
        fixed = fixed + list(extra_fixed_dofs)
    Kf, Mf, _ = _apply_bc(K, M, fixed)
    w2 = eigh(Kf.toarray(), Mf.toarray(), eigvals_only=True)
    f = np.sqrt(np.clip(w2, 0.0, None)) / (2.0 * np.pi)
    if band_hz is not None:
        lo, hi = band_hz[0] * (1.0 - margin), band_hz[1] * (1.0 + margin)
        f = f[(f >= lo) & (f <= hi)]
    return f


def modal_bandwidth_hz(f_r, rayleigh):
    """Half-power (3 dB) bandwidth [Hz] of mode r under Rayleigh damping C = aM*M + aK*K.

    The modal damping ratio is zeta_r = aM/(2 w_r) + aK w_r / 2, and the half-power
    bandwidth of a lightly damped resonance is B_r = 2 zeta_r f_r. With the
    stiffness-proportional term dominant, B_r grows as f_r^2, so a peak that is tens of
    hertz wide at 8 kHz is only a few hertz wide at 2 kHz. Any frequency grid that
    integrates the input power must resolve THIS width, not a fixed one.
    """
    f_r = float(f_r)
    if f_r <= 0.0:
        return 0.0
    aM, aK = rayleigh
    w = 2.0 * np.pi * f_r
    zeta = aM / (2.0 * w) + aK * w / 2.0
    return 2.0 * zeta * f_r


def _adaptive_freq_grid(band_hz, peaks, rayleigh, n_base=64,
                        span=4.0, pts_per_bw=4.0,
                        min_pts_per_peak=9, max_pts_per_peak=161):
    """Uniform base grid plus a cluster at each resonance, sized by the modal bandwidth.

    The cluster around mode r spans f_r +- span*B_r and carries enough points that
    pts_per_bw of them fall inside the half-power bandwidth B_r of that mode. This
    replaces the earlier fixed cluster (+-120 Hz, 7 points), which was calibrated on a
    high-frequency band and silently failed at low frequency: with the damping used
    here, B_r is about 5 Hz at 2 kHz while the fixed cluster sampled every 40 Hz, so the
    trapezoidal rule turned a narrow peak into a wide triangle and overestimated the
    band integral by more than 100%, with errors that differed from route to route and
    could therefore invert the ranking of two candidates. Sizing the cluster by B_r
    removes the bias and, more importantly, makes what bias remains nearly the same for
    every route, which is what the search needs.

    Cost scales with the number of in-band modes, not with the width of the peaks:
    n_base + O(span*pts_per_bw) points per mode.
    """
    lo, hi = float(band_hz[0]), float(band_hz[1])
    g = list(np.linspace(lo, hi, n_base))
    margin = 0.25 * (hi - lo)          # peaks just outside the band still shape it inside
    for p in np.atleast_1d(peaks):
        if not (lo - margin <= p <= hi + margin):
            continue
        B = modal_bandwidth_hz(p, rayleigh)
        if B <= 0.0:                   # undamped: no finite width to resolve
            continue
        half = span * B
        n_pk = int(np.ceil(2.0 * half / (B / float(pts_per_bw)))) + 1
        n_pk = int(min(max(n_pk, min_pts_per_peak), max_pts_per_peak))
        g += list(np.linspace(p - half, p + half, n_pk))
    g = np.array([x for x in g if lo <= x <= hi])
    return np.unique(np.round(g, 4))


def evaluate_mesh_power(nodes, elements, mat, sec, band_hz,
                        freq_mode="adaptive", n_band=160,
                        drive_node=0, fixed_node=None,
                        drive_dir=(0.0, 1.0, 0.0), rayleigh=(0.0, 1e-6),
                        excitation="force", prescribed_disp=(1.0, 1.0, 1.0),
                        constrain_drive_rot=False, prescribed_mask=(True, True, True),
                        KM=None, solver="direct", modal_fcut=3.0):
    """Band-integrated active input power for an EXPLICIT beam mesh.

    solver     : "direct" - banded LU at every frequency (reference).
                 "modal"  - modal superposition of the free block (prescribed
                            excitation only): one eigenproblem per mesh, then O(n)
                            per frequency. modal_fcut = None keeps the complete
                            basis (identical to "direct" to round-off); a number
                            keeps the modes below modal_fcut x f_hi of the band and
                            adds the residual-flexibility correction for the rest
                            (verified error in J below 1e-3 with modal_fcut = 3).

    Works for any node/element set (a polyline meshed by mesh_path, or a smooth
    spline meshed by geometry.mesh_spline). drive_node defaults to the first node,
    fixed_node to the last.

    excitation : "force"      - unit point force along drive_dir at the drive node.
                 "prescribed" - displacements (prescribed_disp) prescribed at the
                                drive-node translations selected by prescribed_mask;
                                unselected translations are free, and the input power
                                is delivered by the reaction forces. Only the
                                prescribed translations are fixed when the resonances
                                are computed, so the modal frequencies match the peaks
                                of the prescribed response.
    """
    if fixed_node is None:
        fixed_node = len(nodes) - 1

    # one assembly per mesh, shared by the eigenproblem and the frequency sweep
    # (previously each of them re-assembled K and M from scratch)
    if KM is None:
        KM = assemble(nodes, elements, mat, sec)

    extra = None
    if excitation == "prescribed":
        extra = [6 * drive_node + i for i in range(3) if prescribed_mask[i]]
        if constrain_drive_rot:
            extra += list(range(6 * drive_node + 3, 6 * drive_node + 6))

    basis = None
    if solver == "modal" and excitation == "prescribed":
        K, M = KM
        pres, u_p, free = _prescribed_sets(drive_node, fixed_node, prescribed_disp,
                                           K.shape[0], constrain_drive_rot, prescribed_mask)
        f_cut = None if modal_fcut is None else float(modal_fcut) * float(band_hz[1])
        basis = modal_basis_prescribed(K, M, pres, free, u_p, f_cut_hz=f_cut)
    elif solver != "direct":
        raise ValueError("solver must be 'direct' or 'modal' (modal needs excitation='prescribed')")

    if freq_mode == "adaptive":
        if basis is not None:
            # the eigenvalues of the modal basis are the peaks of the prescribed response
            f_all = np.sqrt(basis["lam"]) / (2.0 * np.pi)
            lo, hi = band_hz[0] * 0.85, band_hz[1] * 1.15
            peaks = f_all[(f_all >= lo) & (f_all <= hi)]
        else:
            peaks = natural_frequencies(nodes, elements, mat, sec, fixed_node, band_hz,
                                        extra_fixed_dofs=extra, KM=KM)
        freqs = _adaptive_freq_grid(band_hz, peaks, rayleigh)
    else:
        freqs = np.linspace(band_hz[0], band_hz[1], n_band)

    if basis is not None:
        Wbar = input_power_spectrum_modal(freqs, u_p, basis, rayleigh=rayleigh)
    elif excitation == "prescribed":
        Wbar = input_power_spectrum_prescribed(nodes, elements, mat, sec,
                                               drive_node, fixed_node, freqs,
                                               prescribed_disp=prescribed_disp,
                                               rayleigh=rayleigh,
                                               constrain_drive_rot=constrain_drive_rot,
                                               prescribed_mask=prescribed_mask, KM=KM)
    else:
        Wbar = input_power_spectrum(nodes, elements, mat, sec,
                                    drive_node, drive_dir, fixed_node,
                                    freqs, rayleigh=rayleigh, KM=KM)
    o = np.argsort(freqs)
    W = np.real(np.asarray(Wbar)[o])
    # A passive structure cannot deliver net power back through the prescribed motion,
    # so Wbar >= 0 up to round-off. Taking abs() unconditionally would turn a genuine
    # sign error into a positive cost and hide it, so the sign is checked instead.
    Wmax = float(np.max(np.abs(W))) if W.size else 0.0
    if Wmax > 0.0 and float(W.min()) < -1e-9 * Wmax:
        raise RuntimeError("evaluate_mesh_power: negative active input power "
                           "(min=%.3e, max=%.3e); check the excitation/damping setup"
                           % (float(W.min()), Wmax))
    return float(np.trapezoid(np.clip(W, 0.0, None), freqs[o]))


def evaluate_path_power(path_xyz, mat, sec, band_hz,
                        freq_mode="adaptive", n_band=160,
                        max_elem_len=0.02, drive_dir=(0.0, 1.0, 0.0),
                        rayleigh=(0.0, 1e-6),
                        excitation="force", prescribed_disp=(1.0, 1.0, 1.0),
                        constrain_drive_rot=False, prescribed_mask=(True, True, True)):
    """OBJECTIVE: band-integrated active input power for a piecewise-linear path.

    This is the single function OpenPulse would replace. Returns a positive scalar
    (lower is better). The drive node is the first path point, the support (clamp)
    is the last path point. For a smooth manufacturable centreline, mesh the path
    with geometry.mesh_spline and call evaluate_mesh_power instead.

    freq_mode :
        "adaptive" - place samples on the resonances (eigenfrequency-guided).
        "uniform"  - n_band equally spaced points (use >=128 for convergence).
    excitation : "force" or "prescribed" (see evaluate_mesh_power); prescribed_mask
                 selects which drive translations are prescribed in the latter.

    Mesh guideline: max_elem_len <= 0.02 m is converged here.
    """
    nodes, elements = mesh_path(path_xyz, max_elem_len)
    return evaluate_mesh_power(nodes, elements, mat, sec, band_hz,
                               freq_mode=freq_mode, n_band=n_band,
                               drive_dir=drive_dir, rayleigh=rayleigh,
                               excitation=excitation, prescribed_disp=prescribed_disp,
                               constrain_drive_rot=constrain_drive_rot,
                               prescribed_mask=prescribed_mask)

def band_integrated_power(nodes, elements, mat, sec, bands, weights=None, **kwargs):
    """OBJECTIVE over ONE OR SEVERAL frequency bands.

    Integration is performed with respect to frequency in hertz. Consequently, each
    unweighted band integral has units W Hz.

    `bands` is either a single (lo, hi) tuple or a list of (lo, hi) tuples. The
    band-integrated active input power is computed independently on each band (so
    the eigenfrequency-guided sampling and the trapezoidal integration stay within
    each band and never bridge the gap between them) and the contributions are
    summed. `weights`, if given, scales each band before summing; the default is
    equal weight, i.e. a plain sum of the per-band integrals.

    Note on balance: the raw integrals scale with band width and with the local
    power level, so a wider or higher-power band dominates the sum. To trade the
    bands evenly, pass weights (for example the reciprocal band widths to compare
    average power densities, or any application-driven importance).

    All other keyword arguments are forwarded unchanged to evaluate_mesh_power.
    """
    if np.isscalar(bands[0]):          # a single (lo, hi) was passed
        bands = [bands]
    if weights is None:
        weights = [1.0] * len(bands)
    if len(weights) != len(bands):
        raise ValueError("weights must have one entry per band")
    KM = kwargs.pop("KM", None)
    if KM is None:
        KM = assemble(nodes, elements, mat, sec)   # one assembly shared by all bands
    total = 0.0
    for b, w in zip(bands, weights):
        total += float(w) * evaluate_mesh_power(nodes, elements, mat, sec, b, KM=KM,
                                                **kwargs)
    return total
