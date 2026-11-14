"""
geometry.py
Turn a chosen route (sequence of grid waypoints) into a smooth, manufacturable
centreline and mesh it with straight beam elements of a chosen size.

Pipeline:
    waypoints (ACO route)  ->  fit_spline  ->  parametric cubic spline r(u)
    r(u)                   ->  mesh_spline ->  nodes spaced by arc length = elem_len
    r(u)                   ->  min_radius  ->  manufacturability check

The beam element is straight, so a curved centreline is approximated by many short
straight segments: the element size controls BOTH geometric fidelity (chord error)
and numerical accuracy (resolving the bending wavelength). Run a mesh-convergence
sweep to pick it.

smooth = 0 interpolates the waypoints exactly (C2-continuous, so the sharp corners
of the polyline disappear but a tight zig-zag still yields a small radius). Increase
smooth to relax the curve away from the waypoints and raise the minimum radius, at
the cost of moving away from the optimised geometry.
"""

import numpy as np
from scipy.interpolate import splprep, splev


ENDPOINT_TOL = 1e-9      # [m] tolerance for the pinned endpoints of the fitted spline

# Module-wide adoption hook for the per-waypoint RMS smoothing parameterization.
# When set (a length in metres, e.g. 0.005), every fit_spline call that requests a
# POSITIVE absolute smoothing (smooth > 0) is re-parameterized as s = N * rms**2,
# so the search, the verification refits, the plots and the CAD export all use one
# consistent rule without threading an extra argument through every call site.
# Calls with smooth = 0 are explicit interpolation requests (analytic baselines,
# schematic figures) and are never affected. An explicit smooth_rms argument in a
# call takes precedence over this default. None = legacy behaviour everywhere.
DEFAULT_SMOOTH_RMS = None
# Optional prescribed END TANGENTS (unit 3-vectors or None), applied by fit_spline to
# every spline of the run: END_TANGENTS = (dir_A, dir_B), with dir_A the direction in
# which the curve LEAVES A and dir_B the travel direction with which it ENTERS B. A
# clamped cubic B-spline has its end tangents along (P1 - P0) and (P_{n-1} - P_{n-2}),
# so the second and the second-to-last control points are moved onto those directions
# at their original distances; the curve changes only inside the first and last knot
# spans and all downstream checks (length, radius, containment, FEM) see the clamped
# curve. Set by the run scripts (TANGENT_CLAMP); None leaves the tangents free.
END_TANGENTS = (None, None)
# fit_spline(..., end_tangents="global") uses END_TANGENTS; end_tangents=None fits with
# free tangents regardless (used for the template baselines, whose dense point sets
# would develop a kink if their millimetre-long end spans were clamped).


def fit_spline(coords, smooth=0.0, k=3, pin_endpoints=True, smooth_rms=None, end_tangents="global"):
    """Parametric cubic B-spline through (or near) the 3D waypoints.

    With smooth=0 the curve interpolates every waypoint. With smooth>0 the curve is a
    smoothing spline: it may depart from the waypoints, and scipy's splprep would move
    the FIRST and LAST points as well, which silently detaches the tube from the source
    and the support. Those two points are the physical connections of the problem and
    are not design freedoms, so the endpoints are pinned back exactly: splprep returns a
    clamped B-spline, whose curve starts at its first coefficient and ends at its last,
    so overwriting those two coefficients with the true endpoints reproduces them to
    machine precision while keeping the curve a C2 cubic B-spline. The substitution
    perturbs the curve only within the support of the two end basis functions (the
    first and last knot spans); outside those spans the curve coincides with the
    FITPACK solution. The composite curve is, however, no longer the exact optimum of
    the FITPACK smoothing problem, and its residual is not guaranteed to satisfy the
    bound `s`. Set pin_endpoints=False only to reproduce the unpinned behaviour.

    Smoothing parameterization: FITPACK's `s` bounds the SUM of squared residuals,
    so with unit weights it carries the dimension of length squared and a fixed
    `smooth` applies a TIGHTER per-point tolerance to routes with more waypoints --
    within one optimization, candidates with different waypoint counts then receive
    different degrees of smoothing. Pass `smooth_rms` (a length, in metres) instead
    to control an approximate per-waypoint RMS deviation uniformly across routes:
    it overrides `smooth` with s = N * smooth_rms**2, where N is the number of
    distinct waypoints reaching the fit. `smooth` is kept for backward
    compatibility and reproduces earlier runs bit-identically.

    end_tangents: "global" applies the module-level END_TANGENTS clamp (prescribed exit
    and arrival directions, see above); None leaves the end tangents free; a pair
    (dir_A, dir_B) clamps to those directions for this fit only.

    Returns the spline representation `tck` (pass it to the other functions here).
    """
    coords = np.asarray(coords, float)
    if len(coords) < 2:
        raise ValueError("need at least two waypoints")
    A_end, B_end = coords[0].copy(), coords[-1].copy()
    # remove consecutive duplicates (splprep dislikes zero-length spans), THEN pick the
    # spline degree from the number of distinct points that actually reach the fit
    keep = np.r_[True, np.any(np.diff(coords, axis=0) != 0.0, axis=1)]
    coords = coords[keep]
    if smooth_rms is None and smooth > 0.0:
        smooth_rms = DEFAULT_SMOOTH_RMS     # module-wide adoption hook (see above)
    # None AND 0 both mean "disabled" (fall back to the absolute `smooth`): a zero
    # RMS tolerance would otherwise request s = 0, i.e. exact interpolation, which
    # is never what disabling the re-parameterization intends. Explicit
    # interpolation remains available through smooth = 0.
    if smooth_rms:
        smooth = len(coords) * float(smooth_rms) ** 2
    kk = min(k, len(coords) - 1)
    tck, _ = splprep([coords[:, 0], coords[:, 1], coords[:, 2]],
                     s=smooth, k=kk)
    if pin_endpoints and smooth > 0.0:
        t, c, kdeg = tck
        c = [np.array(ci, float) for ci in c]
        for d in range(3):
            c[d][0] = A_end[d]
            c[d][-1] = B_end[d]
        tck = (t, c, kdeg)
        r0 = np.array(splev(0.0, tck)); r1 = np.array(splev(1.0, tck))
        if (np.linalg.norm(r0 - A_end) > ENDPOINT_TOL
                or np.linalg.norm(r1 - B_end) > ENDPOINT_TOL):
            raise RuntimeError("fit_spline: endpoints not reproduced after pinning "
                               "(|dA|=%.3e, |dB|=%.3e m)"
                               % (np.linalg.norm(r0 - A_end), np.linalg.norm(r1 - B_end)))
    dirA, dirB = END_TANGENTS if end_tangents == "global" else ((None, None) if end_tangents is None else end_tangents)
    if (dirA is not None or dirB is not None) and tck[2] >= 2 and len(tck[1][0]) >= 4:
        t, c, kdeg = tck
        c = [np.array(ci, float) for ci in c]
        P = np.column_stack(c)                      # control polygon, n x 3
        if dirA is not None:
            u = np.asarray(dirA, float); u = u / np.linalg.norm(u)
            P[1] = P[0] + np.linalg.norm(P[1] - P[0]) * u
        if dirB is not None:
            u = np.asarray(dirB, float); u = u / np.linalg.norm(u)
            P[-2] = P[-1] - np.linalg.norm(P[-1] - P[-2]) * u
        tck = (t, [P[:, 0], P[:, 1], P[:, 2]], kdeg)
    return tck


def eval_spline(tck, n=400):
    """Evaluate the spline at n points; returns (n,3) coordinates."""
    u = np.linspace(0.0, 1.0, n)
    x, y, z = splev(u, tck)
    return np.column_stack([x, y, z])


def _arc_length(curve):
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]), float(seg.sum())


def total_length(tck, n_dense=4000):
    """Arc length of the smooth curve."""
    _, L = _arc_length(eval_spline(tck, n_dense))
    return L


def resample_by_length(tck, elem_len, n_dense=4000):
    """Sample the spline at uniform arc-length spacing ~ elem_len.

    Returns (nodes [m,3], elements [m-1,2], length). Nodes lie ON the smooth curve,
    so the resulting beam mesh follows the manufacturable shape.
    """
    u_dense = np.linspace(0.0, 1.0, n_dense)
    pts = np.column_stack(splev(u_dense, tck))
    s_cum, L = _arc_length(pts)
    # elem_len is a MAXIMUM element size: round() could return one element of 1.49*elem_len
    n_elem = max(1, int(np.ceil(L / elem_len - 1e-9)))
    s_targets = np.linspace(0.0, L, n_elem + 1)
    u_at = np.interp(s_targets, s_cum, u_dense)
    nodes = np.column_stack(splev(u_at, tck))
    elements = np.array([(i, i + 1) for i in range(n_elem)], dtype=int)
    return nodes, elements, L


def mesh_spline(coords, elem_len, smooth=0.0, k=3, smooth_rms=None):
    """Convenience: waypoints -> spline -> beam mesh. Returns (nodes, elements, L).

    `smooth_rms` (metres), when given, overrides `smooth` with the per-waypoint RMS
    parameterization s = N * smooth_rms**2 (see fit_spline).
    """
    tck = fit_spline(coords, smooth=smooth, k=k, smooth_rms=smooth_rms)
    return resample_by_length(tck, elem_len)


TANGENT_TOL = 1e-12   # ||r'||^2 below this: loss of regularity (cusp), not a straight portion


def _kappa_scalar(u, tck):
    """Curvature at one parameter value."""
    d1 = np.asarray(splev(u, tck, der=1), float)
    d2 = np.asarray(splev(u, tck, der=2), float)
    q = float(np.dot(d1, d1))
    if q < TANGENT_TOL:
        return np.inf
    return float(np.linalg.norm(np.cross(d1, d2))) / q ** 1.5


def curvature(tck, n=800):
    """Curvature kappa(u) and the parameter samples u along the spline (for plots and
    profiles; the constraint uses the exact min_radius below).

    A route with two waypoints is a straight segment (spline degree 1), for which the
    second derivative does not exist; its curvature is zero and its bend radius infinite.
    A vanishing tangent is reported as INFINITE curvature (loss of regularity).
    """
    u = np.linspace(0.0, 1.0, n)
    if tck[2] < 2:
        return np.zeros(n), u
    d1 = np.array(splev(u, tck, der=1)).T
    d2 = np.array(splev(u, tck, der=2)).T
    num = np.linalg.norm(np.cross(d1, d2), axis=1)
    q = np.einsum("ij,ij->i", d1, d1)
    kappa = np.full(n, np.inf)
    ok = q >= TANGENT_TOL
    kappa[ok] = num[ok] / q[ok] ** 1.5
    return kappa, u


def _real_roots_in(coeffs, a, b, pad=1e-12):
    """Real roots of the polynomial (numpy highest-first coefficients) inside [a, b]."""
    coeffs = np.trim_zeros(np.asarray(coeffs, float), "f")
    if coeffs.size < 2:
        return np.empty(0)
    r = np.roots(coeffs)
    r = r[np.abs(r.imag) <= 1e-9 * (1.0 + np.abs(r.real))].real
    return r[(r >= a - pad) & (r <= b + pad)]


def min_radius(tck, n=800):
    """Minimum radius of curvature of the smooth curve, and where it occurs.

    On each knot span a cubic B-spline is a polynomial, so r' = a u^2 + b u + c and
    r'' = 2a u + b; the cubic term of r' x r'' cancels, so N(u) = |r' x r''|^2 has
    degree 4, as has Q(u) = |r'|^2. The squared curvature is N/Q^3, and its stationary
    points are the real roots of the degree-7 polynomial N'Q - 3NQ'. The curvature is
    evaluated at every such root inside every span and at every knot, so the candidates
    for the global maximum of kappa(u) are enumerated algebraically; the residual error
    is that of polynomial root finding, not a sampling resolution.

    Loss of regularity (vanishing tangent, e.g. a curve folding back on itself, where
    the sampled curvature is identically zero) is detected from the minimum of Q on
    each span: Q >= 0 touches zero at a double root, which np.roots returns as a
    complex pair with a small imaginary part, so the zeros of Q are NOT searched
    directly; instead Q is minimized over the real roots of Q' and the span ends, and
    the route is reported with R = 0 when that minimum falls below TANGENT_TOL.
    `n` is kept only for signature compatibility.

    Returns (R_min [m], coordinate of the tightest point); inf for a straight route.
    """
    t, c, k = tck
    if k < 2:
        return np.inf, np.array(splev(0.5, tck))
    from scipy.interpolate import PPoly
    # piecewise polynomial of each coordinate, coefficients highest-first per interval
    pp = [PPoly.from_spline((t, np.asarray(c[d], float), k)) for d in range(3)]
    x = pp[0].x
    k_best, u_best = 0.0, 0.5
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        if h <= 0.0:
            continue
        P = [pp[d].c[:, i] for d in range(3)]              # local polynomials in s = u - x_i
        D1 = [np.polyder(p, 1) for p in P]
        D2 = [np.polyder(p, 2) for p in P]
        # np.polymul strips leading zeros (a constant coordinate gives a shorter array),
        # so polynomial sums must use np.polyadd, which pads on the left
        Q = np.array([0.0])
        for d in range(3):
            Q = np.polyadd(Q, np.polymul(D1[d], D1[d]))
        N = np.array([0.0])
        for a, b in ((1, 2), (2, 0), (0, 1)):
            q = np.polysub(np.polymul(D1[a], D2[b]), np.polymul(D1[b], D2[a]))
            N = np.polyadd(N, np.polymul(q, q))
        # loss of regularity: minimum of Q = |r'|^2 over the span (roots of Q' + ends)
        s_q = np.r_[0.0, h, _real_roots_in(np.polyder(Q), 0.0, h)]
        s_q = np.clip(s_q, 0.0, h)
        q_min = np.polyval(Q, s_q)
        j = int(np.argmin(q_min))
        if q_min[j] < TANGENT_TOL:
            return 0.0, np.array(splev(x[i] + s_q[j], tck))
        stat = np.polysub(np.polymul(np.polyder(N), Q), 3.0 * np.polymul(N, np.polyder(Q)))
        s_cand = np.r_[0.0, h, _real_roots_in(stat, 0.0, h)]
        for s in s_cand:
            s = min(max(s, 0.0), h)
            q = np.polyval(Q, s)
            if q < TANGENT_TOL:
                return 0.0, np.array(splev(x[i] + s, tck))
            kap = np.sqrt(max(np.polyval(N, s), 0.0)) / q ** 1.5
            if kap > k_best:
                k_best, u_best = float(kap), float(x[i] + s)
    if k_best <= 0.0:
        return np.inf, np.array(splev(0.5, tck))
    return 1.0 / k_best, np.array(splev(u_best, tck))

def max_turn_from_pitch(h_min, r_min):
    """Conservative turn-angle pre-filter derived from the interpolating cubic spline
    through two consecutive opposite turns of angle theta on edges of length h_min:
    the vertex radius is h cos^2(theta/2) / (2 sqrt3 (3 - sqrt3) sin(theta/2)).
    Returns theta_max [deg] such that this radius equals r_min."""
    a = 2.0 * np.sqrt(3.0) * (3.0 - np.sqrt(3.0)) * r_min / h_min
    x = (np.sqrt(a * a + 4.0) - a) / 2.0
    return float(np.degrees(2.0 * np.arcsin(x)))

def min_self_distance(tck, n=800, arc_gap=None, r_min=None):
    """Minimum Euclidean distance between points of the curve whose separation ALONG
    the curve exceeds ``arc_gap`` [m], i.e. the self-clearance of the centreline.

    Motivation: a route that is a simple path on the lattice can still cross itself in
    space. Two face diagonals of one lattice cell intersect at the cell centre while
    using four distinct nodes, so the node tabu list does not prevent the crossing,
    and the smoothing fit can additionally pull nearby branches of the curve closer
    together. Physical realizability requires d_min >= OD (plus any assembly margin),
    which this function lets the caller verify a posteriori or enforce in geom_fn.

    The window ``arc_gap`` excludes pairs that are close simply because they are
    neighbours along the curve: within an arc distance of about pi*R at a bend of
    radius R, proximity is a curvature phenomenon already governed by the R_min
    constraint, and only beyond it does proximity mean genuine self-approach. The
    default is arc_gap = pi*r_min when r_min is given, else 10% of the curve length.
    For the pass criterion d_min >= clearance to be meaningful the window must satisfy
    arc_gap >= clearance (a straight line returns d_min = arc_gap by construction).

    Returns (d_min, p_i, p_j, s_i, s_j): the clearance [m], the two closest points and
    their arc-length positions. d_min = inf when no pair of samples is separated by
    more than arc_gap (curve shorter than the window).
    """
    pts = eval_spline(tck, n)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if arc_gap is None:
        arc_gap = np.pi * r_min if r_min is not None else 0.10 * s[-1]
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    D[np.abs(s[:, None] - s[None, :]) < arc_gap] = np.inf
    k = int(np.argmin(D))
    i, j = divmod(k, n)
    d = float(D[i, j])
    if not np.isfinite(d):
        return np.inf, None, None, None, None
    return d, pts[i], pts[j], float(s[i]), float(s[j])