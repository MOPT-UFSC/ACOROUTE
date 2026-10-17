"""Regression tests for geometry.min_self_distance (centreline self-clearance).

Three cases pin down the behaviour:

1. CROSSING DIAGONALS. A route that is a simple path on the lattice (four distinct
   nodes) whose two segments intersect in space, the pattern the node tabu list cannot
   prevent. The measured clearance must be essentially zero, far below any pipe OD.

2. TAPERED HELIX. The open-spindle baseline (a legitimate, feasible space-filling
   route that repeatedly returns near itself coil to coil). The coil-to-coil clearance
   must be measured as a genuinely positive distance, comfortably above the OD, and
   the arc window must NOT flag the curve as self-crossing.

3. U-BEND AT THE MINIMUM RADIUS. A hairpin: two parallel legs joined by a half-circle
   of radius exactly R_min. With the default window arc_gap = pi*R_min, the bend
   itself is excluded (proximity there is curvature, already governed by R_min) and
   the measured clearance is the leg-to-leg distance 2*R_min. No false rejection may
   occur for a curve that merely uses its curvature budget.

Run:  python3 validate_self_clearance.py
"""
import numpy as np
import geometry as geo
import baseline_routes as BL

R_MIN = 0.008          # [m] minimum bend radius used in the paper cases
OD = 2 * 0.001575      # [m] pipe outer diameter used in the paper cases
GAP = np.pi * R_MIN    # [m] default arc window

n_pass = 0


def check(name, cond, detail):
    global n_pass
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {name}: {detail}")
    assert cond, f"{name}: {detail}"
    n_pass += 1


print("SELF-CLEARANCE of the as-built centreline (geometry.min_self_distance)")

# ---- 1. crossing diagonals: simple lattice path, self-intersecting in space ----
h = 0.030                                        # one lattice pitch [m]
corners = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]], float) * h
poly = []                                        # densify each leg so the smoothing
for a, b in zip(corners[:-1], corners[1:]):      # spline follows the polyline shape
    for t in np.linspace(0.0, 1.0, 12, endpoint=False):
        poly.append(a + t * (b - a))
poly.append(corners[-1])
tck = geo.fit_spline(np.array(poly), smooth=5e-4)
d, pi, pj, si, sj = geo.min_self_distance(tck, n=800, arc_gap=GAP)
check("crossing diagonals detected", d < 1e-3,
      f"d_min = {d*1e3:.3f} mm (< 1 mm expected; OD = {OD*1e3:.2f} mm)")
check("crossing located mid-cell",
      pi is not None and np.linalg.norm(pi - np.array([0.5, 0.5, 0.0]) * h) < 0.35 * h,
      f"closest point at {np.round(pi*1e3,1)} mm (cell centre {np.round(0.5*h*1e3,1)} mm)")

# ---- 2. tapered helix baseline: legitimate self-approach, must pass clearance ----
A = np.zeros(3)
B = np.array([0.3, 0.3, 0.3])
pts = BL.helix_points(A, B, turns=2, radius=0.075, n=500)
tck_h = geo.fit_spline(pts, smooth=5e-4)
d_h, _, _, s1, s2 = geo.min_self_distance(tck_h, n=800, arc_gap=GAP)
check("helix is not flagged as crossing", np.isfinite(d_h) and d_h > OD,
      f"d_min = {d_h*1e3:.1f} mm (> OD = {OD*1e3:.2f} mm)")
check("helix clearance is coil-to-coil", d_h > 0.020 and abs(s2 - s1) > GAP,
      f"d_min = {d_h*1e3:.1f} mm between arc positions {s1*1e3:.0f} and {s2*1e3:.0f} mm")

# ---- 3. hairpin at exactly R_min: curvature proximity must not be a false positive --
# ingoing leg along +y at x=0, half-circle of radius R_min from (0,0,0) to (2R,0,0)
# bulging in +y (centre at (R,0,0)), outgoing leg along -y at x=2R
R = R_MIN
leg = 0.100
th = np.linspace(0.0, np.pi, 60)
leg_in = np.column_stack([np.zeros(40), np.linspace(-leg, 0.0, 40, endpoint=False),
                          np.zeros(40)])
bend = np.column_stack([R * (1 - np.cos(th)), R * np.sin(th), np.zeros_like(th)])
leg_out = np.column_stack([np.full(39, 2 * R), np.linspace(0.0, -leg, 40)[1:],
                           np.zeros(39)])
hairpin = np.vstack([leg_in, bend, leg_out])
tck_u = geo.fit_spline(hairpin, smooth=0.0)
R_u, _ = geo.min_radius(tck_u)
d_u, _, _, su1, su2 = geo.min_self_distance(tck_u, n=1200, arc_gap=GAP)
check("hairpin uses the radius budget", abs(R_u - R) < 0.15 * R,
      f"R_min(curve) = {R_u*1e3:.1f} mm (target {R*1e3:.0f} mm)")
check("hairpin clearance = leg separation, no false positive",
      np.isfinite(d_u) and abs(d_u - 2 * R) < 0.15 * (2 * R) and d_u > OD,
      f"d_min = {d_u*1e3:.1f} mm (expected ~2*R_min = {2*R*1e3:.0f} mm, > OD)")

# ---- semantics guard: a straight line returns d_min = arc_gap by construction ----
line = np.column_stack([np.linspace(0, 0.3, 50), np.zeros(50), np.zeros(50)])
d_l, *_ = geo.min_self_distance(geo.fit_spline(line, smooth=0.0), n=800, arc_gap=GAP)
check("straight line returns the window itself", abs(d_l - GAP) < 0.02 * GAP,
      f"d_min = {d_l*1e3:.1f} mm (arc_gap = {GAP*1e3:.1f} mm); hence require arc_gap >= clearance")

print(f"\nAll {n_pass} self-clearance assertions passed.")
