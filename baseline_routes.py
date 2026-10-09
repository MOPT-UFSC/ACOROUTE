"""Deterministic reference routes from A to B, for comparison baselines.

The straight segment and the Monte Carlo distribution are the baselines used in the
paper. This module adds a deterministic, geometry-defined route that fills the length
budget without any optimization, so the optimized tube can be judged against a naive
space-filling route and not only against the straight line.

Two generators are provided:

helix_baseline  - an open (spindle) helix whose axis is the A-B line, tapered so it
                  starts exactly at A and ends exactly at B. It is the natural "open
                  spiral" for an open envelope, and it also handles a central keep-out
                  box: a large enough winding radius makes the coils pass around the
                  box. Among feasible helices it returns the most open one (fewest
                  turns, then largest radius) that stays within the domain, within
                  Lmax, above R_min, and clear of the keep-out boxes.

serpentine_gap_baseline - a helix cannot thread a channel whose walls block
                  whole layers with gaps on opposite sides (the staircase). There the
                  naive route is a smooth serpentine built in space through the centre
                  of each gap, contained in the design box by construction. (An earlier
                  Monte Carlo draw of the constructive operator was abandoned: on a
                  fine lattice it hangs, because a random construction rarely threads
                  the gaps and its zig-zag spline almost always breaks the bend-radius
                  limit.)

serpentine_baseline - a smooth serpentine through an ordered list of hand-placed
                  waypoints, kept for cases where a specific route is wanted; it is
                  checked and reported but not guaranteed feasible.

All return the centreline as an (N,3) array of points in metres, ready to be splined
and fed to the input-power evaluation exactly like any optimized route.
"""
import numpy as np
import geometry as geo


def _orthonormal_frame(u):
    """Two unit vectors spanning the plane perpendicular to the unit axis u."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)                 # already unit, orthogonal to u and e1
    return e1, e2


def helix_points(A, B, turns, radius, n=500, taper=True):
    """Points of a helix from A to B: axis along A-B, given turns and winding radius.

    With taper=True the radius follows sin(pi t), so it is zero at both ends (the tube
    meets A and B exactly) and widest in the middle, an open spindle shape.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    axis = B - A
    D = np.linalg.norm(axis)
    u = axis / D
    e1, e2 = _orthonormal_frame(u)
    t = np.linspace(0.0, 1.0, n)
    R = radius * np.sin(np.pi * t) if taper else radius * np.ones_like(t)
    ang = 2.0 * np.pi * turns * t
    pts = (A[None, :] + np.outer(t * D, u)
           + (R * np.cos(ang))[:, None] * e1[None, :]
           + (R * np.sin(ang))[:, None] * e2[None, :])
    pts[0] = A
    pts[-1] = B
    return pts


def _feasible(pts, Lmax, R_min, gmin=None, gmax=None, boxes=None):
    """Check a candidate centreline against the same limits an optimized route must
    satisfy: developed (spline) length, minimum bend radius, the domain bounding box,
    and any keep-out boxes. Containment and keep-out are tested on the SAMPLED SPLINE,
    not on the input waypoints: a spline through in-box waypoints can still swing far
    outside the box between them (that blind spot let an early serpentine baseline
    leave the domain by ~100 mm while passing this check). Returns (ok, length,
    R_achieved)."""
    tck = geo.fit_spline(pts, smooth=0.0, end_tangents=None)
    L = geo.total_length(tck)
    R_ach, _ = geo.min_radius(tck)
    if L > Lmax:
        return False, L, R_ach
    if R_ach < R_min:
        return False, L, R_ach
    sp = geo.eval_spline(tck, 600)
    if gmin is not None and gmax is not None:
        if (sp < np.asarray(gmin) - 1e-3).any() or (sp > np.asarray(gmax) + 1e-3).any():
            return False, L, R_ach
    if boxes:
        for lo, hi in boxes:
            if np.all((sp > np.asarray(lo)) & (sp < np.asarray(hi)), axis=1).any():
                return False, L, R_ach
    return True, L, R_ach


def helix_baseline(A, B, Lmax, R_min, gmin=None, gmax=None, boxes=None,
                   turns_list=(2, 3, 4, 5), n=500, verbose=False):
    """Most open feasible helix from A to B (fewest turns, then largest radius).

    Sweeps a small grid of turn counts and winding radii, keeps the helices that are
    feasible (within Lmax and the domain, above R_min, clear of keep-out boxes), and
    returns the most open one. Returns (pts, info); pts is None if none is feasible,
    in which case info["tried"] lists what was attempted.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    D = np.linalg.norm(B - A)
    feas = []
    tried = 0
    # The primary sweep starts at two turns, which keeps the historical baseline of the
    # unobstructed cases. A large central keep-out forces a wide winding radius to clear
    # it, and at two or more turns that radius makes the helix longer than Lmax, so when
    # the primary sweep finds nothing a fallback pass tries a fine ladder of low turn
    # counts: a single wide loop is shortest, but a box filling the central half of the
    # domain needs a fractional extra turn (about 1.2) to clear it while staying within
    # the length budget, so the ladder is sampled finely between one and two turns rather
    # than only at the integers. Established results are unchanged because the fallback
    # only runs when the primary sweep is empty, and it prefers the fewest turns, so
    # cases that were solved at one turn still are.
    for pass_turns in (tuple(turns_list), (1.0, 1.2, 1.4, 1.6, 1.8)):
        for turns in pass_turns:
            for rf in np.linspace(0.05, 0.45, 18):
                radius = rf * D
                pts = helix_points(A, B, turns, radius, n=n, taper=True)
                ok, L, R_ach = _feasible(pts, Lmax, R_min, gmin, gmax, boxes)
                tried += 1
                if ok:
                    feas.append((turns, radius, L, R_ach, pts))
        if feas:
            break
    if not feas:
        if verbose:
            print("  helix_baseline: no feasible helix over the sweep (%d tried)" % tried)
        return None, {"tried": tried, "n_feasible": 0}
    # most open: fewest turns, then largest radius
    feas.sort(key=lambda z: (z[0], -z[1]))
    turns, radius, L, R_ach, pts = feas[0]
    info = {"length": L, "turns": turns, "radius": radius, "R_min_achieved": R_ach,
            "Lmax": Lmax, "frac_Lmax": L / Lmax, "n_feasible": len(feas)}
    if verbose:
        print("  helix_baseline: turns=%.2g, radius=%.0f mm, length=%.0f mm (%.0f%% of Lmax), "
              "R_min=%.0f mm" % (turns, radius * 1e3, L * 1e3, 100 * L / Lmax, R_ach * 1e3))
    return pts, info


def serpentine_baseline(waypoints, Lmax=None, R_min=None, gmin=None, gmax=None,
                        boxes=None, densify=6, verbose=False):
    """Smooth serpentine through an ordered list of waypoints (in metres).

    For a channel that forces the route through successive gaps (the staircase), pass
    a few waypoints that thread those gaps, e.g. A, a point in the first gap, a point
    in the second gap, B. The waypoints are interpolated by the same spline used for
    optimized routes and returned densified. If limits are given they are checked and
    reported (a warning only, since the naive baseline may sit near a limit).
    """
    wp = np.asarray(waypoints, float)
    tck = geo.fit_spline(wp, smooth=0.0, end_tangents=None)
    pts = geo.eval_spline(tck, max(densify * len(wp), 200))
    if verbose and (Lmax is not None or R_min is not None):
        ok, L, R_ach = _feasible(pts, Lmax or np.inf, R_min or 0.0, gmin, gmax, boxes)
        print("  serpentine_baseline: length=%.0f mm, R_min=%.0f mm, feasible=%s"
              % (L * 1e3, R_ach * 1e3, ok))
    return pts


def serpentine_gap_baseline(A, B, boxes, gmin, gmax, Lmax, R_min,
                            smooth=0.0, verbose=False):
    """A smooth serpentine reference that threads the baffle gaps and stays INSIDE the
    design box, for the staircase domain.

    On a fine lattice a route that hugs the grid (a shortest path, or a random draw)
    turns too sharply at the baffle corners and its spline breaks the bend-radius limit,
    so a lattice route is a poor and often infeasible naive reference there. This builds
    the reference in space instead: it reads the two baffles from the keep-out boxes,
    places a waypoint at the centre of each gap at the baffle height, and threads a
    template of waypoints whose y and z grow monotonically from A to B (a monotone
    template keeps the fitted spline tame; the first version of this function
    used a flat-y template and its spline swung up to ~100 mm outside the box).
    Containment is then enforced explicitly: the fitted curve is sampled, and while any
    sample leaves the box by more than 1 mm that sample is clamped to the box and
    inserted as an extra waypoint (a few rounds suffice). The contained curve is
    returned as a DENSE polyline (200 points), so that refitting it downstream with any
    smoothing (the driver refits with its own SMOOTH for meshing, the FRF overlay and
    the reaction overlay) reproduces the same curve instead of re-introducing overshoot.
    Feasibility (length, bend radius, keep-out, containment) is measured on that refit,
    at the smoothing the driver will actually use. Containment costs bend radius: the
    unconstrained sweep achieved R of 31-44 mm, the contained one about 15-23 mm, which
    is the honest price of turning inside the box. A small ladder of waypoint templates
    is tried in order and the first feasible one is returned; (pts, info) with pts=None
    if none passes. The waypoints come from the physical geometry, so the same
    reference is obtained at any mesh resolution."""
    gmin = np.asarray(gmin, float); gmax = np.asarray(gmax, float)
    A = np.asarray(A, float); B = np.asarray(B, float)
    span = gmax - gmin
    plates = []                      # (height, slab thickness, gap centre x) per baffle
    for lo, hi in boxes:
        zc = 0.5 * (lo[2] + hi[2]); slab = hi[2] - lo[2]
        xlo = max(lo[0], gmin[0]); xhi = min(hi[0], gmax[0])
        left = lo[0] <= gmin[0] + 1e-9; right = hi[0] >= gmax[0] - 1e-9
        if left and not right:
            gcx = 0.5 * (xhi + gmax[0])                 # baffle on the left, gap right
        elif right and not left:
            gcx = 0.5 * (gmin[0] + xlo)                 # baffle on the right, gap left
        else:                                           # island: take the wider side
            gcx = (0.5 * (gmin[0] + xlo) if (xlo - gmin[0]) >= (gmax[0] - xhi)
                   else 0.5 * (xhi + gmax[0]))
        plates.append((zc, slab, gcx))
    plates.sort(key=lambda p: p[0])                     # lower baffle first
    (z_lo, s_lo, cx_lo), (z_up, s_up, cx_up) = plates[0], plates[-1]
    Ly = span[1]; Lx = gmax[0]
    xm = 0.5 * (cx_lo + cx_up); zm = 0.5 * (z_lo + z_up)

    def _contained(wps, rounds=8, tol=1e-3):
        # clamp-and-refit: interpolate, sample, insert the worst out-of-box sample
        # clamped to the box as an extra waypoint, until the curve stays inside.
        for _ in range(rounds):
            tck = geo.fit_spline(wps, smooth=0.0, end_tangents=None)
            sp = geo.eval_spline(tck, 600)
            viol = np.maximum(np.maximum(gmin - sp, sp - gmax), 0.0).max(axis=1)
            k = int(np.argmax(viol))
            if viol[k] <= tol:
                return tck, float(viol.max())
            p = np.clip(sp[k], gmin, gmax)
            d = np.linalg.norm(wps - p, axis=1); j = int(np.argmin(d))
            nxt = min(j + 1, len(wps) - 1)
            j2 = j if (j > 0 and np.linalg.norm(wps[j - 1] - p) < np.linalg.norm(wps[nxt] - p)) else j + 1
            j2 = max(1, min(len(wps) - 1, j2))
            wps = np.insert(wps, j2, p, axis=0)
        tck = geo.fit_spline(wps, smooth=0.0, end_tangents=None)
        sp = geo.eval_spline(tck, 600)
        return tck, float(np.maximum(np.maximum(gmin - sp, sp - gmax), 0.0).max())

    for f2, f6 in ((0.9, 0.3), (0.85, 0.25), (0.7, 0.3)):
        wps = np.array([
            A,
            [f2 * cx_lo, gmin[1] + Ly / 6, gmin[2] + 0.45 * (z_lo - gmin[2])],
            [cx_lo,      gmin[1] + 2 * Ly / 6, z_lo],                # through the lower gap
            [xm,         gmin[1] + 3 * Ly / 6, zm],                  # midway between baffles
            [cx_up,      gmin[1] + 4 * Ly / 6, z_up],                # through the upper gap
            [cx_up + f6 * (Lx - cx_up), gmin[1] + 5 * Ly / 6, z_up + 0.55 * (gmax[2] - z_up)],
            B,
        ], float)
        tck0, viol0 = _contained(wps)
        pts = geo.eval_spline(tck0, 200)                # dense polyline: refit-stable
        # feasibility measured on the refit the driver will actually perform
        tck = geo.fit_spline(pts, smooth=smooth, end_tangents=None)
        L = geo.total_length(tck); R, _ = geo.min_radius(tck)
        sp = geo.eval_spline(tck, 600)
        viol = float(np.maximum(np.maximum(gmin - sp, sp - gmax), 0.0).max())
        clear = not any(np.any(np.all((sp > lo) & (sp < hi), axis=1)) for lo, hi in boxes)
        if L <= Lmax and R >= R_min and clear and viol <= 2e-3:
            info = {"length": L, "R_min_achieved": R, "smooth": smooth,
                    "frac_Lmax": L / Lmax, "containment_viol": viol}
            if verbose:
                print("  serpentine_gap_baseline: length=%.0f mm (%.0f%% of Lmax), "
                      "R_min=%.1f mm, box excursion<=%.1f mm"
                      % (L * 1e3, 100 * L / Lmax, R * 1e3, viol * 1e3))
            return pts, info
    if verbose:
        print("  serpentine_gap_baseline: no feasible template found")
    return None, {"length": None}


if __name__ == "__main__":
    # quick geometric self-test on the three domain types
    print("base box 0.30^3, A and B opposite corners:")
    A = np.array([0.0, 0.0, 0.0]); B = np.array([0.30, 0.30, 0.30])
    D = np.linalg.norm(B - A); Lmax = 1.7 * D; R_min = 0.015
    gmin = np.array([0.0, 0.0, 0.0]); gmax = np.array([0.30, 0.30, 0.30])
    _, info = helix_baseline(A, B, Lmax, R_min, gmin, gmax, verbose=True)

    print("\ncentral keep-out box in the middle:")
    kbox = (np.array([0.11, 0.11, 0.11]), np.array([0.19, 0.19, 0.19]))
    _, info = helix_baseline(A, B, Lmax, R_min, gmin, gmax, boxes=[kbox], verbose=True)

    print("\nstaircase-like baffles (expect no feasible helix):")
    bx1 = (np.array([-0.01, -0.01, 0.085]), np.array([0.22, 0.17, 0.115]))
    bx2 = (np.array([0.13, -0.01, 0.185]), np.array([0.31, 0.17, 0.215]))
    gmax2 = np.array([0.30, 0.16, 0.30])
    _, info = helix_baseline(np.array([0.0, 0.08, 0.0]), np.array([0.28, 0.08, 0.30]),
                             3.0 * 0.45, R_min, gmin, gmax2, boxes=[bx1, bx2], verbose=True)
