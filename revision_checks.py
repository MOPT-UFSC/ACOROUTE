"""
revision_checks.py
Additional checks on the results of a campaign of the revised manuscript:

  A. reference objective of the winner of every seed, including the seed winners that are not
     among the fifteen candidates re-evaluated by rerank_campaign.py;
  B. random controls against the individual seeds (objectives and numbers of analyses);
  C. convergence of the frequency quadrature of the reported route and of the best route of the
     random control: adaptive grid, grid three times denser, a second refinement and a uniform
     grid with a step of one tenth of the smallest half-power bandwidth of the band;
  D. clearances of the reported route to the domain boundary and to the keep-out boxes, at the
     terminals and along the free run (a length CUT_END excluded at each terminal);
  E. geometric margins of the reported route (excursion beyond the boundary, clearance to the
     keep-outs, self-clearance, length, minimum bend radius, straight lengths at the ends).

The reported route follows the rule of the manuscript: it is the search winner when its
objective on the reference model lies within REPORT_TOL of the best of the candidates
re-evaluated by rerank_campaign.py, and the best of these candidates otherwise. The rule is
applied to rerank_<case>.csv, and the lattice indices of the reported route are written to
revision_checks/route_reported_ids_<case>.npy. Note that route_final_ids_<case>.npy, written by
rerank_campaign.py, holds the winner on the reference model, which is not the reported route
when the search winner lies within the tolerance.

It reads the files written by run_revision.py in revision_results/<case>/ (the archive
revision_results.zip of the release must be extracted there) and writes checks_<case>.json,
checks_<case>.txt, route_reported_ids_<case>.npy and route_reported_<case>.stp (centreline of
the reported route, in millimetres) in the folder revision_checks/. The
files of the eight cases of the manuscript are kept in that folder of the repository.

    python revision_checks.py all                # the eight cases, each with its reference mesh
    python revision_checks.py <case> [reference element size in mm, default 2]
    python revision_checks.py ball_4_8 1         # the ball uses the 1 mm reference mesh
"""
import sys
import os
import json
import gzip
import csv
import numpy as np
from scipy.interpolate import splev

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import run_campaign as RC
import cad_export as CAD

CUT_END = 0.020            # length excluded at each terminal in the free-run clearances [m]
REPORT_TOL = 0.02          # the search winner is reported when within this fraction of the best [-]
OUT_DIR = os.path.join(ROOT, "revision_checks")
# cases of the manuscript and element size of their reference mesh [mm]
CASES = [("rep_05_15", 2.0), ("freecube_1_3", 2.0), ("box_1_3", 2.0), ("freecube_5_7", 2.0), ("freecube_1_2_6_8", 2.0), ("stair_6_10", 2.0), ("freecube_6_10", 2.0), ("ball_4_8", 1.0)]


def reported_route(rr):
    """Reported route from the rows of rerank_<case>.csv, by the rule of the manuscript."""
    J = [float(r["J_fine"]) for r in rr]
    i_search = [int(r["rank_search"]) for r in rr].index(1)
    i_ref = int(np.argmin(J))
    i = i_search if J[i_search] <= (1.0 + REPORT_TOL) * J[i_ref] else i_ref
    kind = "search winner" if i == i_search else "reference winner"
    if i_search == i_ref:
        kind = "search winner, which is also the reference winner"
    return [int(k) for k in rr[i]["path_ids"].split()], J[i], kind


def reference_power(ns, path, h):
    geo, fem, coords = ns["geo"], ns["fem"], ns["grid"]["coords"]
    weights = list(ns.get("band_weights", ns.get("BAND_WEIGHTS", [1.0] * len(ns["BANDS"]))))
    nodes, elems, _ = geo.mesh_spline(coords[list(path)], h, smooth=ns["SMOOTH"])
    return float(fem.band_integrated_power(nodes, elems, ns["mat"], ns["sec"], ns["BANDS"], weights=weights, solver="direct", freq_mode="adaptive", drive_dir=ns["DRIVE"], rayleigh=ns["RAY"], excitation=ns["EXCITATION"], prescribed_disp=ns["PRESCRIBED_DISP"], prescribed_mask=ns["PRESCRIBED_MASK"]))


def quadrature_levels(ns, path, h):
    """Band integral of the input power on five frequency grids (weighted sum over the bands)."""
    geo, fem, coords = ns["geo"], ns["fem"], ns["grid"]["coords"]
    mat, sec, RAY, BANDS = ns["mat"], ns["sec"], ns["RAY"], ns["BANDS"]
    weights = list(ns.get("band_weights", ns.get("BAND_WEIGHTS", [1.0] * len(BANDS))))
    nodes, elems, _ = geo.mesh_spline(coords[list(path)], h, smooth=ns["SMOOTH"])
    KM = fem.assemble(nodes, elems, mat, sec)
    extra = [i for i in range(3) if ns["PRESCRIBED_MASK"][i]]
    tot, npts, steps = {}, {}, []
    for (lo, hi), w in zip(BANDS, weights):
        peaks = fem.natural_frequencies(nodes, elems, mat, sec, len(nodes) - 1, (lo, hi), extra_fixed_dofs=extra, KM=KM)
        bw_lo = RAY[0] / (2.0 * np.pi) + 2.0 * np.pi * RAY[1] * lo ** 2
        step = max(0.02, min(0.5, bw_lo / 10.0))
        steps.append(step)
        grids = {"adaptive": fem._adaptive_freq_grid((lo, hi), peaks, RAY),
                 "dense": fem._adaptive_freq_grid((lo, hi), peaks, RAY, n_base=128, span=6.0, pts_per_bw=8.0, min_pts_per_peak=17, max_pts_per_peak=321),
                 "dense2": fem._adaptive_freq_grid((lo, hi), peaks, RAY, n_base=512, span=12.0, pts_per_bw=24.0, min_pts_per_peak=65, max_pts_per_peak=1281),
                 "uniform05": np.linspace(lo, hi, int(round((hi - lo) / 0.5)) + 1),
                 "uniform_fine": np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)}
        for name, fr in grids.items():
            Wb = fem.input_power_spectrum_prescribed(nodes, elems, mat, sec, 0, len(nodes) - 1, fr, prescribed_disp=ns["PRESCRIBED_DISP"], rayleigh=RAY, prescribed_mask=ns["PRESCRIBED_MASK"], KM=KM)
            tot[name] = tot.get(name, 0.0) + w * float(np.trapezoid(np.clip(np.real(Wb), 0, None), fr))
            npts[name] = npts.get(name, 0) + len(fr)
    return tot, npts, steps


def clearance_profile(ns, path):
    """Centreline distances to the domain boundary and to the keep-out boxes, at the terminals
    and along the free run. Positive values are inside the domain and outside the keep-out."""
    geo, coords = ns["geo"], ns["grid"]["coords"]
    tck = geo.fit_spline(coords[list(path)], smooth=ns["SMOOTH"])
    u = np.linspace(0.0, 1.0, 20001)
    P = np.column_stack(splev(u, tck))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    free = (s > CUT_END) & (s < s[-1] - CUT_END)
    ro = float(ns.get("OUTER_RADIUS", 0.5 * ns.get("OD", 0.0)))
    out = {"outer_radius_mm": ro * 1e3, "length_mm": s[-1] * 1e3, "cut_end_mm": CUT_END * 1e3}
    if ns.get("SPHERE_RADIUS") is not None:
        d_dom = float(ns["SPHERE_RADIUS"]) - np.linalg.norm(P - np.asarray(ns["SPHERE_CENTER"], float), axis=1)
    else:
        gmin = np.asarray(ns.get("GMIN", coords.min(0)), float); gmax = np.asarray(ns.get("GMAX", coords.max(0)), float)
        d_dom = np.minimum(P - gmin, gmax - P).min(axis=1)
    fields = {"domain": d_dom}
    boxes = ns.get("keepout_boxes") or ([ns["keepout_box"]] if ns.get("keepout_box") is not None else [])
    for b, (lo, hi) in enumerate(boxes, 1):
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        d_out = np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)
        inside = np.all((P > lo) & (P < hi), axis=1)
        fields["keepout_%d" % b] = np.where(inside, -np.minimum(P - lo, hi - P).min(axis=1), d_out)
    for name, d in fields.items():
        i = int(np.argmin(d)); j = int(np.argmin(np.where(free, d, np.inf)))
        out[name] = {"min_mm": d[i] * 1e3, "at_s_mm": s[i] * 1e3, "at_A_mm": d[0] * 1e3, "at_B_mm": d[-1] * 1e3, "free_run_min_mm": d[j] * 1e3, "free_run_at_s_mm": s[j] * 1e3, "free_run_surface_mm": (d[j] - ro) * 1e3}
    return out


def margins(ns, path):
    """Geometric margins of a route as in the clearance report of rerank_campaign.py (same
    sampling): excursion of the centreline beyond the domain boundary, clearance of the pipe
    surface to the keep-out boxes, self-clearance of the pipe surface, developed length, minimum
    bend radius and straight lengths at the ends (tangent within 2 deg of the end direction)."""
    geo, coords = ns["geo"], ns["grid"]["coords"]
    OD = float(ns.get("OD", 2.0 * ns.get("OUTER_RADIUS", 0.0)))
    tck = geo.fit_spline(coords[list(path)], smooth=ns["SMOOTH"])
    u = np.linspace(0, 1, 4000)
    P = np.column_stack(splev(u, tck))
    T = np.column_stack(splev(u, tck, der=1)); T /= np.linalg.norm(T, axis=1)[:, None]
    s_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if ns.get("SPHERE_RADIUS") is not None:
        d_dom = float(ns["SPHERE_RADIUS"]) - np.linalg.norm(P - np.asarray(ns["SPHERE_CENTER"], float), axis=1)
    else:
        gmin = np.asarray(ns.get("GMIN", coords.min(0)), float); gmax = np.asarray(ns.get("GMAX", coords.max(0)), float)
        d_dom = np.minimum(P - gmin, gmax - P).min(axis=1)
    out = {"domain_centreline_min_mm": float(d_dom.min()) * 1e3, "domain_excursion_mm": max(0.0, -float(d_dom.min())) * 1e3, "domain_tolerance_mm": float(ns.get("DOMAIN_TOL", float("nan"))) * 1e3}
    boxes = ns.get("keepout_boxes") or ([ns["keepout_box"]] if ns.get("keepout_box") is not None else [])
    for b, (lo, hi) in enumerate(boxes, 1):
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        d_out = np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)
        inside = np.all((P > lo) & (P < hi), axis=1)
        d_ko = np.where(inside, -np.where(inside, np.minimum(P - lo, hi - P).min(axis=1), 0.0), d_out)
        out["keepout_%d_surface_mm" % b] = (float(d_ko.min()) - 0.5 * OD) * 1e3
    d_self = float(geo.min_self_distance(tck, n=800, arc_gap=np.pi * ns["R_MIN"])[0])
    out["self_clearance_surface_mm"] = (d_self - OD) * 1e3
    out["length_mm"] = float(geo.total_length(tck)) * 1e3
    out["min_bend_radius_mm"] = float(geo.min_radius(tck)[0]) * 1e3
    ends = []
    for t_end, idx in ((T[0], slice(None)), (T[-1], slice(None, None, -1))):
        ang = np.degrees(np.arccos(np.clip(T[idx] @ t_end, -1, 1)))
        j = int(np.argmax(ang > 2.0)) if np.any(ang > 2.0) else len(ang) - 1
        ends.append(abs(s_arc[idx][j] - s_arc[idx][0]))
    out["straight_A_mm"] = float(ends[0]) * 1e3; out["straight_B_mm"] = float(ends[1]) * 1e3
    return out


def check_case(case, h_mm):
    h = h_mm * 1e-3
    D = os.path.join(ROOT, "revision_results", case)
    os.makedirs(OUT_DIR, exist_ok=True)
    ns, _var, _cap, _tail = RC._exec_head(os.path.join(D, case + ".py"), quiet=True)
    out = {"case": case, "h_ref_mm": h * 1e3}
    txt = ["additional checks, %s, reference model: direct solver on the %.0f mm mesh" % (case, h * 1e3)]

    # A. seed winners
    with gzip.open(os.path.join(D, "campaign_%s.json.gz" % case), "rt", encoding="utf-8") as f:
        doc = json.load(f)
    with open(os.path.join(D, "rerank_%s.csv" % case)) as f:
        rr = list(csv.DictReader(f))
    top = {tuple(int(i) for i in r["path_ids"].split()): r for r in rr}
    ids, J_rep, kind = reported_route(rr)
    np.save(os.path.join(OUT_DIR, "route_reported_ids_%s.npy" % case), np.asarray(ids, int))
    out["reported_route"] = kind; out["reported_path_ids"] = ids
    txt.append("reported route: %s" % kind)
    seeds = []
    for sd, rec in doc["per_seed"].items():
        p = tuple(rec["best_path"]); r = top.get(p)
        seeds.append({"seed": int(sd), "J_search": float(rec["best_power"]), "n_fem": int(rec["n_fem_calls"]), "in_top": r is not None, "J_ref": float(r["J_fine"]) if r is not None else reference_power(ns, p, h)})
    seeds.sort(key=lambda e: e["J_search"])
    out["seeds"] = seeds; out["J_ref_reported"] = J_rep
    txt.append("A. seed winners (reported route: J_ref = %.4e)" % J_rep)
    for e in seeds:
        txt.append("   seed %2d: J_search %.4e, %5d analyses, %s the re-evaluated candidates, J_ref %.4e (x%.3f the reported route)" % (e["seed"], e["J_search"], e["n_fem"], "among" if e["in_top"] else "NOT among", e["J_ref"], e["J_ref"] / J_rep))

    # B. random controls
    ctl = []
    bs = os.path.join(D, "baseline_search_%s.csv" % case)
    if os.path.exists(bs):
        with open(bs) as f:
            for r in csv.DictReader(f):
                if r.get("path_ids"):
                    p = [int(i) for i in r["path_ids"].split()]
                    ctl.append({"seed": int(r["seed"]), "n_fem": int(r["fem"]), "J_search": float(r["J_control"]), "J_ref": reference_power(ns, p, h), "path": p})
    out["controls"] = [{k: v for k, v in e.items() if k != "path"} for e in ctl]
    n_tot = sum(e["n_fem"] for e in seeds)
    txt.append("B. random controls (the five ACO runs together: %d analyses)" % n_tot)
    for e in ctl:
        ratios = sorted(e["J_ref"] / s_["J_ref"] for s_ in seeds)
        txt.append("   control of seed %2d: %5d analyses, J_search %.4e, J_ref %.4e; seeds with a lower J_ref: %d of %d; ratio control/seed: min %.2f, median %.2f, max %.2f" % (e["seed"], e["n_fem"], e["J_search"], e["J_ref"], sum(1 for x in ratios if x > 1.0), len(ratios), ratios[0], ratios[len(ratios) // 2], ratios[-1]))

    # C. frequency quadrature
    q, n, steps = quadrature_levels(ns, ids, h)
    out["quadrature_reported"] = q; out["quadrature_points"] = n; out["uniform_fine_step_hz"] = steps
    txt.append("C. frequency quadrature, relative to the fine uniform grid (step %s Hz); points: %s" % (", ".join("%.3f" % x for x in steps), n))
    txt.append("   reported route: " + ", ".join("%s %+.2f%%" % (k, 100.0 * (q[k] / q["uniform_fine"] - 1.0)) for k in ("adaptive", "dense", "dense2", "uniform05")))
    if ctl:
        best = min(ctl, key=lambda e: e["J_ref"])
        qc, _n, _s = quadrature_levels(ns, best["path"], h)
        out["quadrature_control"] = qc; out["quadrature_control_seed"] = best["seed"]
        txt.append("   best control (seed %d): " % best["seed"] + ", ".join("%s %+.2f%%" % (k, 100.0 * (qc[k] / qc["uniform_fine"] - 1.0)) for k in ("adaptive", "dense", "dense2", "uniform05")))
        txt.append("   ratio control/reported: adaptive %.3f, converged %.3f" % (qc["adaptive"] / q["adaptive"], qc["uniform_fine"] / q["uniform_fine"]))

    # D. clearances
    cl = clearance_profile(ns, ids)
    out["clearances"] = cl
    txt.append("D. centreline clearances of the reported route (pipe outer radius %.3f mm; free run excludes %.0f mm at each terminal)" % (cl["outer_radius_mm"], cl["cut_end_mm"]))
    for name, v in cl.items():
        if isinstance(v, dict):
            txt.append("   %s: minimum %+.2f mm at s = %.1f mm (at A %+.2f, at B %+.2f); free run %+.2f mm at s = %.1f mm, pipe surface %+.2f mm" % (name, v["min_mm"], v["at_s_mm"], v["at_A_mm"], v["at_B_mm"], v["free_run_min_mm"], v["free_run_at_s_mm"], v["free_run_surface_mm"]))

    tck = ns["geo"].fit_spline(ns["grid"]["coords"][ids], smooth=ns["SMOOTH"])
    CAD.write_step(tck, os.path.join(OUT_DIR, "route_reported_%s.stp" % case))
    mg = margins(ns, ids)
    out["margins"] = mg
    txt.append("E. geometric margins of the reported route (sampling of rerank_campaign.py)")
    txt.append("   " + ", ".join("%s %+.2f" % (k, v) for k, v in mg.items()))

    with open(os.path.join(OUT_DIR, "checks_%s.json" % case), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT_DIR, "checks_%s.txt" % case), "w") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "all":
        for case, h_mm in CASES:
            check_case(case, h_mm)
    else:
        check_case(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)


if __name__ == "__main__":
    main()
