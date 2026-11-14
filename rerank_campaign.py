"""
rerank_campaign.py
Re-evaluation of the best candidates of a campaign with the reference model, and the
verification tables requested for the reported route: ranking stability (search model
vs direct solver vs converged mesh), frequency-quadrature convergence, and the
geometric clearances of the as-built centreline.

    python rerank_campaign.py run_spline.py [campaign_run_spline.json.gz] [k] [h_fine_mm] [h_ref_mm]

Defaults: campaign file campaign_<stem>.json.gz, k = 15, h_fine = 2 mm, h_ref = 1 mm.
Inputs: the run script (its header is executed to rebuild lattice, objective settings
and geometry tests) and the campaign file written by run_campaign.py.
Outputs (next to the script): rerank_<stem>.csv / .tex (one row per candidate),
rerank_<stem>.txt (summary, quadrature check, clearances), and
route_final_ids_<stem>.npy (the candidate that wins on the reference model).
"""
import os, sys, io, gzip, json, time, contextlib, csv
import numpy as np
from scipy.interpolate import splev
from scipy.stats import spearmanr, kendalltau

script = sys.argv[1]
stem = os.path.splitext(os.path.basename(script))[0]
camp = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".gz") else "campaign_%s.json.gz" % stem
_rest = [a for a in sys.argv[2:] if not a.endswith(".gz")]
K = int(_rest[0]) if len(_rest) > 0 else 15
H_FINE = float(_rest[1]) * 1e-3 if len(_rest) > 1 else 0.002
H_REF = float(_rest[2]) * 1e-3 if len(_rest) > 2 else 0.001

import run_campaign as RC
ns, _var, cap, _tail = RC._exec_head(script, quiet=True)
G, geo, fem, grid = ns["G"], ns["geo"], ns["fem"], ns["grid"]
coords = grid["coords"]
mat, sec, BANDS = ns["mat"], ns["sec"], ns["BANDS"]
weights = list(ns.get("band_weights", ns.get("BAND_WEIGHTS", [1.0] * len(BANDS))))
RAY, ELEM, SMOOTH = ns["RAY"], ns["ELEM"], ns["SMOOTH"]
MODAL_FCUT = ns.get("MODAL_FCUT", 3.0)
kw = dict(freq_mode="adaptive", drive_dir=ns["DRIVE"], rayleigh=RAY, excitation=ns["EXCITATION"],
          prescribed_disp=ns["PRESCRIBED_DISP"], prescribed_mask=ns["PRESCRIBED_MASK"])
geom_fn = ns["geom_fn"]
R_MIN = ns["R_MIN"]; Lmax = ns["Lmax"]
OD = ns.get("OD", 2.0 * ns.get("OUTER_RADIUS", 0.0))

with gzip.open(camp, "rt", encoding="utf-8") as f:
    doc = json.load(f)
union = {}
for s, rec in doc["per_seed"].items():
    for path, J in rec["cache"]:
        union[tuple(path)] = J
order = sorted(union.items(), key=lambda t: t[1])
top = order[:K]
print("%s | %s | %d distinct routes evaluated by the search | top %d re-evaluated"
      % (script, camp, len(union), len(top)))
print("search model: %s solver, %.0f mm mesh, cut-off %s x f_hi | reference: direct solver, %.0f mm (check at %.0f mm)"
      % (ns.get("HARMONIC_SOLVER", "modal"), ELEM * 1e3, MODAL_FCUT, H_FINE * 1e3, H_REF * 1e3))


def power(c, h, solver="direct", fcut=MODAL_FCUT):
    nodes, elems, _ = geo.mesh_spline(c, h, smooth=SMOOTH)
    return fem.band_integrated_power(nodes, elems, mat, sec, BANDS, weights=weights,
                                     solver=solver, modal_fcut=fcut, **kw)


# ---------------------------------------------------------------- 1. ranking stability
rows = []
t0 = time.perf_counter()
for rank, (path, Js) in enumerate(top, 1):
    c = coords[list(path)]
    L, R, ok = geom_fn(c)
    Jd5 = power(c, ELEM, "direct")
    Jf = power(c, H_FINE, "direct")
    rows.append(dict(rank_search=rank, J_search=Js, J_direct=Jd5, J_fine=Jf,
                     d_dir=100 * (Jd5 / Js - 1), d_mesh=100 * (Jf / Jd5 - 1),
                     L_mm=L * 1e3, R_mm=R * 1e3, n_wp=len(path), path=list(path)))
    print("  %2d  J_search %.4e  J_direct %.4e (%+5.1f%%)  J_fine %.4e (%+5.1f%%)  L %.0f mm  R %.1f mm"
          % (rank, Js, Jd5, rows[-1]["d_dir"], Jf, rows[-1]["d_mesh"], L * 1e3, R * 1e3), flush=True)
print("  (%.0f s)" % (time.perf_counter() - t0))
Jf_all = np.array([r["J_fine"] for r in rows]); Jd_all = np.array([r["J_direct"] for r in rows])
Js_all = np.array([r["J_search"] for r in rows])
rank_fine = np.argsort(np.argsort(Jf_all)) + 1
rank_dir = np.argsort(np.argsort(Jd_all)) + 1
for r, rf, rd in zip(rows, rank_fine, rank_dir):
    r["rank_direct"] = int(rd); r["rank_fine"] = int(rf)
rho_s, _ = spearmanr(Js_all, Jf_all); tau_s, _ = kendalltau(Js_all, Jf_all)
rho_d, _ = spearmanr(Jd_all, Jf_all)
i_win = int(np.argmin(Jf_all))
summary = []
summary.append("ranking stability over the top %d: Spearman rho(search, reference) = %.3f, Kendall tau = %.3f, "
               "Spearman rho(direct on search mesh, reference) = %.3f" % (len(rows), rho_s, tau_s, rho_d))
summary.append("search winner: rank %d on the reference model (J_fine %.4e); reference winner: search rank %d (J_fine %.4e)"
               % (rank_fine[0], Jf_all[0], i_win + 1, Jf_all[i_win]))
summary.append("spread of the top %d on the reference model: max/min = %.3f; deviations search->direct %+.1f%% to %+.1f%%, "
               "direct->fine %+.1f%% to %+.1f%%"
               % (len(rows), Jf_all.max() / Jf_all.min(), min(r["d_dir"] for r in rows), max(r["d_dir"] for r in rows),
                  min(r["d_mesh"] for r in rows), max(r["d_mesh"] for r in rows)))
for s in summary:
    print(s)

# mesh check of the reference winner at h_ref
c_win = coords[rows[i_win]["path"]]
J_ref = power(c_win, H_REF, "direct")
summary.append("reference winner at %.0f mm: J = %.4e (%+.1f%% vs %.0f mm)" % (H_REF * 1e3, J_ref, 100 * (J_ref / Jf_all[i_win] - 1), H_FINE * 1e3))
print(summary[-1])
np.save("route_final_ids_%s.npy" % stem, np.asarray(rows[i_win]["path"], int))

# ---------------------------------------------------------------- 2. frequency quadrature
def quadrature_check(c, h, label):
    nodes, elems, _ = geo.mesh_spline(c, h, smooth=SMOOTH)
    KM = fem.assemble(nodes, elems, mat, sec)
    extra = [i for i in range(3) if ns["PRESCRIBED_MASK"][i]]
    out = ["quadrature check, %s, direct solver on the %.0f mm mesh:" % (label, h * 1e3)]
    tot = dict(adaptive=0.0, dense=0.0, uniform=0.0)
    for (lo, hi), w in zip(BANDS, weights):
        peaks = fem.natural_frequencies(nodes, elems, mat, sec, len(nodes) - 1, (lo, hi),
                                        extra_fixed_dofs=extra, KM=KM)
        grids = {"adaptive": fem._adaptive_freq_grid((lo, hi), peaks, RAY),
                 "dense": fem._adaptive_freq_grid((lo, hi), peaks, RAY, n_base=128, span=6.0, pts_per_bw=8.0,
                                                  min_pts_per_peak=17, max_pts_per_peak=321),
                 "uniform": np.linspace(lo, hi, int(round((hi - lo) / 0.5)) + 1)}
        vals = {}
        for name, fr in grids.items():
            Wb = fem.input_power_spectrum_prescribed(nodes, elems, mat, sec, 0, len(nodes) - 1, fr,
                                                     prescribed_disp=ns["PRESCRIBED_DISP"], rayleigh=RAY,
                                                     prescribed_mask=ns["PRESCRIBED_MASK"], KM=KM)
            vals[name] = float(np.trapezoid(np.clip(np.real(Wb), 0, None), fr))
            tot[name] += w * vals[name]
        inb = peaks[(peaks >= lo) & (peaks <= hi)]
        near_lo = peaks[np.argmin(np.abs(peaks - lo))] if len(peaks) else np.nan
        near_hi = peaks[np.argmin(np.abs(peaks - hi))] if len(peaks) else np.nan
        out.append("  band %.0f-%.0f Hz: adaptive %d pts %.4e | dense %d pts %.4e (%+.2f%%) | uniform 0.5 Hz %d pts %.4e (%+.2f%%)"
                   % (lo, hi, len(grids["adaptive"]), vals["adaptive"], len(grids["dense"]), vals["dense"],
                      100 * (vals["dense"] / vals["adaptive"] - 1), len(grids["uniform"]), vals["uniform"],
                      100 * (vals["uniform"] / vals["adaptive"] - 1)))
        out.append("    %d natural frequencies inside the band; nearest to the edges: %.0f Hz (%+.0f Hz from %.0f) and %.0f Hz (%+.0f Hz from %.0f); "
                   "modal half-power bandwidth at the edges %.1f and %.1f Hz"
                   % (len(inb), near_lo, near_lo - lo, lo, near_hi, near_hi - hi, hi,
                      fem.modal_bandwidth_hz(lo, RAY), fem.modal_bandwidth_hz(hi, RAY)))
    if len(BANDS) > 1:
        out.append("  weighted total: adaptive %.4e | dense %.4e (%+.2f%%) | uniform %.4e (%+.2f%%)"
                   % (tot["adaptive"], tot["dense"], 100 * (tot["dense"] / tot["adaptive"] - 1),
                      tot["uniform"], 100 * (tot["uniform"] / tot["adaptive"] - 1)))
    return out

n0 = len(summary)
summary += quadrature_check(c_win, H_FINE, "reference winner")
if i_win != 0:
    summary += quadrature_check(coords[rows[0]["path"]], H_FINE, "search winner")
for s in summary[n0:]:
    print(s)


# ---------------------------------------------------------------- 3. clearances of the centreline
def clearance_report(c, label):
    tck = geo.fit_spline(c, smooth=SMOOTH)
    u = np.linspace(0, 1, 4000)
    P = np.column_stack(splev(u, tck))
    T = np.column_stack(splev(u, tck, der=1)); T /= np.linalg.norm(T, axis=1)[:, None]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s_arc = np.concatenate([[0.0], np.cumsum(seg)])
    out = ["clearances of the as-built centreline, %s (pipe OD %.2f mm):" % (label, OD * 1e3)]
    # domain boundary: signed distance (positive inside)
    if "SPHERE_RADIUS" in ns:
        C, Rb = np.asarray(ns["SPHERE_CENTER"], float), float(ns["SPHERE_RADIUS"])
        d_dom = Rb - np.linalg.norm(P - C, axis=1)
        dom = "ball of radius %.0f mm" % (Rb * 1e3)
    else:
        gmin = np.asarray(ns.get("GMIN", coords.min(0)), float); gmax = np.asarray(ns.get("GMAX", coords.max(0)), float)
        d_dom = np.minimum(P - gmin, gmax - P).min(axis=1)
        dom = "box %s mm" % np.round((gmax - gmin) * 1e3).astype(int).tolist()
    out.append("  domain (%s): centreline min distance to the boundary %+.2f mm, max excursion beyond it %.2f mm, "
               "tolerance in geom_fn %.2f mm; pipe-surface clearance (minus OD/2) %+.2f mm"
               % (dom, d_dom.min() * 1e3, max(0.0, -d_dom.min()) * 1e3, ns.get("DOMAIN_TOL", float("nan")) * 1e3,
                  (d_dom.min() - 0.5 * OD) * 1e3))
    boxes = ns.get("keepout_boxes") or ([ns["keepout_box"]] if ns.get("keepout_box") is not None else [])
    for b, (lo, hi) in enumerate(boxes, 1):
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        outside = np.maximum(np.maximum(lo - P, P - hi), 0.0)
        d_out = np.linalg.norm(outside, axis=1)
        inside = np.all((P > lo) & (P < hi), axis=1)
        d_in = np.where(inside, np.minimum(P - lo, hi - P).min(axis=1), 0.0)
        d_ko = np.where(inside, -d_in, d_out)
        out.append("  keep-out %d: centreline min distance %+.2f mm (negative = inside), pipe-surface clearance %+.2f mm"
                   % (b, d_ko.min() * 1e3, (d_ko.min() - 0.5 * OD) * 1e3))
    d_self = geo.min_self_distance(tck, n=800, arc_gap=np.pi * R_MIN)[0]
    out.append("  self-clearance of the centreline %.2f mm (%.1f OD); pipe-surface self-clearance %.2f mm"
               % (d_self * 1e3, d_self / OD, (d_self - OD) * 1e3))
    R, _ = geo.min_radius(tck)
    out.append("  developed length %.1f mm (L_max %.0f mm), minimum bend radius %.2f mm (R_min %.0f mm)"
               % (geo.total_length(tck) * 1e3, Lmax * 1e3, R * 1e3, R_MIN * 1e3))
    ends = []
    for t_end, idx in ((T[0], slice(None)), (T[-1], slice(None, None, -1))):
        Tt = T[idx]; ang = np.degrees(np.arccos(np.clip(Tt @ t_end, -1, 1)))
        j = int(np.argmax(ang > 2.0)) if np.any(ang > 2.0) else len(ang) - 1
        ends.append(abs(s_arc[idx][j] - s_arc[idx][0]))
    out.append("  straight sections at the ends (tangent within 2 deg of the end direction): %.1f mm at A, %.1f mm at B"
               % (ends[0] * 1e3, ends[1] * 1e3))
    return out

n0 = len(summary)
summary += clearance_report(c_win, "reference winner")
for s in summary[n0:]:
    print(s)



# ---------------------------------------------------------------- 4. same-model comparison
# the template route selected by the run script (template_<stem>.json, written by
# run_campaign.py) and the best routes of the random controls (baseline_search_<stem>.csv
# with the path_ids column) are evaluated on the reference model, next to the winner
cmp_rows = [("ACO, reference winner", Jf_all[i_win], rows[i_win]["L_mm"], rows[i_win]["R_mm"])]
tpl = "template_%s.json" % stem
if os.path.exists(tpl):
    with open(tpl) as f:
        td = json.load(f)
    bp = np.asarray(td["base_pts"], float)
    tckb = geo.fit_spline(bp, smooth=SMOOTH, end_tangents=None)
    def power_tck(tck, h):
        nodes, elems, _ = geo.resample_by_length(tck, h)
        return fem.band_integrated_power(nodes, elems, mat, sec, BANDS, weights=weights, solver="direct", **kw)
    Jt5 = power_tck(tckb, ELEM); Jtf = power_tck(tckb, H_FINE)
    Rt, _ = geo.min_radius(tckb)
    cmp_rows.append(("template: %s" % td["base_label"], Jtf, geo.total_length(tckb) * 1e3, Rt * 1e3))
    summary.append("template (%s): J = %.4e on the search mesh (direct) and %.4e on the %.0f mm mesh; ratio to the ACO reference winner %.1f"
                   % (td["base_label"], Jt5, Jtf, H_FINE * 1e3, Jtf / Jf_all[i_win]))
bs = "baseline_search_%s.csv" % stem
if os.path.exists(bs):
    with open(bs) as f:
        rd = list(csv.DictReader(f))
    for ctl in sorted({r["control"] for r in rd}):
        best = None
        for r in rd:
            if r["control"] != ctl or not r.get("path_ids"):
                continue
            path = [int(i) for i in r["path_ids"].split()]
            c = coords[path]; Jf = power(c, H_FINE, "direct")
            L, R, _ = geom_fn(c)
            summary.append("control %s, seed %s: J_search %s -> J_fine %.4e (L %.0f mm, R %.1f mm)" % (ctl, r["seed"], r["J_control"], Jf, L * 1e3, R * 1e3))
            if best is None or Jf < best[0]:
                best = (Jf, L * 1e3, R * 1e3, r["seed"])
        if best is not None:
            cmp_rows.append(("best of %s control (seed %s)" % (ctl, best[3]), best[0], best[1], best[2]))
            summary.append("best-of-seeds %s control on the reference model: %.4e, ratio to the ACO reference winner %.2f" % (ctl, best[0], best[0] / Jf_all[i_win]))
if len(cmp_rows) > 1:
    summary.append("same-model comparison (direct solver, %.0f mm mesh):" % (H_FINE * 1e3))
    for name, J, L, R in cmp_rows:
        summary.append("  %-40s J = %.4e  (x%.1f)  L %.0f mm  R %.1f mm" % (name, J, J / cmp_rows[0][1], L, R))
    with open("comparison_%s.tex" % stem, "w") as f:
        f.write("%% route & J [W.Hz] & ratio & L [mm] & R_min [mm]  (direct solver, %.0f mm mesh)\n" % (H_FINE * 1e3))
        for name, J, L, R in cmp_rows:
            f.write("%s & %.3e & %.1f & %.0f & %.1f \\\\\n" % (name, J, J / cmp_rows[0][1], L, R))
for s_ in summary:
    pass

# ---------------------------------------------------------------- outputs
import csv
with open("rerank_%s.csv" % stem, "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["rank_search", "rank_direct", "rank_fine", "J_search", "J_direct", "J_fine", "d_dir_pct", "d_mesh_pct", "L_mm", "R_mm", "n_wp", "path_ids"])
    for r in rows:
        wtr.writerow([r["rank_search"], r["rank_direct"], r["rank_fine"], "%.6e" % r["J_search"], "%.6e" % r["J_direct"],
                      "%.6e" % r["J_fine"], "%.2f" % r["d_dir"], "%.2f" % r["d_mesh"], "%.1f" % r["L_mm"], "%.2f" % r["R_mm"],
                      r["n_wp"], " ".join(map(str, r["path"]))])
with open("rerank_%s.tex" % stem, "w") as f:
    f.write("%% rank(search) & J_search & J_direct(%.0f mm) & J_fine(%.0f mm) & rank(fine) & L [mm] & R [mm]\n" % (ELEM * 1e3, H_FINE * 1e3))
    for r in rows:
        f.write("%d & %.3e & %.3e & %.3e & %d & %.0f & %.1f \\\\\n" % (r["rank_search"], r["J_search"], r["J_direct"], r["J_fine"], r["rank_fine"], r["L_mm"], r["R_mm"]))
with open("rerank_%s.txt" % stem, "w") as f:
    f.write("\n".join(summary) + "\n")
for s_ in summary:
    if s_.startswith(("template", "control", "best-of", "same-model", "  ")) and "J =" in s_ or s_.startswith("same-model"):
        print(s_)
print("saved rerank_%s.csv, rerank_%s.tex, rerank_%s.txt, route_final_ids_%s.npy%s" % (stem, stem, stem, stem, (", comparison_%s.tex" % stem) if len(cmp_rows) > 1 else ""))
