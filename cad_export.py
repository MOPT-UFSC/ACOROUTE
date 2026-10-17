"""Export the optimized centreline to CAD interchange formats (IGES / STEP).

The spline returned by geometry.fit_spline is a tck tuple (t, c, k) from FITPACK, i.e.
a clamped, non-rational B-spline: t is the full knot vector, c holds the control-point
coordinates and k is the degree. IGES entity 126 and the STEP entity
B_SPLINE_CURVE_WITH_KNOTS store exactly the same data, so the export is a change of
syntax, not an approximation: the curve handed to the CAD system is bit-for-bit the
curve that was checked against the geometric constraints and meshed for the finite
element analysis. Both writers use only the standard library and NumPy, so the
three-library dependency set of the package is preserved.

Note on what is exported: the CENTRELINE, not a swept solid. Sweeping a circular
profile along the curve is a boolean/kernel operation that needs a CAD kernel; in the
receiving CAD system it is a one-step "sweep" or "pipe" feature applied to this curve.

Typical use on a final design:
    import cad_export as CAD
    CAD.write_iges(tck, "route_best.igs", scale=1e3)   # metres -> millimetres
    CAD.write_step(tck, "route_best.stp", scale=1e3)
"""
import time

import numpy as np

__all__ = ["control_points", "write_iges", "write_step", "roundtrip_error"]


def control_points(tck):
    """Return (knots, P, k): the knot vector, the (ncp, 3) control-point array and the
    degree, from a splprep/fit_spline tck tuple.

    FITPACK allocates the coefficient arrays with the length of the knot vector and
    leaves trailing entries unused, so the meaningful control points are the first
    len(t) - k - 1 columns. Older SciPy releases return the padded arrays and newer
    ones already truncate; slicing covers both.
    """
    t, c, k = tck
    t = np.asarray(t, float)
    P = np.asarray(c, float)
    if P.ndim != 2 or P.shape[0] != 3:
        raise ValueError("expected three coefficient arrays (x, y, z); got shape %r"
                         % (P.shape,))
    ncp = len(t) - k - 1
    if P.shape[1] < ncp:
        raise ValueError("coefficient array shorter than len(t)-k-1")
    return t, P[:, :ncp].T, int(k)


# --------------------------------------------------------------------------- IGES

def _iges_lines(body, section, seq0, cols=72):
    """Split `body` into fixed 80-column IGES records of the given section letter."""
    out = []
    seq = seq0
    for i in range(0, len(body), cols):
        chunk = body[i:i + cols]
        out.append("%-*s%s%7d" % (cols, chunk, section, seq))
        seq += 1
    return out, seq


def _iges_param_lines(body, de_pointer, seq0):
    """Parameter-data records: data in columns 1-64, the owning directory-entry
    pointer right-justified in 65-72, then the section letter and sequence number."""
    out = []
    seq = seq0
    for i in range(0, len(body), 64):
        chunk = body[i:i + 64]
        out.append("%-64s%8d%s%7d" % (chunk, de_pointer, "P", seq))
        seq += 1
    return out, seq


def _fmt(x):
    """Compact but loss-free decimal for IGES/STEP: 17 significant digits round-trip
    an IEEE double exactly, and the exponent form is accepted by both formats."""
    return repr(float(x))


def write_iges(tck, path, scale=1e3, units="MM", label="CENTRELINE",
               author="", organization=""):
    """Write the centreline as an IGES entity 126 (B-spline curve).

    scale multiplies the coordinates: the package works in metres and CAD systems
    usually expect millimetres, hence the default 1e3. units must agree with scale
    and is one of "MM" or "M".
    """
    knots, P, k = control_points(tck)
    P = P * float(scale)
    ncp = P.shape[0]
    K = ncp - 1                      # upper index of the control-point sum
    unit_flag, unit_name = (2, "MM") if units.upper() == "MM" else (6, "M")

    # ---- parameter data for entity 126 ----
    # 126, K, M, PROP1(nonplanar=0), PROP2(open=0), PROP3(polynomial=1), PROP4(nonperiodic=0)
    vals = ["126", str(K), str(k), "0", "0", "1", "0"]
    vals += [_fmt(v) for v in knots]                 # K + M + 2 knots = len(t)
    vals += ["1.0"] * ncp                            # weights (non-rational)
    for p in P:
        vals += [_fmt(p[0]), _fmt(p[1]), _fmt(p[2])]
    vals += [_fmt(knots[k]), _fmt(knots[len(knots) - k - 1])]   # start/end parameter
    vals += ["0.0", "0.0", "0.0"]                    # unit normal (unused, nonplanar)
    pd_body = ",".join(vals) + ";"

    pd_lines, _ = _iges_param_lines(pd_body, 1, 1)
    n_pd = len(pd_lines)

    # ---- start section ----
    start_lines, _ = _iges_lines(
        "Pipe centreline exported by ACOPATH: optimized route as a B-spline curve.",
        "S", 1)

    # ---- global section ----
    stamp = time.strftime("%Y%m%d.%H%M%S")
    name = path.split("/")[-1]
    g = ["1H,", "1H;",
         "%dH%s" % (len(label), label),
         "%dH%s" % (len(name), name),
         "7HACOPATH", "7HACOPATH",
         "32", "38", "6", "308", "15",
         "%dH%s" % (len(label), label),
         "1.0", str(unit_flag), "%dH%s" % (len(unit_name), unit_name),
         "1", "0.0",
         "15H%s" % stamp,
         "1.0E-9", _fmt(float(np.abs(P).max())),
         "%dH%s" % (len(author), author) if author else "",
         "%dH%s" % (len(organization), organization) if organization else "",
         "11", "0", "15H%s" % stamp]
    global_lines, _ = _iges_lines(",".join(g) + ";", "G", 1)

    # ---- directory entry (two records) ----
    de1 = "%8d%8d%8d%8d%8d%8d%8d%8d%8s" % (126, 1, 0, 0, 0, 0, 0, 0, "00000000")
    de2 = "%8d%8d%8d%8d%8d%8d%8d%8s%8d" % (126, 0, 0, n_pd, 0, 0, 0,
                                           label[:8].ljust(8), 0)
    dir_lines = ["%-72s%s%7d" % (de1, "D", 1), "%-72s%s%7d" % (de2, "D", 2)]

    # ---- terminate ----
    term = "S%7dG%7dD%7dP%7d" % (len(start_lines), len(global_lines),
                                 len(dir_lines), n_pd)
    term_line = "%-72s%s%7d" % (term, "T", 1)

    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(start_lines + global_lines + dir_lines + pd_lines
                           + [term_line]) + "\n")
    return path


# --------------------------------------------------------------------------- STEP

def _knot_multiplicities(knots, tol=1e-12):
    """Compress a full knot vector into (distinct values, multiplicities), which is
    how STEP stores knots, unlike IGES and FITPACK."""
    vals, mult = [], []
    for x in knots:
        if vals and abs(x - vals[-1]) <= tol:
            mult[-1] += 1
        else:
            vals.append(float(x))
            mult.append(1)
    return vals, mult


def write_step(tck, path, scale=1e3, units="MM", label="centreline",
               author="ACOPATH", organization="ACOPATH"):
    """Write the centreline as a STEP AP203 file holding a B_SPLINE_CURVE_WITH_KNOTS.

    The curve is wrapped in the product/shape-representation structure that CAD
    systems expect, so the file imports as a wireframe curve ready to be swept.
    """
    knots, P, k = control_points(tck)
    P = P * float(scale)
    kv, km = _knot_multiplicities(knots)
    si_prefix = ".MILLI." if units.upper() == "MM" else "$"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    e = []                                   # entity lines, index = id - 1
    def add(text):
        e.append(text)
        return len(e)                        # STEP ids are 1-based

    pt_ids = [add("CARTESIAN_POINT('',(%s,%s,%s))" % (_fmt(p[0]), _fmt(p[1]), _fmt(p[2])))
              for p in P]
    curve = add("B_SPLINE_CURVE_WITH_KNOTS('%s',%d,(%s),.UNSPECIFIED.,.F.,.F.,(%s),(%s),"
                ".UNSPECIFIED.)"
                % (label, k, ",".join("#%d" % i for i in pt_ids),
                   ",".join(str(m) for m in km),
                   ",".join(_fmt(v) for v in kv)))

    length_unit = add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(%s,.METRE.))" % si_prefix)
    angle_unit = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    solid_unit = add("(NAMED_UNIT(*)SOLID_ANGLE_UNIT()SI_UNIT($,.STERADIAN.))")
    uncert = add("UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#%d,"
                 "'distance_accuracy_value','confusion accuracy')" % length_unit)
    ctx = add("(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
              "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#%d))"
              "GLOBAL_UNIT_ASSIGNED_CONTEXT((#%d,#%d,#%d))REPRESENTATION_CONTEXT('',''))"
              % (uncert, length_unit, angle_unit, solid_unit))

    gcs = add("GEOMETRIC_CURVE_SET('%s',(#%d))" % (label, curve))
    shape = add("SHAPE_REPRESENTATION('%s',(#%d),#%d)" % (label, gcs, ctx))

    app_ctx = add("APPLICATION_CONTEXT('automotive design')")
    add("APPLICATION_PROTOCOL_DEFINITION('international standard',"
        "'automotive_design',2000,#%d)" % app_ctx)
    product = add("PRODUCT('%s','%s','',(#%d))"
                  % (label, label, add("PRODUCT_CONTEXT('',#%d,'mechanical')" % app_ctx)))
    pdf = add("PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',#%d,.NOT_KNOWN.)"
              % product)
    pd = add("PRODUCT_DEFINITION('design','',#%d,#%d)"
             % (pdf, add("PRODUCT_DEFINITION_CONTEXT('part definition',#%d,'design')"
                         % app_ctx)))
    pds = add("PRODUCT_DEFINITION_SHAPE('','',#%d)" % pd)
    add("SHAPE_DEFINITION_REPRESENTATION(#%d,#%d)" % (pds, shape))

    body = "\n".join("#%d = %s;" % (i + 1, s) for i, s in enumerate(e))
    header = ("ISO-10303-21;\n"
              "HEADER;\n"
              "FILE_DESCRIPTION(('pipe centreline'),'2;1');\n"
              "FILE_NAME('%s','%s',('%s'),('%s'),'ACOPATH','ACOPATH','');\n"
              "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
              "ENDSEC;\n"
              "DATA;\n" % (path.split("/")[-1], stamp, author, organization))
    with open(path, "w", newline="\n") as fh:
        fh.write(header + body + "\nENDSEC;\nEND-ISO-10303-21;\n")
    return path


# ---------------------------------------------------------------- verification

def roundtrip_error(tck, path, scale=1e3, n=2000):
    """Read the curve back from an exported IGES or STEP file, evaluate it and return
    the maximum distance to the original spline, in metres.

    This is the check that matters: it proves the exported entity is the same curve
    that was analysed, rather than a resampled approximation of it. Run it once per
    exported design; the expected magnitude is the double-precision round-off of the
    coordinates (about 1e-16 m for the dimensions used here).
    """
    from scipy.interpolate import BSpline, splev

    txt = open(path).read()
    if txt.lstrip().startswith("ISO-10303-21"):
        import re
        m = re.search(r"B_SPLINE_CURVE_WITH_KNOTS\('[^']*',(\d+),\(([^)]*)\),"
                      r"[^,]*,[^,]*,[^,]*,\(([^)]*)\),\(([^)]*)\)", txt)
        k = int(m.group(1))
        ids = [int(s.strip()[1:]) for s in m.group(2).split(",")]
        mult = [int(s) for s in m.group(3).split(",")]
        kv = [float(s) for s in m.group(4).split(",")]
        pts = {}
        for mm in re.finditer(r"#(\d+) = CARTESIAN_POINT\('',\(([^)]*)\)\)", txt):
            pts[int(mm.group(1))] = [float(s) for s in mm.group(2).split(",")]
        P = np.array([pts[i] for i in ids])
        knots = np.repeat(kv, mult)
    else:
        pd = "".join(ln[:64] for ln in txt.splitlines() if len(ln) > 72 and ln[72] == "P")
        f = pd.split(";")[0].split(",")
        K, k = int(f[1]), int(f[2])
        ncp, nk = K + 1, K + k + 2
        knots = np.array([float(x) for x in f[7:7 + nk]])
        base = 7 + nk + ncp                       # skip the weights
        P = np.array([[float(f[base + 3 * i + j]) for j in range(3)] for i in range(ncp)])

    curve = BSpline(knots, P / float(scale), k, extrapolate=False)
    u = np.linspace(knots[k], knots[-k - 1], n)
    ref = np.column_stack(splev(u, tck))
    return float(np.abs(curve(u) - ref).max())
