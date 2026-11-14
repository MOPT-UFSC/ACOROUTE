"""
run_revision.py
One command for the whole revision campaign: for every case it writes a case script
(a copy of run_spline.py / run_staircase.py / run_sphere.py with the case settings),
runs the multi-start campaign, the random control at equal budget and the re-ranking
on the reference model, and moves everything the case produced into
revision_results/<case>/. Cases run one after the other (each campaign already uses
one process per seed). A case whose folder already holds rerank_<case>.txt is
skipped, so the command can be re-issued after an interruption (--force redoes it).

    python run_revision.py                       # all cases, in order of cost
    python run_revision.py box_1_3 ball_4_8      # selected cases
    python run_revision.py all --no-random       # skip the random controls
    python run_revision.py all --dry             # only write the case scripts
    python run_revision.py all --check           # only the preflight check

Cases (names are the file stems used throughout):
    rep_05_15        free cube, 15^3, 0.5-1.5 kHz   (representative case; random control on all 5 seeds)
    rep_05_15_modes  same, conventional objective (modes in band) for the controlled comparison
    freecube_1_3     free cube, 15^3, 1-3 kHz
    box_1_3          central keep-out 0.5, 15^3, 1-3 kHz
    freecube_5_7     free cube, 20^3, 5-7 kHz
    freecube_1_2_6_8 free cube, 21^3, 1-2 + 6-8 kHz, weights 5 and 1
    stair_6_10       staircase (run_staircase.py as it is)
    freecube_6_10    free cube, 23^3, 6-10 kHz
    ball_4_8         ball with central keep-out (run_sphere.py as it is)
Base scripts are taken as they are on disk; only the listed settings are overwritten
(SEEDS is forced to [3, 5, 7, 11, 13], SHOW to False, and the log name follows the
case). The random control of the non-representative cases runs with the budget of
the winning seed only (that is the comparator used in the paper); the representative
case keeps the five paired controls.
Requires in the same folder: run_campaign.py and baseline_search.py WITH the
persistence patch, rerank_campaign.py, altobjective.py, and the package modules.
"""
import os, re, sys, glob, gzip, json, shutil, subprocess, time

SEEDS = "[3, 5, 7, 11, 13]"
CASES = [
    ("rep_05_15",        "run_spline.py",    dict(NODES="(15, 15, 15)", BANDS="[(500.0, 1500.0)]",  BAND_WEIGHTS="(1.0,)",      EXCLUDE_CENTER="None"), dict(random="all")),
    ("rep_05_15_modes",  "run_spline.py",    dict(NODES="(15, 15, 15)", BANDS="[(500.0, 1500.0)]",  BAND_WEIGHTS="(1.0,)",      EXCLUDE_CENTER="None"), dict(random="none", modes=True)),
    ("freecube_1_3",     "run_spline.py",    dict(NODES="(15, 15, 15)", BANDS="[(1000.0, 3000.0)]", BAND_WEIGHTS="(1.0,)",      EXCLUDE_CENTER="None"), dict(random="winner")),
    ("box_1_3",          "run_spline.py",    dict(NODES="(15, 15, 15)", BANDS="[(1000.0, 3000.0)]", BAND_WEIGHTS="(1.0,)",      EXCLUDE_CENTER="0.5"),  dict(random="winner")),
    ("freecube_5_7",     "run_spline.py",    dict(NODES="(20, 20, 20)", BANDS="[(5000.0, 7000.0)]", BAND_WEIGHTS="(1.0,)",      EXCLUDE_CENTER="None"), dict(random="winner")),
    ("freecube_1_2_6_8", "run_spline.py",    dict(NODES="(21, 21, 21)", BANDS="[(1000.0, 2000.0), (6000.0, 8000.0)]", BAND_WEIGHTS="(5.0, 1.0)", EXCLUDE_CENTER="None"), dict(random="winner")),
    ("stair_6_10",       "run_staircase.py", dict(), dict(random="winner")),
    ("freecube_6_10",    "run_spline.py",    dict(NODES="(23, 23, 23)", BANDS="[(6000.0, 10000.0)]", BAND_WEIGHTS="(1.0,)",     EXCLUDE_CENTER="None"), dict(random="winner")),
    ("ball_4_8",         "run_sphere.py",    dict(), dict(random="winner", fine_mm="1", ref_mm="0.7")),
]
K_TOP, FINE_MM, REF_MM = "15", "2", "1"
REQUIRED = {
    "run_campaign.py": "_save_campaign",      # persistence patch applied
    "baseline_search.py": "path_ids",         # control routes written to the csv
    "rerank_campaign.py": "ranking stability",
    "altobjective.py": "def modes_in_band",
    "aco.py": "def optimize_multistart", "grid.py": "def build_grid", "geometry.py": "def fit_spline",
    "fem_beam3d.py": "def band_integrated_power", "baseline_routes.py": "def helix",
}


def preflight():
    bad = []
    for f, marker in REQUIRED.items():
        if not os.path.exists(f):
            bad.append("%s is missing from this folder" % f)
        elif marker not in open(f, encoding="utf-8", errors="ignore").read():
            bad.append("%s is not the revision version (expected '%s' in it)" % (f, marker))
    for base in sorted({c[1] for c in CASES}):
        if not os.path.exists(base):
            bad.append("%s is missing" % base)
    if bad:
        print("preflight failed:\n  - " + "\n  - ".join(bad) + "\nCopy the five files of the revision patch into this folder and run again.")
        sys.exit(1)
    print("preflight ok: all required files present with the revision versions.")


def set_var(src, name, value, script):
    pat = re.compile(r"^(%s\s*=\s*)([^#\n]*)(.*)$" % re.escape(name), re.M)
    m = pat.findall(src)
    if len(m) != 1:
        raise RuntimeError("%s: expected exactly one assignment of %s at line start, found %d" % (script, name, len(m)))
    return pat.sub(lambda mm: "%s%s  %s" % (mm.group(1), value, mm.group(3).strip()), src, count=1)


def write_case_script(case, base, settings, flags):
    src = open(base, encoding="utf-8").read()
    for k, v in settings.items():
        src = set_var(src, k, v, base)
    src = set_var(src, "SHOW", "False", base)
    src = set_var(src, "SEEDS", SEEDS, base)
    src, n = re.subn(r'log_path\s*=\s*"[^"]*"', 'log_path="%s_aco_log.txt"' % case, src)
    if n != 1:
        raise RuntimeError("%s: expected one log_path=\"...\" in the optimize_multistart call, found %d" % (base, n))
    if flags.get("modes"):
        anchor = "def objective_spline(coords):\n    nodes, elems, _ = geo.mesh_spline(coords, ELEM, smooth=SMOOTH)\n"
        if anchor not in src:
            raise RuntimeError("%s: objective_spline() does not start as expected; add the modes_in_band branch by hand (see altobjective.py)" % base)
        src = src.replace(anchor, anchor + "    if OBJECTIVE == \"modes_in_band\":\n        import altobjective as AO\n"
                          "        return AO.modes_in_band(nodes, elems, mat, sec, BANDS, band_weights, PRESCRIBED_MASK)\n", 1)
        src = re.sub(r"^(SHOW\s*=)", "OBJECTIVE = \"modes_in_band\"   # controlled comparison: conventional resonance-avoidance objective\n\\1", src, count=1, flags=re.M)
    fname = "%s.py" % case
    open(fname, "w", encoding="utf-8").write(src)
    return fname


def run(cmd, log):
    print("  $ %s" % " ".join(cmd), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ %s\n" % " ".join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            f.write(line); sys.stdout.write("    " + line); sys.stdout.flush()
        p.wait()
    if p.returncode != 0:
        print("  !! exit code %d (see %s)" % (p.returncode, log), flush=True)
    return p.returncode


def winning_seed(case):
    with gzip.open("campaign_%s.json.gz" % case, "rt", encoding="utf-8") as f:
        return int(json.load(f)["winning_seed"])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a for a in sys.argv[1:] if a.startswith("--")}
    preflight()
    if "--check" in opts:
        return
    wanted = [c for c in CASES if not args or "all" in args or c[0] in args]
    missing = [a for a in args if a != "all" and a not in {c[0] for c in CASES}]
    if missing:
        print("unknown case(s): %s" % missing); sys.exit(1)
    os.makedirs("revision_results", exist_ok=True)
    for case, base, settings, flags in wanted:
        outdir = os.path.join("revision_results", case)
        if os.path.exists(os.path.join(outdir, "rerank_%s.txt" % case)) and "--force" not in opts:
            print("\n=== %s already complete (rerank_%s.txt present); skipped, use --force to redo" % (case, case), flush=True)
            continue
        print("\n=== %s (from %s) ===" % (case, base), flush=True)
        fname = write_case_script(case, base, settings, flags)
        if "--dry" in opts:
            continue
        os.makedirs(outdir, exist_ok=True)
        log = os.path.join(outdir, "console_%s.txt" % case)
        before = {f: os.path.getmtime(f) for f in glob.glob("*")}
        t0 = time.time()
        rc = run([sys.executable, "run_campaign.py", fname, "nproc=5"], log)
        if rc == 0 and not os.path.exists("campaign_%s.json.gz" % case):
            print("  !! campaign_%s.json.gz was not written: run_campaign.py is not the patched version" % case); sys.exit(1)
        if rc == 0 and flags.get("random", "winner") != "none" and "--no-random" not in opts:
            cmd = [sys.executable, "baseline_search.py", fname, "controls=random", "nproc=5"]
            if flags.get("random") == "winner":
                cmd.append("seeds=%d" % winning_seed(case))
            run(cmd, log)
        if rc == 0:
            run([sys.executable, "rerank_campaign.py", fname, K_TOP, flags.get("fine_mm", FINE_MM), flags.get("ref_mm", REF_MM)], log)
        for f in glob.glob("*"):
            if os.path.isdir(f) or f == fname or f.endswith(".py"):
                continue
            if f not in before or os.path.getmtime(f) > before[f]:
                shutil.move(f, os.path.join(outdir, f))
        shutil.copy(fname, os.path.join(outdir, fname))
        print("=== %s done in %.0f min; results in %s" % (case, (time.time() - t0) / 60, outdir), flush=True)
    print("\nall done: zip the folder revision_results/ and send it.")


if __name__ == "__main__":
    main()
