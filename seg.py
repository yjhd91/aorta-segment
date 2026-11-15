#!/usr/bin/env python
"""seg.py - the reproducible entry point for the aortic segmentation pipeline.

    python seg.py check                     # is this machine able to run it?
    python seg.py run  <case>               # one case -> out_pipeline/
    python seg.py batch <root>              # every case under root
    python seg.py info <case>               # what came out
    python seg.py render <case>             # PNG views

Every run writes out_pipeline/manifest.json recording the tool version, the segmentation models actually used,
the TotalSegmentator licence state, the input hashes and the segment table - so a result can be traced back to
the exact configuration that produced it, and re-run later.

Nothing here decides geometry; the geometry lives in aorta_geom.py and the segmentation in aorta_pipeline.py.
This file is the stable front door to both.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

VERSION = "1.1.0"

# The invariants this pipeline holds, each one established by a measurement that went wrong without it.
# Printed by `check` so they travel with the tool instead of living only in the document.
CONSTRAINTS = [
    "acceptance thresholds are per segment: aorta ~171 mL, celiac ~1.6 mL, one coronary ~0.14 mL",
    "the trunk's radius comes from normal-plane rays; branches keep the slab measure - mixing them bulges ostia",
    "no global rescale to force the volume: that inflated the trunk by 15 % once",
    "the trunk continues into the straighter iliac along ITS OWN path; the other leaves from the fork point",
    "the side branch's PATH must start at the fork - changing only its radius changes nothing",
    "QC reads the radius profile from acceptance.json, never re-derives it from labels",
    "a multi-piece mesh is a failure unless it is welded or re-marched; never delete the fragment to pass",
    "boolean union of capped tubes is forbidden: use one 0.6 mm voxel grid",
    "morphological closing fills concavities only - it cannot remove a convex lump",
]
HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, "aorta_pipeline.py")
def _slicer_py():
    """Path to 3D Slicer's PythonSlicer; override with SLICER_PY."""
    p = os.environ.get("SLICER_PY")
    if p:
        return p
    for c in (r"D:\Slicer 5.12.4\bin\PythonSlicer.exe",
              "/Applications/Slicer.app/Contents/MacOS/PythonSlicer",
              "/opt/slicer/bin/PythonSlicer"):
        if os.path.exists(c):
            return c
    return "PythonSlicer"


SLICER_PY = _slicer_py()

TS_MODELS = {
    "total": ("ts_full", ["aorta.nii.gz", "iliac_artery_left.nii.gz"]),
    "renal_arteries": ("ts_visceral", ["celiac_trunk.nii.gz", "superior_mesenteric_artery.nii.gz",
                                       "renal_arteries.nii.gz"]),
    "coronary_arteries": ("ts_coronary", ["coronary_arteries.nii.gz"]),
}


def py():
    return SLICER_PY if os.path.exists(SLICER_PY) else sys.executable


def ts_license():
    try:
        out = subprocess.run([py(), "-c",
                              "from totalsegmentator.config import get_license_number;"
                              "print(get_license_number())"],
                             capture_output=True, text=True, timeout=120)
        return (out.stdout or "").strip() or "(none)"
    except Exception as e:
        return "(error: %r)" % e


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and f.tell() > limit:
                break
    return h.hexdigest()[:16]


def case_ready(case):
    return (os.path.exists(os.path.join(case, "ct.nii.gz"))
            and os.path.exists(os.path.join(case, "crop_meta.json")))


def cases_under(root):
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and case_ready(d):
            out.append(d)
    return out


def cmd_check(args):
    print("seg.py %s" % VERSION)
    print("  python        : %s" % py())
    ok = os.path.exists(SLICER_PY)
    print("  Slicer python : %s" % ("found" if ok else "NOT FOUND (falling back to %s)" % sys.executable))
    lic = ts_license()
    print("  TS licence    : %s" % lic)
    print("  licence ok    : %s" % (lic.startswith("aca_") or lic.startswith("com_")))
    for task, (folder, files) in TS_MODELS.items():
        print("  model %-18s -> %s/  (%s)" % (task, folder, "wired" if files else "-"))
    print("  geometry      : %s" % os.path.join(HERE, "aorta_geom.py"))
    print("  pipeline      : %s" % PIPE)
    print("  docs          : %s" % os.path.join(HERE, "docs", "SEG_TOOL.md"))
    print("\n  hard rules (each one bought with a measured failure, see docs/SEG_TOOL.md §7):")
    for line in CONSTRAINTS:
        print("   - %s" % line)
    return 0


def cmd_run(args):
    case = os.path.abspath(args.case)
    if not case_ready(case):
        print("not a case directory (needs ct.nii.gz and crop_meta.json): %s" % case)
        return 2
    argv = [sys.executable, PIPE, case]
    if args.no_ts:
        argv.append("--no-ts")
    if args.coronary:
        argv.append("--coronary")
        argv.append("auto")
    for e in args.extra or []:
        argv += ["--extra", e]
    t0 = time.time()
    r = subprocess.run(argv, text=True)
    if r.returncode != 0:
        print("pipeline failed with %d" % r.returncode)
        return r.returncode
    write_manifest(case, time.time() - t0, args)
    return 0


def write_manifest(case, seconds, args):
    out = os.path.join(case, "out_pipeline")
    acc_path = os.path.join(out, "acceptance.json")
    acc = json.load(open(acc_path, encoding="utf-8")) if os.path.exists(acc_path) else {}
    manifest = dict(tool="seg.py", version=VERSION, when=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    seconds=round(seconds, 1), case=os.path.basename(case),
                    inputs={}, models={}, licence=ts_license(),
                    args=dict(no_ts=args.no_ts, coronary=args.coronary,
                              extra=getattr(args, "extra", None) or []),
                    acceptance=acc)
    for f in ("ct.nii.gz", "crop_meta.json"):
        p = os.path.join(case, f)
        if os.path.exists(p):
            manifest["inputs"][f] = dict(size=os.path.getsize(p), sha256=sha256(p, limit=64 << 20))
    for task, (folder, files) in TS_MODELS.items():
        d = os.path.join(case, folder)
        manifest["models"][task] = dict(
            folder=folder, present=os.path.isdir(d),
            files=[f for f in files if os.path.exists(os.path.join(d, f))])
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("manifest -> %s" % os.path.join(out, "manifest.json"))


def cmd_batch(args):
    root = os.path.abspath(args.root)
    cases = cases_under(root)
    if not cases:
        print("no case directories under %s" % root)
        return 2
    print("%d cases under %s" % (len(cases), root))
    log_path = os.path.join(root, "seg_batch_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
    done, failed = [], []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        name = os.path.basename(case)
        acc_path = os.path.join(case, "out_pipeline", "acceptance.json")
        if args.skip_done and os.path.exists(acc_path):
            try:
                if json.load(open(acc_path, encoding="utf-8")).get("ok"):
                    print("[%d/%d] %s  skipped (already passed)" % (i, len(cases), name))
                    continue
            except Exception:
                pass
        line = "[%d/%d] %s" % (i, len(cases), name)
        print(line, flush=True)
        open(log_path, "a", encoding="utf-8").write(line + "\n")
        t1 = time.time()
        argv = [sys.executable, PIPE, case]
        if args.no_ts:
            argv.append("--no-ts")
        if args.coronary:
            argv += ["--coronary", "auto"]
        r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
        acc = {}
        if os.path.exists(acc_path):
            try:
                acc = json.load(open(acc_path, encoding="utf-8"))
            except Exception:
                acc = {}
        ok = bool(acc.get("ok"))
        (done if ok else failed).append(name)
        summary = "    %s  %.0fs  %.1f mL  comps %s  missing %s" % (
            "PASS" if ok else "FAIL", time.time() - t1, acc.get("mesh_mL", 0),
            acc.get("components", "?"), ",".join(acc.get("missing", []) or []) or "-")
        print(summary, flush=True)
        open(log_path, "a", encoding="utf-8").write(summary + "\n")
        write_manifest(case, time.time() - t1, args)
    print("\n%d/%d passed in %.1f min" % (len(done), len(cases), (time.time() - t0) / 60))
    if failed:
        print("failed: %s" % ", ".join(failed))
    print("log -> %s" % log_path)
    return 0


def cmd_info(args):
    case = os.path.abspath(args.case)
    for f in ("acceptance.json", "manifest.json"):
        p = os.path.join(case, "out_pipeline", f)
        if not os.path.exists(p):
            print("%s: (missing)" % f)
            continue
        d = json.load(open(p, encoding="utf-8"))
        if f == "acceptance.json":
            print("mesh      : %.1f mL  %d faces  watertight %s  components %s  ok %s" %
                  (d.get("mesh_mL", 0), d.get("faces", 0), d.get("watertight"), d.get("components"),
                   d.get("ok")))
            print("segments  :")
            for k, v in sorted((d.get("segments") or {}).items()):
                print("   %-16s nodes %4d  r %.2f mm  mask %.2f mL" %
                      (k, v.get("nodes", 0), v.get("r_med", 0), v.get("mask_mL", 0)))
            if d.get("missing"):
                print("missing   : %s" % ", ".join(d["missing"]))
        else:
            print("manifest  : seg.py %s, %s, %.0fs, licence %s" %
                  (d.get("version"), d.get("when"), d.get("seconds", 0), d.get("licence")))
    return 0


def cmd_render(args):
    case = os.path.abspath(args.case)
    out = os.path.join(case, "out_pipeline", "model.stl")
    if not os.path.exists(out):
        print("no model.stl in %s" % case)
        return 2
    with open(os.path.join(HERE, "render_job.txt"), "w", encoding="utf-8") as f:
        f.write(case)
    r = subprocess.run([py(), os.path.join(os.environ.get("USERPROFILE", ""), ".dsh", "skills", "slicer",
                                           "scripts", "slicer.py"), "--file",
                        os.path.join(HERE, "render_any.py")], text=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(prog="seg.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify this machine can run the pipeline").set_defaults(fn=cmd_check)

    p = sub.add_parser("run", help="process one case")
    p.add_argument("case")
    p.add_argument("--no-ts", action="store_true", help="reuse the cached TotalSegmentator output")
    p.add_argument("--coronary", action="store_true", help="also reconstruct the coronary arteries")
    p.add_argument("--extra", action="append", metavar="name=file.nii.gz",
                   help="add any other vessel mask")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("batch", help="process every case under a directory")
    p.add_argument("root")
    p.add_argument("--no-ts", action="store_true")
    p.add_argument("--coronary", action="store_true")
    p.add_argument("--skip-done", action="store_true", help="skip cases whose acceptance already passed")
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("info", help="summarise a finished case")
    p.add_argument("case")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("render", help="write PNG views of a finished case")
    p.add_argument("case")
    p.set_defaults(fn=cmd_render)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
