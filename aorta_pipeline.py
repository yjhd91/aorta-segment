#!/usr/bin/env python
"""aorta_pipeline.py - one command from a CTA case directory to a fused vessel tree.

    python aorta_pipeline.py <case_dir> [--extra name=file.nii.gz ...] [--no-ts]

A "case directory" is whatever already holds the study, as long as it contains:

    ct.nii.gz          the CTA volume
    crop_meta.json     {"sp_zyx": [...], "org_c": [...], "crop": [[k0,k1],[i0,i1],[j0,j1]]}

Pipeline (every step validated on a real case, see docs/AORTA_PIPELINE.md):

  1  segment      TotalSegmentator for the trunk, the arch branches and the iliacs; a minimum-cost path
                  to an organ for the celiac / renals; a downward track for the SMA.  Any vessel that
                  TotalSegmentator does not provide can be dropped in as an extra mask (this is the
                  extension point, and how the coronary interface is used).
  2  centre lines the medial axis of each vessel, ordered as a graph diameter (never a nearest-neighbour
                  walk: that zig-zags and inflates a swept tube by ~50 %).
  3  radius       area of the mask cut by a plane normal to the centre line; for the trunk the smaller of
                  that and the inscribed sphere times an ellipticity factor calibrated on the straight
                  stations, because a tangential slab inflates a bent vessel.
  4  calibre      each segment's radius profile is scaled so the swept tube's volume equals the mask
                  volume exactly - the shape stays the sweep, only the calibre is pinned.
  5  junctions    branches are extended at constant radius past the trunk's centre line; the trunk tapers
                  over its last 25 mm to r = sqrt(rL^2 + rR^2) and the two iliacs leave from that point.
  6  union        every tube is rasterised into one 0.6 mm grid and OR-ed.  A boolean union of capped
                  tubes leaves a collar at every joint; a shared voxel grid cannot.
  7  surface      the signed distance field of that grid is blurred by 1.8 mm and the zero level set is
                  extracted, which rounds the concave creases of the joints; the mesh is then cleaned and
                  lightly Taubin-smoothed.
  8  deliver      model.stl, model.step, a self-contained orbit viewer, and acceptance.json with per-segment
                  volume / radius / connectivity.

Adding a vessel later is a one-line entry in SEGMENTS, or `--extra` for a mask you already have.
"""
import argparse
import base64
import json
import os
import subprocess
import sys

import numpy as np
import scipy.ndimage as ndi
import trimesh
import nibabel as nib
from skimage import measure
from skimage.morphology import skeletonize
from scipy.interpolate import UnivariateSpline
from skimage.graph import route_through_array

HERE = os.path.dirname(os.path.abspath(__file__))
import aorta_geom as G
sys.path.insert(0, HERE)

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

GRID = 0.6            # mm: voxel size of the union grid
BAND = 1.5            # mm: half thickness of the slab used to measure a radius
SIG_FIELD = 1.0       # mm: blur of the signed field, rounds the joints
TAUBIN = 40

# --------------------------------------------------------------------------------------------------
# the segment table: adding a vessel = adding a row
# --------------------------------------------------------------------------------------------------
SEGMENTS = [
    dict(key="aorta", label=1, role="trunk", ts=("aorta",)),
    dict(key="brachiocephalic", label=2, role="branch", ts=("brachiocephalic_trunk",)),
    dict(key="lcca", label=3, role="branch", ts=("common_carotid_artery_left",)),
    dict(key="lsa", label=4, role="branch", ts=("subclavian_artery_left",)),
    dict(key="iliac_l", label=5, role="bifurcation", ts=("iliac_artery_left",)),
    dict(key="iliac_r", label=6, role="bifurcation", ts=("iliac_artery_right",)),
    dict(key="celiac", label=7, role="branch", ts=(), tracer="organ", organ="pancreas"),
    dict(key="sma", label=8, role="branch", ts=(), tracer="downward"),
    dict(key="renal_l", label=9, role="branch", ts=(), tracer="organ", organ="kidney_left"),
    dict(key="renal_r", label=10, role="branch", ts=(), tracer="organ", organ="kidney_right"),
    # --- reserved interfaces: declare them now, enable when the data becomes available -----------
    dict(key="coronary_l", label=20, role="branch", ts=("left_coronary_artery",), enabled=False, min_mL=0.06,
         note="TotalSegmentator's coronary_arteries task needs a licence; alternatively pass a mask "
              "with --extra coronary_l=<mask.nii.gz>"),
    dict(key="coronary_r", label=21, role="branch", ts=("right_coronary_artery",), enabled=False, min_mL=0.06,
         note="same as coronary_l"),
]

ORGAN_FILES = ("kidney_left", "kidney_right", "liver", "spleen", "pancreas", "small_bowel")


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------------
# 1  segmentation
# --------------------------------------------------------------------------------------------------
def run_totalsegmentator(case, ct, task="total", fast=True):
    out = os.path.join(case, "ts_full")
    need = [os.path.join(out, n + ".nii.gz") for n in
            ("aorta", "brachiocephalic_trunk", "common_carotid_artery_left",
             "subclavian_artery_left", "iliac_artery_left", "iliac_artery_right")]
    if all(os.path.exists(p) for p in need):
        log("  TotalSegmentator: cached")
        return out, True
    py = SLICER_PY
    if not os.path.exists(py):
        py = sys.executable
    os.makedirs("G:/tmp/ts", exist_ok=True)
    os.makedirs("G:/tmp/joblib", exist_ok=True)
    env = dict(os.environ, TMP="G:/tmp/ts", TEMP="G:/tmp/ts", JOBLIB_TEMP_FOLDER="G:/tmp/joblib")
    cmd = [py, os.path.join(HERE, "run_ts.py"), ct, out, task, "1" if fast else "0"]
    log("  TotalSegmentator: running (%s) ..." % task)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    ok = all(os.path.exists(p) for p in need)
    if not ok:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        log("  TotalSegmentator FAILED: %s" % " | ".join(tail))
    return out, ok


def load_crop(path, crop, shape):
    if not os.path.exists(path):
        return None
    a = np.asanyarray(nib.load(path).dataobj) > 0
    (z0, z1), (y0, y1), (x0, x1) = crop
    m = np.ascontiguousarray(np.transpose(a[y0:y1, x0:x1, z0:z1], (2, 0, 1)))
    return m if m.shape == shape else None


def trace_organ(vol, seg, shape, SP, lb, trunk, organ):
    """minimum-cost path from the aortic wall to the organ, keeping the contrast run"""
    if organ is None or organ.sum() < 100 or not trunk.any():
        return False
    hu = vol.astype(np.float32)
    cost = np.full_like(hu, 8.0)
    cost[(hu > 150) & (hu < 420)] = 1.0
    cost[(hu >= 420)] = 12.0
    cost[(hu < -150)] = 15.0
    cost = ndi.gaussian_filter(cost, 0.8)
    q = np.argwhere(trunk)
    oc = np.argwhere(organ).mean(axis=0).astype(int)
    d = ((q - oc) ** 2).sum(axis=1)
    p_a = q[int(np.argmin(d))]
    lo = np.maximum(np.minimum(p_a, oc) - 22, 0)
    hi = np.minimum(np.maximum(p_a, oc) + 22, np.array(shape))
    box = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    try:
        path, _ = route_through_array(cost[box], tuple((p_a - lo).astype(int)),
                                      tuple((oc - lo).astype(int)), fully_connected=True,
                                      geometric=True)
    except Exception:
        return False
    path = np.array(path, float) + lo
    hp = np.array([vol[tuple(int(v) for v in p)] for p in path])
    good = hp > 140
    best, i = (0, 0, 0), 0
    while i < len(good):
        if good[i]:
            j = i
            while j + 1 < len(good) and good[j + 1]:
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, i, j)
            i = j + 1
        else:
            i += 1
    if best[0] < 6:
        return False
    seg[seg == lb] = 0
    seg[_tube_mask(shape, SP, path[best[1]:best[2] + 1], 2.8)] = lb
    return True


def trace_downward(vol, seg, shape, SP, lb, k_start, trunk):
    """SMA: seed anterior to the aortic wall and follow the brightest voxel downward"""
    for k in range(k_start - 4, k_start - 24, -2):
        if k < 5:
            break
        sl = trunk[k]
        if not sl.any():
            continue
        idx = np.argwhere(sl)
        ymax = int(idx[:, 0].max())
        band = idx[idx[:, 0] >= ymax - 2]
        seed = None
        for y in range(ymax + 3, ymax + 13):
            for x in range(int(band[:, 1].min()), int(band[:, 1].max()) + 1):
                if 130 < vol[k, y, x] < 460:
                    seed = (k, y, x)
                    break
            if seed:
                break
        if not seed:
            continue
        cur = np.array(seed, float)
        path = [cur.copy()]
        for _ in range(90):
            kk = int(round(cur[0])) - 1
            if kk < 5:
                break
            best_v, best_p = 0, None
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    yy, xx = int(round(cur[1])) + dy, int(round(cur[2])) + dx
                    if not (0 <= yy < shape[1] and 0 <= xx < shape[2]):
                        continue
                    v = vol[kk, yy, xx]
                    if 120 < v < 470 and v > best_v:
                        best_v, best_p = v, (kk, yy, xx)
            if best_p is None:
                break
            cur = np.array(best_p, float)
            path.append(cur.copy())
        if len(path) >= 10:
            seg[seg == lb] = 0
            seg[_tube_mask(shape, SP, np.array(path, float), 2.9)] = lb
            return True
    return False


def _tube_mask(shape, SP, path, r_mm):
    out = np.zeros(shape, bool)
    rad = max(r_mm / float(min(SP)), 0.9)
    for p in path:
        lo = np.maximum(np.floor(p - rad).astype(int), 0)
        hi = np.minimum(np.ceil(p + rad).astype(int) + 1, np.array(shape))
        if (hi <= lo).any():
            continue
        g = np.ogrid[0:hi[0] - lo[0], 0:hi[1] - lo[1], 0:hi[2] - lo[2]]
        g = [g[0] + (lo[0] - p[0]), g[1] + (lo[1] - p[1]), g[2] + (lo[2] - p[2])]
        d2 = (g[0] * SP[0]) ** 2 + (g[1] * SP[1]) ** 2 + (g[2] * SP[2]) ** 2
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= (d2 <= r_mm * r_mm)
    return out


VISCERAL_TS = {"celiac": "celiac_trunk", "sma": "superior_mesenteric_artery"}
VISCERAL_DIR = "ts_visceral"


def run_visceral_ts(case, ct):
    """the dedicated model; needs the free academic licence, and says so if it is missing"""
    out = os.path.join(case, VISCERAL_DIR)
    want = [os.path.join(out, n + ".nii.gz") for n in
            ("celiac_trunk", "superior_mesenteric_artery", "renal_arteries")]
    if all(os.path.exists(w) for w in want):
        return out, True
    py = SLICER_PY
    if not os.path.exists(py):
        py = sys.executable
    os.makedirs("G:/tmp/ts", exist_ok=True)
    os.makedirs("G:/tmp/joblib", exist_ok=True)
    env = dict(os.environ, TMP="G:/tmp/ts", TEMP="G:/tmp/ts", JOBLIB_TEMP_FOLDER="G:/tmp/joblib")
    log("  visceral arteries: running the TotalSegmentator renal_arteries model ...")
    r = subprocess.run([py, os.path.join(HERE, "run_ts.py"), ct, out, "renal_arteries", "0"],
                       env=env, capture_output=True, text=True)
    ok = all(os.path.exists(w) for w in want)
    if not ok:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        log("  visceral model unavailable: %s" % " | ".join(tail))
    return out, ok


CORONARY_DIR = "ts_coronary"
CORONARY_FILE = "coronary_arteries.nii.gz"


def run_coronary_ts(case, ct):
    out = os.path.join(case, CORONARY_DIR)
    want = os.path.join(out, CORONARY_FILE)
    if os.path.exists(want):
        return out, True
    py = SLICER_PY
    if not os.path.exists(py):
        py = sys.executable
    os.makedirs("G:/tmp/ts", exist_ok=True)
    os.makedirs("G:/tmp/joblib", exist_ok=True)
    env = dict(os.environ, TMP="G:/tmp/ts", TEMP="G:/tmp/ts", JOBLIB_TEMP_FOLDER="G:/tmp/joblib")
    log("  coronary arteries: running the TotalSegmentator coronary_arteries model ...")
    r = subprocess.run([py, os.path.join(HERE, "run_ts.py"), ct, out, "coronary_arteries", "0"],
                       env=env, capture_output=True, text=True)
    ok = os.path.exists(want)
    if not ok:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        log("  coronary model unavailable: %s" % " | ".join(tail))
    return out, ok


def load_coronary_ts(cdir, crop, shape, seg, SP, enabled):
    if not ({"coronary_l", "coronary_r"} & enabled):
        return seg, {}
    m = load_crop(os.path.join(cdir, CORONARY_FILE), crop, shape)
    if m is None or m.sum() < 200:
        log("  coronary mask missing or too small")
        return seg, {}
    lc, nc = ndi.label(m, structure=ndi.generate_binary_structure(3, 1))
    sizes = np.bincount(lc.ravel())
    order = [i for i in np.argsort(sizes[1:])[::-1] + 1 if sizes[i] >= 150]
    got = {}
    lb_l = next(s_["label"] for s_ in SEGMENTS if s_["key"] == "coronary_l")
    lb_r = next(s_["label"] for s_ in SEGMENTS if s_["key"] == "coronary_r")
    if len(order) >= 2:
        cx = [float(np.argwhere(lc == i)[:, 2].mean()) for i in order[:2]]
        left, right = (order[0], order[1]) if cx[0] >= cx[1] else (order[1], order[0])
        seg[lc == left] = lb_l
        seg[lc == right] = lb_r
        got["coronary_l"] = int((lc == left).sum())
        got["coronary_r"] = int((lc == right).sum())
    else:
        seg[lc == order[0]] = lb_l          # a single trunk; call it the left
        got["coronary_l"] = int((lc == order[0]).sum())
    for k, v in got.items():
        log("  coronary %-10s %.2f mL" % (k, v * float(np.prod(SP)) / 1000))
    return seg, got


def load_visceral_ts(vdir, crop, shape, seg, SP, enabled):
    """celiac and SMA by name; the two renal arteries split by connected component"""
    got = {}
    for key, fname in VISCERAL_TS.items():
        if key not in enabled:
            continue
        m = load_crop(os.path.join(vdir, fname + ".nii.gz"), crop, shape)
        if m is None or m.sum() < 40:
            continue
        lb = next(s_["label"] for s_ in SEGMENTS if s_["key"] == key)
        seg[m] = lb
        got[key] = int(m.sum())
    if "renal_l" in enabled and "renal_r" in enabled:
        m = load_crop(os.path.join(vdir, "renal_arteries.nii.gz"), crop, shape)
        if m is not None and m.sum() >= 80:
            lc, nc = ndi.label(m, structure=ndi.generate_binary_structure(3, 1))
            sizes = np.bincount(lc.ravel())
            order = [i for i in np.argsort(sizes[1:])[::-1] + 1 if sizes[i] >= 60][:2]
            if len(order) == 2:
                # left is the one with the larger x (LPS)
                cx = [np.argwhere(lc == i)[:, 2].mean() for i in order]
                pairs = sorted(zip(order, cx), key=lambda t: -t[1])
                lb_r = next(s_["label"] for s_ in SEGMENTS if s_["key"] == "renal_r")
                lb_l = next(s_["label"] for s_ in SEGMENTS if s_["key"] == "renal_l")
                seg[lc == pairs[0][0]] = lb_r
                seg[lc == pairs[1][0]] = lb_l
                got["renal_l"] = int((lc == pairs[1][0]).sum())
                got["renal_r"] = int((lc == pairs[0][0]).sum())
    return seg, got


def build_segmentation(case, ts_dir, crop, vol, SP, extras, enabled_keys):
    shape = vol.shape
    seg = np.zeros(shape, np.int16)
    missing = []
    for s in SEGMENTS:
        if s["key"] not in enabled_keys:
            continue
        if not s["ts"]:
            continue
        m = None
        for name in s["ts"]:
            m = load_crop(os.path.join(ts_dir, name + ".nii.gz"), crop, shape)
            if m is not None:
                break
        if m is None:
            missing.append(s["key"])
            continue
        seg[m] = s["label"]
    if missing:
        log("  missing TotalSegmentator classes: %s" % ", ".join(missing))
    # extras: masks supplied by the caller (coronaries, any other vessel)
    for key, path in extras.items():
        m = load_crop(path, crop, shape)
        if m is None:
            log("  extra '%s' could not be read or has the wrong shape" % key)
            continue
        lb = next((s["label"] for s in SEGMENTS if s["key"] == key), None)
        if lb is None:
            lb = 30 + len([k for k in extras if k != key])
            SEGMENTS.append(dict(key=key, label=lb, role="branch", ts=()))
        seg[m] = lb
        log("  extra '%s' ingested as label %d (%.2f mL)" % (key, lb, float(m.sum()) * float(np.prod(SP)) / 1000))
    # the dedicated model first; the tracers below are only the fallback
    cdir = os.path.join(case, CORONARY_DIR)
    if os.path.isdir(cdir):
        seg, got_c = load_coronary_ts(cdir, crop, shape, seg, SP, enabled_keys)
    vis_ok = False
    vdir = os.path.join(case, VISCERAL_DIR)
    if os.path.isdir(vdir):
        seg, got = load_visceral_ts(vdir, crop, shape, seg, SP, enabled_keys)
        vis_ok = len(got) >= 3
        if got:
            log("  visceral model produced: %s" % ", ".join("%s %.2f mL" % (k, v * float(np.prod(SP)) / 1000)
                                                            for k, v in sorted(got.items())))
    # tracers for the visceral branches (fallback)
    trunk = seg == 1
    organs = {}
    for nm in ORGAN_FILES:
        organs[nm] = load_crop(os.path.join(ts_dir, nm + ".nii.gz"), crop, shape)
    for s in SEGMENTS:
        if s["key"] not in enabled_keys or s["ts"]:
            continue
        if vis_ok and s["key"] in ("celiac", "sma", "renal_l", "renal_r"):
            continue
        if s.get("tracer") == "organ":
            ok = trace_organ(vol, seg, shape, SP, s["label"], trunk, organs.get(s["organ"]))
            log("  traced %-8s -> %s" % (s["key"], "ok" if ok else "not found"))
        elif s.get("tracer") == "downward":
            k_cel = int(np.argwhere(seg == 7)[:, 0].min()) if (seg == 7).any() else 600
            ok = trace_downward(vol, seg, shape, SP, s["label"], k_cel, trunk)
            log("  traced %-8s -> %s" % (s["key"], "ok" if ok else "not found"))
    return seg


# --------------------------------------------------------------------------------------------------
# 1c  coronary tracer (no licensed TotalSegmentator task needed)
# --------------------------------------------------------------------------------------------------
def trace_coronary(vol, seg, SP, ORG, trunk_path, trunk_r, label_l=20, label_r=21, root_len=45.0):
    """look for coronary ostia on the aortic root's circumference and follow them outward"""
    if trunk_path is None or len(trunk_path) < 8:
        return {}
    st = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(trunk_path, axis=0), axis=1))])
    # the root is the inferior end of the ascending aorta
    root_end = 0 if trunk_path[0][2] < trunk_path[-1][2] else len(trunk_path) - 1
    order = range(len(trunk_path)) if root_end == 0 else range(len(trunk_path) - 1, -1, -1)
    order = list(order)
    out = {}
    for side, lab in (("left", label_l), ("right", label_r)):
        best = None
        for idx in order[:int(90)]:                      # first ~80 mm from the root
            p = trunk_path[idx]
            if np.linalg.norm(p - trunk_path[root_end]) > root_len:
                break
            i2 = min(idx + 3, len(trunk_path) - 1)
            t = trunk_path[i2] - p
            n = np.linalg.norm(t)
            if n < 1e-6:
                continue
            t = t / n
            # build two vectors across the vessel
            a = np.array([1.0, 0.0, 0.0])
            if abs(t @ a) > 0.9:
                a = np.array([0.0, 1.0, 0.0])
            u = np.cross(t, a)
            u /= np.linalg.norm(u)
            w = np.cross(t, u)
            r_here = trunk_r[idx]
            for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                d = np.cos(ang) * u + np.sin(ang) * w
                if side == "left" and d[1] < 0.2:
                    continue
                if side == "right" and d[1] > -0.2:
                    continue
                for step in np.arange(r_here + 1.0, r_here + 6.0, 1.0):
                    q = p + d * step
                    qv = np.round(np.array([(q[2] - ORG[2]) / SP[0], (q[1] - ORG[1]) / SP[1],
                                            (q[0] - ORG[0]) / SP[2]])).astype(int)
                    if not (0 <= qv[0] < vol.shape[0] and 0 <= qv[1] < vol.shape[1]
                            and 0 <= qv[2] < vol.shape[2]):
                        break
                    if seg[qv[0], qv[1], qv[2]] != 0:
                        continue
                    hu = vol[qv[0], qv[1], qv[2]]
                    if 150 < hu < 500:
                        if best is None or hu > best[0]:
                            best = (float(hu), qv.copy(), d.copy())
                        break
        if best is None:
            log("  coronary %-5s: no contrast tube found on the root" % side)
            continue
        hu0, qv, d = best
        cur = qv.astype(float)
        path = [cur.copy()]
        for _ in range(40):
            best_v, best_p = 0, None
            for dk in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        q = (int(cur[0]) + dk, int(cur[1]) + dy, int(cur[2]) + dx)
                        if not (0 <= q[0] < vol.shape[0] and 0 <= q[1] < vol.shape[1]
                                and 0 <= q[2] < vol.shape[2]):
                            continue
                        if seg[q] != 0:
                            continue
                        v = vol[q]
                        if 140 < v < 520 and v > best_v:
                            best_v, best_p = v, q
            if best_p is None:
                break
            cur = np.array(best_p, float)
            path.append(cur.copy())
        L = float(len(path)) * float(min(SP))
        if len(path) < 5:
            log("  coronary %-5s: seed at %.0f HU but the path died after %.0f mm" % (side, hu0, L))
            continue
        pm = np.array(path)
        pm_mm = np.stack([ORG[0] + pm[:, 2] * SP[2], ORG[1] + pm[:, 1] * SP[1],
                          ORG[2] + pm[:, 0] * SP[0]], axis=1)
        seg[_tube_mask(seg.shape, SP, pm, 1.6)] = lab
        out[side] = dict(label=lab, nodes=len(path), mm=L, seed_hu=hu0)
        log("  coronary %-5s: %.0f mm, seed %.0f HU, %.3f mL" %
            (side, L, hu0, float((seg == lab).sum()) * float(np.prod(SP)) / 1000))
    return out


# --------------------------------------------------------------------------------------------------
# 1b  component shape rejection  (the commercial pipelines do this silently; ours prints every decision)
# --------------------------------------------------------------------------------------------------
def component_features(m, SP):
    """volume, sphericity, elongation and how much of it lies on the volume border"""
    vol_mm3 = float(m.sum()) * float(np.prod(SP))
    if vol_mm3 <= 0:
        return None
    ix = np.argwhere(m).astype(float) * SP
    if len(ix) < 8:
        # too few voxels for a stable covariance; treat it as an elongated dot
        return dict(mL=vol_mm3 / 1000.0, elong=1.0, sphericity=1.0, border=0.0)
    cov = np.cov((ix - ix.mean(axis=0)).T)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        w = np.sort(np.linalg.eigvalsh(cov))[::-1]
    except np.linalg.LinAlgError:
        return None
    w = np.maximum(np.nan_to_num(w, nan=1e-9), 1e-9)
    # elongation: how much longer the longest axis is than the other two
    elong = float(np.sqrt(w[0]) / max(np.sqrt(w[1]), 1e-6))
    # sphericity of the equivalent ellipsoid: 1 for a sphere, small for a tube
    sph = float((w[1] * w[2]) ** 0.25 / np.sqrt(w[0]) * 3.0 ** 0.25)
    border = 0.0
    for ax in range(3):
        for side in (0, -1):
            sl = [slice(None)] * 3
            sl[ax] = side
            border += float(m[tuple(sl)].sum())
    border = border / max(float(np.count_nonzero(m[~np.pad(m, 1)[1:-1, 1:-1, 1:-1]]) if False else 1), 1)
    # border voxels / total
    nb = 0
    for ax in range(3):
        for side in (0, -1):
            sl = [slice(None)] * 3
            sl[ax] = side
            nb += int(m[tuple(sl)].sum())
    border = nb / max(int(m.sum()), 1)
    return dict(mL=vol_mm3 / 1000.0, elong=elong, sphericity=sph, border=border)


def shape_filter(seg, SP, enabled):
    """drop components that cannot be the vessel they claim to be, and say why"""
    ST = ndi.generate_binary_structure(3, 1)
    trunk = seg == 1
    d_trunk = ndi.distance_transform_edt(~trunk, sampling=SP) if trunk.any() else None
    kept, dropped = [], []
    for s_ in SEGMENTS:
        if s_["key"] not in enabled:
            continue
        m = seg == s_["label"]
        if not m.any():
            continue
        lc, nc = ndi.label(m, structure=ST)
        if nc == 1:
            continue
        sizes = np.bincount(lc.ravel())
        for i in range(1, nc + 1):
            comp = lc == i
            f = component_features(comp, SP)
            if f is None:
                continue
            reason = None
            if f["mL"] < 0.05:
                reason = "smaller than 0.05 mL"
            elif s_["role"] != "trunk":
                touch = (d_trunk is not None and
                         float(d_trunk[comp].min()) <= 3.0)
                if not touch:
                    reason = "does not touch the trunk (%.1f mm away)" % (
                        float(d_trunk[comp].min()) if d_trunk is not None else -1)
                elif f["sphericity"] > 0.72 and f["mL"] > 3.0:
                    reason = "round like a chamber (sphericity %.2f, %.1f mL)" % (f["sphericity"], f["mL"])
                elif f["border"] > 0.30 and f["mL"] > 1.0:
                    reason = "mostly on the volume border (%.0f%%)" % (100 * f["border"])
            if reason:
                seg[comp] = 0
                dropped.append((s_["key"], round(f["mL"], 3), reason))
            else:
                kept.append((s_["key"], round(f["mL"], 3),
                             "elong %.1f sph %.2f border %.2f" % (f["elong"], f["sphericity"], f["border"])))
    if dropped:
        log("  shape filter dropped %d component(s):" % len(dropped))
        for k, v, why in dropped:
            log("    - %-8s %.3f mL  %s" % (k, v, why))
    else:
        log("  shape filter: nothing to drop")
    return seg


# --------------------------------------------------------------------------------------------------
# 2-4  centre line, radius, calibre
# --------------------------------------------------------------------------------------------------
def graph_path(pts, k_nn=6):
    n = len(pts)
    if n < 4:
        return pts
    idx = np.zeros((n, 0), int)
    for i in range(0, n, 512):
        block = pts[i:i + 512]
        dd = ((block[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        part = np.argsort(dd, axis=1)[:, 1:k_nn + 1]
        idx = part if idx.size == 0 else np.vstack([idx, part])
    from collections import deque
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in idx[i]:
            adj[i].append(int(j))
            adj[int(j)].append(i)

    def bfs(s0):
        dist = -np.ones(n, int)
        par = -np.ones(n, int)
        dist[s0] = 0
        q = deque([s0])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
        return int(np.argmax(dist)), par

    a, _ = bfs(0)
    b, par = bfs(a)
    p, cur = [], b
    while cur >= 0:
        p.append(cur)
        cur = par[cur]
    p.reverse()
    return pts[np.array(p)]


def frame_seq(p):
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    n = np.linalg.norm(t, axis=1)
    n[n == 0] = 1.0
    return t / n[:, None]


def ray_radius(mask, p, t, ORG, SP, max_r=25.0, nray=24, step=0.25, exclude=None):
    """median exit distance of rays cast outward in the plane normal to the centre line"""
    a = np.array([0.0, 0.0, 1.0])
    if abs(t @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(t, a)
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        return np.nan
    u /= nu
    w = np.cross(t, u)
    hits = []
    for k in range(nray):
        ang = 2 * np.pi * k / nray
        d = np.cos(ang) * u + np.sin(ang) * w
        last = 0.0
        r = step
        spoiled = False
        while r <= max_r:
            q = p + d * r
            i0 = int(round((q[2] - ORG[2]) / SP[0]))
            i1 = int(round((q[1] - ORG[1]) / SP[1]))
            i2 = int(round((q[0] - ORG[0]) / SP[2]))
            if not (0 <= i0 < mask.shape[0] and 0 <= i1 < mask.shape[1] and 0 <= i2 < mask.shape[2]):
                break
            if exclude is not None and exclude[i0, i1, i2]:
                spoiled = True          # this ray ran into the trunk; it says nothing about the branch
                break
            if mask[i0, i1, i2]:
                last = r
            elif last > 0:
                break
            r += step
        if last > 0 and not spoiled:
            hits.append(last)
    return float(np.median(hits)) if hits else np.nan


def band_radius(cloud, p, t, vox_mm3, band=BAND):
    d = cloud - p
    h = d @ t
    sel = np.abs(h) <= band
    return float(np.sqrt(max(sel.sum() * vox_mm3 / (2 * band), 1e-6) / np.pi))


def centre_lines(seg, SP, ORG, VT):
    data = {}
    e_trunk = ndi.distance_transform_edt(seg == 1, sampling=SP) if (seg == 1).any() else None
    for s in SEGMENTS:
        m = seg == s["label"]
        if m.sum() < 20:
            continue
        sk = skeletonize(m)
        pts = np.argwhere(sk).astype(float) if sk.sum() >= 6 else np.argwhere(m).astype(float)
        raw = graph_path(pts).astype(int)
        vx = np.stack([ORG[0] + raw[:, 2] * SP[2], ORG[1] + raw[:, 1] * SP[1],
                       ORG[2] + raw[:, 0] * SP[0]], axis=1)
        vx = np.stack([ndi.gaussian_filter1d(vx[:, i], 9.0, mode="nearest") for i in range(3)], axis=1)
        vi = np.argwhere(m).astype(float)
        cloud = np.stack([ORG[0] + vi[:, 2] * SP[2], ORG[1] + vi[:, 1] * SP[1],
                          ORG[2] + vi[:, 0] * SP[0]], axis=1)
        tg = frame_seq(vx)
        # the trunk's calibre comes from the rays; the branches keep the slab measurement, which is what the
        # accepted model used.  Ray casting near an ostium reads the aorta's half-width and makes the branch
        # tube wider than the opening it leaves.
        if s["role"] == "trunk":
            r = np.array([G.ray_radius(m, vx[i], tg[i], ORG, SP) for i in range(len(vx))])
        else:
            r = np.array([band_radius(cloud, vx[i], tg[i], VT * 1000.0) for i in range(len(vx))])
        bad = ~np.isfinite(r) | (r <= 0.2)
        if bad.any() and (~bad).any():
            r[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), r[~bad])
        length_mm = float(np.linalg.norm(np.diff(vx, axis=0), axis=1).sum())
        mask_mL = float(m.sum()) * float(np.prod(SP)) / 1000.0
        # a short mask is not a stub. The celiac trunk is genuinely 10-15 mm and its medial axis is shorter
        # still, so reject on the mask volume - that is what says whether a vessel is there at all.
        # the threshold is per segment: a coronary branch is 0.1-0.3 mL and would be rejected by the
        # threshold that suits the abdominal branches
        if mask_mL < float(s.get("min_mL", 0.30)) or len(vx) < 5:
            log("  %-14s skipped: %.2f mL over %.1f mm is not a vessel" % (s["key"], mask_mL, length_mm))
            continue
        k = np.arange(len(r))
        try:
            spl = UnivariateSpline(k, r, k=3, s=float(len(r)) * 1.44)
            fit = spl(k)
            if fit.min() > 0.5:
                r = fit
        except Exception:
            pass
        ds = np.linalg.norm(np.diff(vx, axis=0), axis=1)
        v_an = float(np.pi * np.sum(r[1:] ** 2 * ds)) / 1000.0
        v_mask = float(m.sum()) * VT
        # reported, not applied: a mismatch here means the mask and the sweep disagree, and that is a finding
        kfac = (v_an / v_mask) if v_mask > 0 else 1.0
        data[s["key"]] = dict(path=vx, r=r, v=float(m.sum()) * VT, label=s["label"],
                              role=s["role"], kfac=kfac, v_tube=v_an)
        log("  %-14s %4d pts  %.0f mm  r med %.2f  mask %.2f mL" %
            (s["key"], len(vx), float(ds.sum()), float(np.median(r)), float(m.sum()) * VT))
    return data


# --------------------------------------------------------------------------------------------------
# 5-7  junctions, union, surface
# --------------------------------------------------------------------------------------------------
def prepare_junctions(data):
    return G.prepare_junctions(data, log=log)


def union_and_mesh(data, SP):
    return G.union_and_mesh(data, log=log)


# --------------------------------------------------------------------------------------------------
# 8  deliverables
# --------------------------------------------------------------------------------------------------
def write_viewer(mesh, out_path, title):
    target = 90000
    M = mesh
    if len(M.faces) > target:
        try:
            M = M.simplify_quadric_decimation(face_count=target)
        except Exception:
            pass
    glb = M.export(file_type="glb")
    b64 = base64.b64encode(glb).decode("ascii")
    html = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>__T__</title>
<style>html,body{margin:0;height:100%;background:#0d1117;overflow:hidden;font-family:-apple-system,"Segoe UI",sans-serif}
#i{position:fixed;left:14px;top:12px;color:#c9d1d9;font-size:13px}#h{position:fixed;left:14px;bottom:12px;color:#8b949e;font-size:12px}</style></head>
<body><div id="i">__T__</div><div id="h">左键拖动旋转 · 滚轮缩放 · 右键平移</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const sc=new THREE.Scene(); sc.background=new THREE.Color(0x0d1117);
const cam=new THREE.PerspectiveCamera(35,innerWidth/innerHeight,1,5000);
const rn=new THREE.WebGLRenderer({antialias:true}); rn.setPixelRatio(Math.min(devicePixelRatio,2));
rn.setSize(innerWidth,innerHeight); document.body.appendChild(rn.domElement);
const k=new THREE.DirectionalLight(0xffffff,2.6); k.position.set(0.6,0.9,0.8); sc.add(k);
const f=new THREE.DirectionalLight(0xbfd4ff,1.1); f.position.set(-0.9,0.2,-0.4); sc.add(f);
const r2=new THREE.DirectionalLight(0xffd9c9,1.0); r2.position.set(0.1,-0.9,0.5); sc.add(r2);
sc.add(new THREE.AmbientLight(0xffffff,0.35));
const mat=new THREE.MeshPhysicalMaterial({color:0xc0392b,roughness:0.42,metalness:0.05,clearcoat:0.35,clearcoatRoughness:0.5});
const b=atob("__B64__"); const u=new Uint8Array(b.length); for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);
new GLTFLoader().parse(u.buffer,'',(g)=>{const root=g.scene;const box=new THREE.Box3().setFromObject(root);
 const c=box.getCenter(new THREE.Vector3()); const s=box.getSize(new THREE.Vector3()).length();
 root.position.sub(c); root.traverse(o=>{if(o.isMesh)o.material=mat}); sc.add(root);
 cam.position.set(s*0.85,-s*1.05,s*0.35); cam.up.set(0,0,1); cam.lookAt(0,0,0);
 const ctl=new OrbitControls(cam,rn.domElement); ctl.target.set(0,0,0); ctl.enableDamping=true;
 ctl.dampingFactor=0.08; ctl.minDistance=s*0.12; ctl.maxDistance=s*3.5; ctl.update();
 const an=()=>{requestAnimationFrame(an);ctl.update();rn.render(sc,cam)}; an();});
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight)});
</script></body></html>"""
    html = html.replace("__B64__", b64).replace("__T__", title)
    open(out_path, "w", encoding="utf-8").write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--extra", action="append", default=[], metavar="name=file.nii.gz")
    ap.add_argument("--no-ts", action="store_true")
    ap.add_argument("--enable", action="append", default=[], metavar="key",
                    help="enable a reserved segment (e.g. coronary_l) once its data exists")
    ap.add_argument("--coronary", choices=("off", "auto"), default="off",
                    help="auto = look for the coronary ostia on the aortic root and trace them")
    a = ap.parse_args()
    case = os.path.abspath(a.case)
    out = os.path.join(case, "out_pipeline")
    os.makedirs(out, exist_ok=True)
    extras = {}
    for e in a.extra:
        k, _, v = e.partition("=")
        extras[k] = v

    log("case %s" % case)
    P = json.load(open(os.path.join(case, "crop_meta.json"), encoding="utf-8"))
    SP = np.array(P["sp_zyx"], float)
    ORG = np.array(P["org_c"], float)
    crop = P["crop"]
    VT = float(np.prod(SP)) / 1000.0
    import case_io as B
    vol, _ = B.load_case(case)
    log("  volume %s spacing %s" % (vol.shape, np.round(SP, 3)))

    ct = os.path.join(case, "ct.nii.gz")
    if a.no_ts:
        ts_dir, ok = os.path.join(case, "ts_full"), True
    else:
        ts_dir, ok = run_totalsegmentator(case, ct)
    if not ok:
        log("segmentation failed; stopping")
        return 2

    enabled = {s_["key"] for s_ in SEGMENTS if s_.get("enabled", True)} | set(a.enable)
    if a.coronary == "auto":
        enabled |= {"coronary_l", "coronary_r"}      # asking for coronaries enables those two segments
    if not a.no_ts:
        run_visceral_ts(case, ct)
        if a.coronary == "auto":
            run_coronary_ts(case, ct)
    seg = build_segmentation(case, ts_dir, crop, vol, SP, extras, enabled)
    if a.coronary == "auto" and not (seg == 20).any() and not (seg == 21).any():
        # needs the trunk's centre line, so do a quick one here; the full pass comes later
        _m = seg == 1
        if _m.sum() > 500:
            _sk = skeletonize(_m)
            _pts = np.argwhere(_sk).astype(float)
            if len(_pts) > 20:
                _raw = graph_path(_pts).astype(int)
                _p = np.stack([ORG[0] + _raw[:, 2] * SP[2], ORG[1] + _raw[:, 1] * SP[1],
                               ORG[2] + _raw[:, 0] * SP[0]], axis=1)
                _p = np.stack([ndi.gaussian_filter1d(_p[:, i], 9.0, mode="nearest") for i in range(3)],
                              axis=1)
                _e = ndi.distance_transform_edt(_m, sampling=SP)
                _idx = np.clip(np.stack([(_p[:, 2] - ORG[2]) / SP[0], (_p[:, 1] - ORG[1]) / SP[1],
                                         (_p[:, 0] - ORG[0]) / SP[2]], axis=1), 0,
                               np.array(seg.shape) - 1).astype(int)
                _r = _e[tuple(_idx.T)]
                trace_coronary(vol, seg, SP, ORG, _p, _r)
    seg = shape_filter(seg, SP, enabled | set(extras))
    np.save(os.path.join(out, "seg.npy"), seg)
    log("  labels present: %s" % sorted(set(np.unique(seg).tolist()) - {0}))

    data = centre_lines(seg, SP, ORG, VT)
    data = prepare_junctions(data)
    M, mask = union_and_mesh(data, SP)

    stl = os.path.join(out, "model.stl")
    M.export(stl)
    step = os.path.join(out, "model.step")
    try:
        subprocess.run([sys.executable, os.path.join(HERE, "stl2step_writer.py"), stl, step,
                        "--name", "aorta"], check=False, capture_output=True)
    except Exception as e:
        log("  STEP writer failed: %r" % e)

    write_viewer(M, os.path.join(out, "viewer.html"), "主动脉模型 · 鼠标拖动环绕")

    acc = dict(case=os.path.basename(case), mesh_mL=round(abs(M.volume) / 1000, 2),
               faces=int(len(M.faces)), watertight=bool(M.is_watertight),
               components=len(M.split(only_watertight=False)),
               segments={k: dict(nodes=len(v["path"]), r_med=round(float(np.median(v["r"])), 2),
                                 mask_mL=round(v["v"], 2), kfac=round(v["kfac"], 3))
                         for k, v in data.items()},
               tube_vs_mask={k: round(v.get("kfac", 1.0), 3) for k, v in data.items()},
               # the profile the geometry was actually built from, so QC can read it instead of
               # re-deriving it from labels (that re-derivation produced impossible numbers)
               profiles={k: dict(r=[round(float(x), 2) for x in v["r"][::max(len(v["r"]) // 40, 1)]],
                                 end=[round(float(x), 1) for x in v["path"][-1]])
                         for k, v in data.items()},
               missing=[s["key"] for s in SEGMENTS if s["key"] not in data],
               reserved=[s["key"] for s in SEGMENTS if not s.get("enabled", True)])
    n_comp = len(M.split(only_watertight=False))
    acc["ok"] = bool(M.is_watertight and n_comp == 1)
    if not acc["ok"]:
        log("  NOT ACCEPTED: watertight %s, %d components" % (M.is_watertight, n_comp))
    json.dump(acc, open(os.path.join(out, "acceptance.json"), "w", encoding="utf-8"),
              indent=1, default=float)
    log("final: %.1f mL  %d faces  watertight %s  components %d" %
        (abs(M.volume) / 1000, len(M.faces), M.is_watertight, len(M.split(only_watertight=False))))
    log("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
