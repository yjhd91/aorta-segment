"""aorta_geom.py - the accepted geometry stage, in one place.

This module exists because the pipeline and the accepted model had drifted apart: the junction construction
was fixed in one file and the pipeline kept the broken one, so a batch rebuilt the wrong geometry.  Both the
single-case script and the pipeline now call exactly this code.

What it implements, and why each choice was measured rather than assumed:

  radius     normal-plane ray casting (24 rays, median exit distance). A tangential slab inflates a bent
             vessel; on c000 the slab gave 10.25 mm and the finished model 11.00 mm where the image shows
             9.60 mm, and the ray measurement gives 9.79 mm.
  junctions  a branch is extended at constant radius past the trunk's centre line; the trunk continues into
             the straighter iliac along that iliac's own path, and the other iliac leaves from the side.
             Three tubes meeting at one point always leave a collar; a T-junction does not.
  union      every tube is rasterised into one 0.6 mm grid and OR-ed. A boolean union of capped tubes leaves
             a collar at every joint.
  surface    the signed distance field is blurred by 1.8 mm and the zero level set extracted; that rounds
             the concave creases of the joints.
"""
import numpy as np
import scipy.ndimage as ndi
import trimesh
from skimage import measure

GRID = 0.6
SIG_FIELD = 1.8
JUNCTION_FILL = 3.5
BRIDGE_R = 1.2
TAUBIN = 40


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
                spoiled = True
                break
            if mask[i0, i1, i2]:
                last = r
            elif last > 0:
                break
            r += step
        if last > 0 and not spoiled:
            hits.append(last)
    return float(np.median(hits)) if hits else np.nan


def prepare_junctions(data, log=print):
    """trunk continues into the straighter iliac; every other branch enters the trunk from the side"""
    trunk = next((k for k, v in data.items() if v["role"] == "trunk"), None)
    if trunk is None:
        return data
    ap = data[trunk]["path"]
    r_tr = data[trunk]["r"]

    # 1. trunk continues along the straighter iliac's own path
    t_end = ap[-1] - ap[-6] if len(ap) > 6 else ap[-1] - ap[0]
    t_end = t_end / max(np.linalg.norm(t_end), 1e-9)
    best, best_cos = None, -2.0
    for k, v in data.items():
        if v["role"] != "bifurcation":
            continue
        pp = v["path"]
        n0 = min(len(pp) - 1, 8)
        d = pp[n0] - pp[0]
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        c = float((d / nd) @ t_end)
        if c > best_cos:
            best, best_cos = k, c
    if best is not None:
        r_cont = float(np.median(data[best]["r"]))
        bp = data[best]["path"]
        br = data[best]["r"]
        dseg = np.linalg.norm(np.diff(bp, axis=0), axis=1)
        dsar = np.concatenate([[0.0], np.cumsum(dseg)])
        n_take = max(2, min(int(np.searchsorted(dsar, 60.0)) + 1, len(bp) - 2))
        data[trunk]["path"] = np.concatenate([ap, bp[1:n_take + 1]], axis=0)
        ap2 = data[trunk]["path"]
        # Three attempts at shaping this taper each produced a different artefact (a step, a flattened
        # aorta, then a bulb into a cliff). So stop shaping it and constrain it instead: no radius may rise
        # faster than 0.22 mm per mm of arc as you move up from the joint. That admits no bulb by
        # construction, and the descent into the joint is then whatever the trunk's own profile allows.
        base = np.concatenate([r_tr, br[1:n_take + 1]])
        sarc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ap2, axis=0), axis=1))])
        n = min(len(base), len(sarc))
        base, sarc = np.array(base[:n], float), sarc[:n]
        k_join = int(np.argmin(np.abs(sarc - (sarc[-1] - float(dsar[n_take])))))
        slope = 0.22
        r_new = base.copy()
        r_new[k_join] = min(r_new[k_join], r_cont)
        for i in range(k_join - 1, -1, -1):
            ds = max(float(sarc[i + 1] - sarc[i]), 1e-3)
            r_new[i] = min(r_new[i], r_new[i + 1] + slope * ds)
        data[trunk]["r"] = r_new
        log("  trunk radius limited to %.2f mm/mm rising toward the joint, %.2f mm there"
            % (slope, r_new[k_join]))
        # THE JUNCTION POINT carries one radius. Every tube that leaves it starts at that radius and ramps to
        # its own over a short run. Without this the branches meet at three different calibres, and the union
        # shows a lump on one side and a notch on the other - the same defect seen from two sides.
        r_join = float(data[trunk]["r"][len(ap) - 1])
        P_join = ap[-1]
        for k, v in data.items():
            if v["role"] != "bifurcation" or k == best:
                continue
            pp, rr = v["path"], v["r"]
            d0 = float(np.linalg.norm(pp[0] - P_join))
            d1 = float(np.linalg.norm(pp[-1] - P_join))
            k0 = 0 if d0 <= d1 else -1
            start = pp[k0]
            gap = float(np.linalg.norm(start - P_join))
            # The branch must leave FROM the fork, not from wherever its own centre line happens to start.
            # That offset is what produced the rounded shoulder seen at the c013 fork: the tube's cylindrical
            # start sat beside the trunk's axis and the union built a lump around it. Earlier code only
            # re-assigned this branch's radius, never its path, so nothing changed.
            if gap > 0.5:
                dirv = (start - P_join) / gap
                steps = np.arange(0.6, gap, 0.6)
                lead = np.stack([P_join + dirv * t for t in steps], axis=0)
                lead_r = np.linspace(r_join, float(rr[k0]), len(steps) + 1)[1:]
                if k0 == 0:
                    v["path"] = np.concatenate([lead, pp], axis=0)
                    v["r"] = np.concatenate([lead_r, np.array(rr, float)])
                else:
                    v["path"] = np.concatenate([pp, lead[::-1]], axis=0)
                    v["r"] = np.concatenate([np.array(rr, float), lead_r[::-1]])
            else:
                v["r"] = np.array(rr, float)
            log("  %-14s leaves the fork (gap %.1f mm), %.2f -> %.2f mm" %
                (k, gap, r_join, float(rr[k0])))
        log("  trunk continues into %s along its own path (%.0f mm), calibre %.2f mm"
            % (best, float(dsar[n_take]), r_cont))

    # 2. every other branch enters the trunk at constant radius, past its centre line
    ap = data[trunk]["path"]
    for key, d in data.items():
        if key == trunk or d["role"] == "bifurcation" and best is not None and key == best:
            continue
        p, r = d["path"], d["r"]
        d0 = float(np.min(np.linalg.norm(ap - p[0], axis=1)))
        d1 = float(np.min(np.linalg.norm(ap - p[-1], axis=1)))
        k_end = -1 if d1 < d0 else 0
        pe = p[k_end]
        j = int(np.argmin(np.linalg.norm(ap - pe, axis=1)))
        dirv = ap[j] - pe
        L = float(np.linalg.norm(dirv))
        if L < 1e-6:
            continue
        dirv = dirv / L
        # the branch must stop short of the trunk's far wall. A fixed overshoot of 7 mm works when the trunk
        # is wide and pokes out through the opposite wall when it is not - that is the knob seen on the left
        # of the c013 bifurcation, where the trunk is only ~5 mm in radius at the join.
        r_trunk_here = float(data[trunk]["r"][j]) if j < len(data[trunk]["r"]) else 5.0
        overshoot = min(7.0, max(2.5, 0.55 * r_trunk_here))
        steps = np.arange(0.8, L + overshoot, 0.8)
        ext = np.stack([pe + dirv * t for t in steps], axis=0)
        ext_r = np.full(len(steps), r[k_end] * 0.98)
        if k_end == 0:
            d["path"] = np.concatenate([ext[::-1], p], axis=0)
            d["r"] = np.concatenate([ext_r[::-1], r])
        else:
            d["path"] = np.concatenate([p, ext], axis=0)
            d["r"] = np.concatenate([r, ext_r])
    return data


def union_and_mesh(data, log=print):
    allp = np.vstack([d["path"] for d in data.values()])
    maxr = max(float(d["r"].max()) for d in data.values())
    lo = np.floor((allp.min(axis=0) - maxr - 10) / GRID).astype(int)
    hi = np.ceil((allp.max(axis=0) + maxr + 10) / GRID).astype(int)
    shape = tuple(int(x) for x in (hi - lo))
    log("  union grid %s = %.1f M voxels" % (shape, np.prod(shape) / 1e6))
    mask = np.zeros(shape, bool)
    for d in data.values():
        p, r = d["path"], d["r"]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        L = float(s[-1])
        sn = np.linspace(0.0, L, max(int(L / 0.7) + 2, 4))
        pp = np.stack([np.interp(sn, s, p[:, i]) for i in range(3)], axis=1)
        rr = np.interp(sn, s, r)
        for q, rad in zip(pp, rr):
            c = (q / GRID - lo).astype(int)
            k = int(np.ceil(rad / GRID)) + 1
            a0 = np.maximum(c - k, 0)
            a1 = np.minimum(c + k + 1, np.array(shape))
            if (a1 <= a0).any():
                continue
            g = np.ogrid[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
            d2 = ((g[0] + lo[0] - q[0] / GRID) * GRID) ** 2 + \
                 ((g[1] + lo[1] - q[1] / GRID) * GRID) ** 2 + \
                 ((g[2] + lo[2] - q[2] / GRID) * GRID) ** 2
            mask[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]] |= (d2 <= rad * rad)
    lc, nc = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    if nc > 1:
        sz = np.bincount(lc.ravel())
        big = int(np.argmax(sz[1:])) + 1
        for i in range(1, nc + 1):
            if i != big and sz[i] * GRID ** 3 / 1000.0 < 0.05:
                mask[lc == i] = False
        lc, nc = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    # A detached piece that survives the speck filter is a real vessel fragment that the field blur cut off
    # at a thin point - on p06 it was a 1.4 mm thread 40 mm long. Deleting it would hide the defect, so it is
    # welded back with a thin bridge, and the bridge is reported.
    if nc > 1:
        sz = np.bincount(lc.ravel())
        big = int(np.argmax(sz[1:])) + 1
        main_pts = np.argwhere(lc == big)
        for i in range(1, nc + 1):
            if i == big:
                continue
            pts = np.argwhere(lc == i)
            if len(pts) == 0:
                continue
            # nearest pair between this piece and the main body, cheap search on sub-samples
            step = max(len(pts) // 400, 1)
            sub = pts[::step].astype(float)
            mstep = max(len(main_pts) // 4000, 1)
            msub = main_pts[::mstep].astype(float)
            d2 = ((sub[:, None, :] - msub[None, :, :]) ** 2).sum(axis=2)
            ai, bi = np.unravel_index(np.argmin(d2), d2.shape)
            a_pt, b_pt = sub[ai], msub[bi]
            L = float(np.linalg.norm(a_pt - b_pt)) * GRID
            n_step = max(int(L / 0.5) + 2, 3)
            for t in np.linspace(0.0, 1.0, n_step):
                q = a_pt + (b_pt - a_pt) * t
                c = np.round(q).astype(int)
                k = int(np.ceil(BRIDGE_R / GRID)) + 1
                a0 = np.maximum(c - k, 0)
                a1 = np.minimum(c + k + 1, np.array(mask.shape))
                g = np.ogrid[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
                dd = ((g[0] - q[0]) * GRID) ** 2 + ((g[1] - q[1]) * GRID) ** 2 + \
                     ((g[2] - q[2]) * GRID) ** 2
                mask[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]] |= (dd <= BRIDGE_R ** 2)
            log("  welded a %.3f mL fragment back with a %.1f mm bridge over %.1f mm"
                % (len(pts) * GRID ** 3 / 1000, BRIDGE_R, L))
        lc, nc = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    log("  union: %.1f mL, %d component(s)" % (float(mask.sum()) * GRID ** 3 / 1000, nc))
    # A junction between two cylinders at a shallow angle has a convex lump on one side and a concave crease
    # on the other, and they are two faces of the same defect. A morphological closing with a small ball
    # fillets every concave crease by that radius and leaves the convex surfaces where they are, so the
    # crotch stops being a notch and the lump stops standing proud of it. It is applied before the field
    # blur, which alone was too weak to fill a shallow-angle crotch.
    ball_r = int(np.ceil(JUNCTION_FILL / GRID))
    g = np.ogrid[-ball_r:ball_r + 1, -ball_r:ball_r + 1, -ball_r:ball_r + 1]
    ball = (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= ball_r ** 2
    before = float(mask.sum()) * GRID ** 3 / 1000
    mask = ndi.binary_closing(mask, structure=ball)
    after = float(mask.sum()) * GRID ** 3 / 1000
    log("  junction fillet %.1f mm: %.1f -> %.1f mL" % (JUNCTION_FILL, before, after))
    # The mesh can come apart even when the voxel union is one piece: a neck only a voxel or two across is
    # pinched by the 1.8 mm field blur, and marching cubes then emits two shells. That is what failed p06
    # (0.059 mL, a 1.4 mm thread 40 mm long). If it happens, thicken the neck in voxel space and re-march,
    # up to three steps, recording each.
    mask_work = mask
    for attempt in range(4):
        pad = 6
        mpad = np.pad(mask_work, [(pad, pad)] * 3, mode="constant")
        lo_p = lo - pad
        din = ndi.distance_transform_edt(mpad, sampling=(GRID, GRID, GRID))
        dout = ndi.distance_transform_edt(~mpad, sampling=(GRID, GRID, GRID))
        sdf = ndi.gaussian_filter((dout - din).astype(np.float32), sigma=[SIG_FIELD / GRID] * 3)
        v, f, _, _ = measure.marching_cubes(sdf, level=0.0, spacing=(GRID, GRID, GRID),
                                            allow_degenerate=False)
        V = np.stack([v[:, 0] + lo_p[0] * GRID, v[:, 1] + lo_p[1] * GRID,
                      v[:, 2] + lo_p[2] * GRID], axis=1)
        M = trimesh.Trimesh(vertices=V, faces=f, process=True)
        cs = sorted(M.split(only_watertight=False), key=lambda c: -abs(c.volume))
        M = trimesh.util.concatenate([c for c in cs if abs(c.volume) / 1000.0 >= 0.01])
        n_now = len(M.split(only_watertight=False))
        if n_now <= 1:
            break
        if attempt == 3:
            log("  WARNING: still %d pieces after three neck thickenings" % n_now)
            break
        k_r = attempt + 1                      # 1, 2, 3 voxels
        gg = np.ogrid[-k_r:k_r + 1, -k_r:k_r + 1, -k_r:k_r + 1]
        ball2 = (gg[0] ** 2 + gg[1] ** 2 + gg[2] ** 2) <= k_r ** 2
        before = float(mask_work.sum()) * GRID ** 3 / 1000
        mask_work = ndi.binary_closing(mask_work, structure=ball2)
        log("  mesh came out in %d pieces; thickening necks by %d voxel(s): %.1f -> %.1f mL"
            % (n_now, k_r, before, float(mask_work.sum()) * GRID ** 3 / 1000))
    mask = mask_work
    try:
        from trimesh import smoothing
        raw = M.copy()
        n_before = len(raw.split(only_watertight=False))
        smoothing.filter_taubin(M, lamb=0.5, nu=-0.53, iterations=TAUBIN)
        if len(M.split(only_watertight=False)) > n_before:
            log("  taubin split the mesh, reverting")
            M = raw
    except Exception as e:
        log("  taubin skipped: %r" % e)
    return M, mask
