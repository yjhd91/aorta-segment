"""case_io.py - read one case directory (ct.nii.gz + crop_meta.json)."""
import json
import os

import nibabel as nib
import numpy as np


def load_case(d):
    """Return (volume (z, y, x), spacing (z, y, x)) for one case directory."""
    img = nib.load(os.path.join(d, "ct.nii.gz"))
    a = np.asarray(img.dataobj, dtype=np.float32)       # original grid, (y, x, z)
    with open(os.path.join(d, "crop_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    sp = tuple(float(x) for x in meta["sp_zyx"])
    (z0, z1), (y0, y1), (x0, x1) = meta["crop"]
    a = a[y0:y1, x0:x1, z0:z1]
    vol = np.ascontiguousarray(np.transpose(a, (2, 0, 1)))   # (z, y, x) like the masks
    return vol, sp
