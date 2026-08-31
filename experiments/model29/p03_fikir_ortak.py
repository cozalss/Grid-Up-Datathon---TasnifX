"""p03 ortak yardimcilar."""

import numpy as np


def komsu_ozellik(gec, hed_meta, k=8):
    """Her hedef trafo icin: ayni ilcede idnum'a en yakin k komsunun ortalama log seviyesi.

    gec: gecmis satirlari (tanim, ilce, idnum, ly)
    hed_meta: hedef trafolarin benzersiz meta tablosu (tanim, ilce, idnum)
    """
    lv = gec.groupby("tanim").agg(ly=("ly", "mean")).reset_index()
    meta = gec.drop_duplicates("tanim")[["tanim", "ilce", "idnum"]]
    lv = lv.merge(meta, on="tanim").dropna(subset=["idnum"])
    out = {}
    hm = hed_meta.dropna(subset=["idnum"])
    for ilce, gh in hm.groupby("ilce", observed=True):
        gl = lv[lv.ilce == ilce]
        if len(gl) < 2:
            continue
        arr = gl.idnum.to_numpy()
        order = np.argsort(arr)
        xs = arr[order]
        vs = gl.ly.to_numpy()[order]
        ids = gl.tanim.to_numpy()[order]
        for tn, xv in zip(gh.tanim.to_numpy(), gh.idnum.to_numpy()):
            j = np.searchsorted(xs, xv)
            lo, hi = max(0, j - k), min(len(xs), j + k)
            sel = np.arange(lo, hi)
            sel = sel[ids[sel] != tn]
            if len(sel):
                out[tn] = float(vs[sel].mean())
    return out
