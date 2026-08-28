"""EKSEN 5 duzgun: gecmis-uzunlugu KOVASINA gore optimal shrinkage agirligi."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()
KOVA = [(1, 4), (4, 8), (8, 15), (15, 30), (30, 60), (60, 120), (120, 250), (250, 10**9)]
W = np.arange(0, 1.01, 0.1)
res = {}
for kesim in KESIMLER:
    gec, hed = hazirla(tr, kesim)
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    kok = float(gec.ly.mean())
    base = 0.8 * geri_dolgu(hed, oz.k7, oz.ly_all, kok=kok) + 0.2 * geri_dolgu(
        hed, oz.ly_all, kok=kok
    )
    gi = gec.groupby(["guc", "ilce"]).ly.mean()
    gucm = gec.groupby("guc").ly.mean()
    prior = (
        pd.Series(hed.set_index(["guc", "ilce"]).index.map(gi), index=hed.index)
        .fillna(hed.guc.map(gucm))
        .fillna(kok)
        .values
    )
    E = hed.tanim.map(gr).values == "E_normal"
    y = hed.ly.values
    nsat = hed.tanim.map(oz.nsat).values.astype(float)
    print(f"\n===== {kesim} =====")
    print(
        f"{'kova':14s} {'satir':>8s} {'w=1(kendi)':>10s} {'w=0(prior)':>10s} {'en iyi w':>9s} {'en iyi':>8s} {'kazanc':>8s}"
    )
    res[kesim] = {}
    for lo, hi in KOVA:
        m = E & (nsat >= lo) & (nsat < hi)
        if m.sum() < 200:
            print(f"[{lo},{hi}) satir {m.sum()} - atlandi")
            continue
        e = [float(np.sqrt(((y[m] - (w * base[m] + (1 - w) * prior[m])) ** 2).mean())) for w in W]
        b = int(np.argmin(e))
        res[kesim][f"{lo}-{hi}"] = {
            "n": int(m.sum()),
            "egri": dict(zip([f"{w:.1f}" for w in W], e)),
            "en_iyi_w": float(W[b]),
        }
        print(
            f"[{lo:>4},{hi if hi < 10**8 else 'inf':>4}) {m.sum():8,d} {e[-1]:10.4f} {e[0]:10.4f} {W[b]:9.1f} {e[b]:8.4f} {e[-1] - e[b]:8.4f}"
        )
json_yaz("eksen5_shrink_kova", res)
print("\n=== 4 kesim toplu (satir agirlikli) en iyi w ===")
for lo, hi in KOVA:
    kk = f"{lo}-{hi}"
    ns = [res[k][kk]["n"] for k in KESIMLER if kk in res[k]]
    if not ns:
        continue
    egri = np.array([[res[k][kk]["egri"][f"{w:.1f}"] for w in W] for k in KESIMLER if kk in res[k]])
    ns = np.array(ns, float)
    mse = ((egri**2) * ns[:, None]).sum(0) / ns.sum()
    b = int(np.argmin(mse))
    print(
        f"[{lo},{hi if hi < 10**8 else 'inf'}) n={int(ns.sum()):,} en iyi w={W[b]:.1f} rmsle {np.sqrt(mse[b]):.4f} vs w=1 {np.sqrt(mse[-1]):.4f} (kazanc {np.sqrt(mse[-1]) - np.sqrt(mse[b]):.4f})"
    )
