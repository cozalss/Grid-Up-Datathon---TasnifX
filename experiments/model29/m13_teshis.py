"""SICAK hata kutlesinin nereden geldigi: sifir gecisleri, gecmis uzunlugu, seviye."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *

tr = yukle()
out = {}
for kesim in ["2025-10-31", "2025-11-30"]:
    k = pd.Timestamp(kesim)
    gec, hed = hazirla(tr, kesim)
    tam = pencere_seviye(gec, kesim, None, "mean")
    kok = float(gec.ly.mean())
    s7 = pencere_seviye(gec, kesim, 7, "mean")
    p = geri_dolgu(hed, s7, tam, kok=kok)
    e2 = (p - hed.ly.values) ** 2
    tot = e2.sum()
    print(f"\n===== {kesim} SICAK RMSLE {np.sqrt(e2.mean()):.4f} n={len(hed):,} =====")
    print(
        f"hedefte tuketim==0 orani %{100 * (hed.tuketim == 0).mean():.2f}, <1 %{100 * (hed.tuketim < 1).mean():.2f}"
    )
    m0 = hed.tuketim.values < 1
    print(
        f"  hedef<1 satirlarin hata kutlesi payi: %{100 * e2[m0].sum() / tot:.1f} (RMSLE {np.sqrt(e2[m0].mean()):.3f})"
    )
    # gecmisin son 7/28 gununde sifir orani
    for W in [7, 28]:
        g = gec[gec.tarih > k - pd.Timedelta(days=W)]
        sf = g.assign(z=(g.tuketim < 1)).groupby("tanim").z.mean()
        hed["_sf"] = hed.tanim.map(sf)
        for lo, hi, ad in [
            (-0.01, 0.001, "son%d hic sifir yok" % W),
            (0.001, 0.999, "son%d kismi sifir" % W),
            (0.999, 1.01, "son%d TAMAMEN sifir" % W),
        ]:
            m = (hed._sf.values > lo) & (hed._sf.values <= hi)
            if m.sum() == 0:
                continue
            hz = (hed.tuketim.values[m] < 1).mean()
            print(
                f"  {ad:24s} n={m.sum():7,d} kutle %{100 * e2[m].sum() / tot:5.1f} RMSLE {np.sqrt(e2[m].mean()):.3f} | hedefte sifir orani %{100 * hz:.1f}"
            )
    # gecmis uzunlugu
    nlen = gec.groupby("tanim").size()
    hed["_n"] = hed.tanim.map(nlen)
    print("  --- gecmis uzunlugu (satir) ---")
    for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 120), (120, 250), (250, 10**9)]:
        m = (hed._n.values >= lo) & (hed._n.values < hi)
        if m.sum() == 0:
            continue
        print(
            f"   n_gecmis [{lo},{hi}): satir={m.sum():7,d} kutle %{100 * e2[m].sum() / tot:5.1f} RMSLE {np.sqrt(e2[m].mean()):.3f}"
        )
    # seviyeye gore
    print("  --- kestirilen seviyeye gore ---")
    for lo, hi in [(-1, 0.5), (0.5, 3), (3, 5), (5, 7), (7, 9), (9, 99)]:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        print(
            f"   pred ly [{lo},{hi}): satir={m.sum():7,d} kutle %{100 * e2[m].sum() / tot:5.1f} RMSLE {np.sqrt(e2[m].mean()):.3f}"
        )
    # en kotu 1% satirin katkisi
    q = np.sort(e2)[::-1]
    for f in [0.001, 0.005, 0.01, 0.05, 0.1]:
        c = int(len(q) * f)
        print(f"   en kotu %{100 * f:g} satir -> kutlenin %{100 * q[:c].sum() / tot:.1f}'i")
    out[kesim] = {
        "sicak_rmsle": float(np.sqrt(e2.mean())),
        "hedef_sifir_kutle_pay": float(e2[m0].sum() / tot),
    }
json_yaz("teshis", out)
