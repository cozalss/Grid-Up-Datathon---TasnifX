"""EKSEN 3 (trend), 5 (kisa gecmis shrinkage), 6 (haftanin gunu)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()
res = {"trend": {}, "shrink": {}, "gun": {}}
for kesim in KESIMLER:
    k = pd.Timestamp(kesim)
    gec, hed = hazirla(tr, kesim)
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    kok = float(gec.ly.mean())
    base = 0.8 * geri_dolgu(hed, oz.k7, oz.ly_all, kok=kok) + 0.2 * geri_dolgu(
        hed, oz.ly_all, kok=kok
    )
    E = hed.tanim.map(gr).values == "E_normal"
    y = hed.ly.values
    r0 = float(np.sqrt(((y[E] - base[E]) ** 2).mean()))
    rT = float(np.sqrt(((y - base) ** 2).mean()))
    print(f"\n===== {kesim} | taban E {r0:.4f} / TUM sicak {rT:.4f} =====")

    # --- EKSEN 3: son 90 gun egimi (gunluk log egim) ---
    g = gec[gec.tarih > k - pd.Timedelta(days=90)].copy()
    g["x"] = (g.tarih - k).dt.days.astype(float)
    gb = g.groupby("tanim")
    xm = gb.x.transform("mean")
    ym_ = gb.ly.transform("mean")
    num = ((g.x - xm) * (g.ly - ym_)).groupby(g.tanim).sum()
    den = ((g.x - xm) ** 2).groupby(g.tanim).sum()
    egim = (num / den).replace([np.inf, -np.inf], np.nan)
    n90 = gb.size()
    egim[n90 < 30] = np.nan
    # hedef gununun kesimden uzakligi
    dt = (hed.tarih - k).dt.days.values.astype(float)
    e = hed.tanim.map(egim).fillna(0).values
    print(
        f"  egim dagilimi: std {np.nanstd(egim):.5f}/gun, kapsam %{100 * hed.tanim.isin(egim.dropna().index).mean():.1f}"
    )
    res["trend"][kesim] = {}
    for lam in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        # kirpma ile: egim*gun, +-1.5 log ile sinirli
        p = base + np.clip(lam * e * dt, -1.5, 1.5)
        rr = float(np.sqrt(((y[E] - p[E]) ** 2).mean()))
        res["trend"][kesim][f"lam{lam}"] = rr
        print(f"   trend lam={lam:<4} E RMSLE {rr:.4f} ({rr - r0:+.4f})")

    # --- EKSEN 5: kisa gecmis -> grup ortalamasina shrinkage ---
    gucm = gec.groupby("guc").ly.mean()
    ilcem = gec.groupby("ilce").ly.mean()
    gi = gec.groupby(["guc", "ilce"]).ly.mean()
    prior = hed.set_index(["guc", "ilce"]).index.map(gi)
    prior = pd.Series(prior, index=hed.index).fillna(hed.guc.map(gucm)).fillna(kok).values
    nsat = hed.tanim.map(oz.nsat).values.astype(float)
    res["shrink"][kesim] = {}
    print(f"   [shrink] n<30 satir: {(nsat < 30).sum():,} / n<15: {(nsat < 15).sum():,}")
    for K in [0, 1, 3, 5, 10, 20, 40, 80]:
        w = nsat / (nsat + K)
        p = w * base + (1 - w) * prior
        rr = float(np.sqrt(((y[E] - p[E]) ** 2).mean()))
        rrT = float(np.sqrt(((y - p) ** 2).mean()))
        res["shrink"][kesim][f"K{K}"] = {"E": rr, "tum": rrT}
        print(f"   shrink K={K:<3} E {rr:.4f} ({rr - r0:+.4f}) | TUM {rrT:.4f} ({rrT - rT:+.4f})")

    # --- EKSEN 6: haftanin gunu ---
    gd = gec.copy()
    gd["dw"] = gd.tarih.dt.dayofweek
    gd["dev"] = gd.ly - gd.groupby("tanim").ly.transform("mean")
    gsel = gd[gd.tanim.map(oz.maxt) >= 1]
    dwg = gsel.groupby("dw").dev.mean()
    dwg = dwg - dwg.mean()
    dwt = gsel.groupby(["tanim", "dw"]).dev.mean()
    print("   gun etkisi (global):", " ".join(f"{d}:{v:+.3f}" for d, v in dwg.items()))
    hdw = hed.tarih.dt.dayofweek.values
    dglob = pd.Series(hdw).map(dwg).fillna(0).values
    dtraf = (
        pd.Series(dwt.reindex(pd.MultiIndex.from_arrays([hed.tanim.values, hdw])).values)
        .fillna(0)
        .values
    )
    res["gun"][kesim] = {}
    for ad, d in [("global", dglob), ("trafo", dtraf)]:
        for lam in [0, 0.5, 1.0]:
            p = base + lam * d
            rr = float(np.sqrt(((y[E] - p[E]) ** 2).mean()))
            res["gun"][kesim][f"{ad}@{lam}"] = rr
            print(f"   gun {ad}@{lam} E {rr:.4f} ({rr - r0:+.4f})")
    res["gun"][kesim]["taban_E"] = r0
json_yaz("eksen356_trend_shrink_gun", res)
