"""EKSEN 2c: takvim-ayi duzeltmesi DURUST geri-test (profil sadece <=kesim verisinden)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()


def ay_profili(gec):
    """<=kesim verisinden takvim-ayi profili: trafo-ici sapmanin (yil,ay) ortalamasi -> ay ortalamasi."""
    g = gec.copy()
    n = g.groupby("tanim").tarih.transform("nunique")
    g = g[n >= 120]  # yeterince gecmisi olanlar
    g = g[g.groupby("tanim").tuketim.transform("max") >= 1]
    g["dev"] = g.ly - g.groupby("tanim").ly.transform("mean")
    g["ym"] = g.tarih.dt.to_period("M")
    return g.groupby("ym").dev.mean()


SENARYO = [
    ("2025-11-30", 4, [2, 3, 4]),
    ("2025-12-31", 3, [1, 2, 3]),
]  # (kesim, ufuk, gecerli ay_ofsetleri)
res = {}
for kesim, uf, gecerli in SENARYO:
    k = pd.Timestamp(kesim)
    gec, hed = hazirla(tr, kesim, uf)
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    kok = float(gec.ly.mean())
    prof = ay_profili(gec)
    base = 0.8 * geri_dolgu(hed, oz.k7, oz.ly_all, kok=kok) + 0.2 * geri_dolgu(
        hed, oz.ly_all, kok=kok
    )
    # taban penceresinin ayi = kesim ayi
    taban_ay = pd.Period(k, freq="M")
    hed_ym = hed.tarih.dt.to_period("M")
    # gecen yilin ayni takvim ayi
    onceki = pd.PeriodIndex([p - 12 for p in hed_ym])
    delta = pd.Series(onceki.map(prof).values, index=hed.index) - prof.get(taban_ay - 12, np.nan)
    delta_k = pd.Series(onceki.map(prof).values, index=hed.index) - prof.get(
        taban_ay, np.nan
    )  # taban ayi BU yildan
    m = (hed.tanim.map(gr).values == "E_normal") & hed.ay_ofset.isin(gecerli).values
    print(
        f"\n===== {kesim} (ufuk {uf}), gecerli ay ofsetleri {gecerli}, E_normal n={m.sum():,} ====="
    )
    print(
        f"  profil taban ayi {taban_ay} = {prof.get(taban_ay, np.nan):+.4f}; {taban_ay - 12} = {prof.get(taban_ay - 12, np.nan)}"
    )
    r0 = np.sqrt(((hed.ly.values[m] - base[m]) ** 2).mean())
    print(f"  duzeltmesiz RMSLE {r0:.4f}  (artik ort {(hed.ly.values[m] - base[m]).mean():+.4f})")
    res[kesim] = {"taban_rmsle": float(r0), "lam": {}}
    for ad, dl in [("gecenyil-taban", delta), ("buyil-taban", delta_k)]:
        d = dl.values
        if np.isnan(d).all():
            print(f"  {ad}: profil yok")
            continue
        d = np.nan_to_num(d)
        print(f"  {ad}: ortalama delta {d[m].mean():+.4f}")
        for lam in [0, 0.25, 0.5, 0.75, 1.0, 1.25]:
            p = base + lam * d
            rr = float(np.sqrt(((hed.ly.values[m] - p[m]) ** 2).mean()))
            res[kesim]["lam"][f"{ad}@{lam}"] = rr
            print(f"     lam={lam:.2f} RMSLE {rr:.4f} ({rr - r0:+.4f})")
    # (c) trafonun KENDI gecen yil ayni ay seviyesi
    g = gec.copy()
    g["ym"] = g.tarih.dt.to_period("M")
    aylik = g.groupby(["tanim", "ym"]).ly.mean()
    oy = pd.Series(list(zip(hed.tanim.values, onceki)))
    kendi = pd.Series(
        aylik.reindex(pd.MultiIndex.from_arrays([hed.tanim.values, onceki])).values, index=hed.index
    )
    tb_oy = pd.Series(
        aylik.reindex(
            pd.MultiIndex.from_arrays(
                [hed.tanim.values, pd.PeriodIndex([taban_ay - 12] * len(hed))]
            )
        ).values,
        index=hed.index,
    )
    dk = (kendi - tb_oy).values
    var = ~np.isnan(dk)
    print(
        f"  KENDI gecen-yil delta: kapsam %{100 * var[m].mean():.1f}, ort {np.nanmean(dk[m & var]):+.3f} std {np.nanstd(dk[m & var]):.3f}"
    )
    dk2 = np.where(var, dk, 0.0)
    for lam in [0, 0.25, 0.5, 0.75, 1.0]:
        p = base + lam * dk2
        rr = float(np.sqrt(((hed.ly.values[m] - p[m]) ** 2).mean()))
        res[kesim]["lam"][f"kendi@{lam}"] = rr
        print(f"     kendi lam={lam:.2f} RMSLE {rr:.4f} ({rr - r0:+.4f})")
json_yaz("eksen2c_mevsim_durust", res)
