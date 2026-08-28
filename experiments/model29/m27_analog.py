"""Gercek testin TAKVIM IKIZI: kesim 2025-03-31 -> Nis-Tem 2025.
Ay profili YARI-ORNEK ile (A yarisinda kestir, B yarisinda uygula) => kesitsel transfer testi."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()
kesim = "2025-03-31"
k = pd.Timestamp(kesim)
gec, hed = hazirla(tr, kesim, 4)
oz = ozellik(gec, kesim)
gr = grupla(oz)
kok = float(gec.ly.mean())
g = hed.tanim.map(gr).values
y = hed.ly.values
print(
    f"kesim {kesim}: gecmis {len(gec):,} satir, SICAK hedef {len(hed):,} satir / {hed.tanim.nunique():,} trafo"
)
print("grup dagilimi:", hed.tanim.map(gr).value_counts().to_dict())
C = {c: geri_dolgu(hed, oz[c], oz.ly_all, kok=kok) for c in ["k7", "k28", "ly_all", "s28"]}
gd = gec.copy()
gd["dw"] = gd.tarih.dt.dayofweek
gd["dev"] = gd.ly - gd.groupby("tanim").ly.transform("mean")
dwt = gd[gd.tanim.map(oz.maxt) >= 1].groupby(["tanim", "dw"]).dev.mean()
C["dw"] = np.nan_to_num(
    pd.Series(
        dwt.reindex(
            pd.MultiIndex.from_arrays([hed.tanim.values, hed.tarih.dt.dayofweek.values])
        ).values
    ).values.astype(float)
)
p = (
    np.where(
        g == "B_bayat", 0.7 * C["s28"] + 0.3 * C["ly_all"], 0.75 * C["k7"] + 0.25 * C["ly_all"]
    )
    + 0.5 * C["dw"]
)
for gg, cc in [("A_tum_sifir", 0.6), ("C_son28_sifir", 1.1), ("D_son7_sifir", 1.3)]:
    p = np.where(g == gg, cc, p)
r0 = float(np.sqrt(((y - p) ** 2).mean()))
E = g == "E_normal"
print(
    f"\nmevsim duzeltmesiz: TUM SICAK {r0:.4f} | E_normal {np.sqrt(((y[E] - p[E]) ** 2).mean()):.4f}"
)
ao = hed.tarih.dt.month.values
for m in [4, 5, 6, 7]:
    mm = E & (ao == m)
    print(
        f"   ay {m}: artik ort {(y[mm] - p[mm]).mean():+.4f} RMSLE {np.sqrt(((y[mm] - p[mm]) ** 2).mean()):.4f}"
    )
# --- yari ornek: A yarisindan ay profili, B yarisinda degerlendir ---
rng = np.random.default_rng(7)
tanimlar = hed.tanim.unique()
yari = pd.Series(rng.integers(0, 2, len(tanimlar)), index=tanimlar)
hy = hed.tanim.map(yari).values
res = {"duzeltmesiz_tum": r0, "duzeltmesiz_E": float(np.sqrt(((y[E] - p[E]) ** 2).mean()))}
for lam in [0, 0.25, 0.5, 0.75, 1.0]:
    p2 = p.copy()
    toplam = []
    for h in [0, 1]:
        kay = ~(hy == h)  # profili DIGER yaridan kestir
        prof = {
            m: float((y[E & kay & (ao == m)] - p[E & kay & (ao == m)]).mean()) for m in [4, 5, 6, 7]
        }
        d = pd.Series(ao).map(prof).values
        sel = hy == h
        p2[sel] = p[sel] + lam * d[sel]
    rr = float(np.sqrt(((y - p2) ** 2).mean()))
    rE = float(np.sqrt(((y[E] - p2[E]) ** 2).mean()))
    res[f"global_lam{lam}"] = {"tum": rr, "E": rE}
    print(f"  global ay profili lam={lam:.2f}: TUM {rr:.4f} ({rr - r0:+.4f}) | E {rE:.4f}")
# --- ilce x ay profili (yari ornekle) ---
ilce = hed.ilce.values
for lam in [0.5, 0.75, 1.0]:
    p2 = p.copy()
    for h in [0, 1]:
        kay = ~(hy == h)
        dd = (
            pd.DataFrame({"ilce": ilce, "ay": ao, "r": y - p})[kay & E]
            .groupby(["ilce", "ay"])
            .r.agg(["mean", "size"])
        )
        gm = {
            m: float((y[E & kay & (ao == m)] - p[E & kay & (ao == m)]).mean()) for m in [4, 5, 6, 7]
        }
        # guvenilirlik kucultmesi: n/(n+K)
        K = 300
        val = dd.apply(
            lambda r_: (
                gm[r_.name[1]] + (r_["mean"] - gm[r_.name[1]]) * r_["size"] / (r_["size"] + K)
            ),
            axis=1,
        )
        d = pd.Series(pd.MultiIndex.from_arrays([ilce, ao]).map(val)).astype(float)
        d = d.fillna(pd.Series(ao).map(gm)).values
        sel = hy == h
        p2[sel] = p[sel] + lam * d[sel]
    rr = float(np.sqrt(((y - p2) ** 2).mean()))
    rE = float(np.sqrt(((y[E] - p2[E]) ** 2).mean()))
    res[f"ilce_lam{lam}"] = {"tum": rr, "E": rE}
    print(f"  ilce x ay profili lam={lam:.2f}: TUM {rr:.4f} ({rr - r0:+.4f}) | E {rE:.4f}")
json_yaz("analog_2025_03_31", res)
