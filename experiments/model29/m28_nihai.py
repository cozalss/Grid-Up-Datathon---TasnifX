"""NIHAI sicak kestirimci: tam tarif + tum kesimlerde olcum + 2026 mevsim katsayilari."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()


def tahmin(gec, hed, kesim, mevsim=None, lam=0.0):
    """mevsim: {takvim_ayi: kayma} sozlugu. Dondurur: log1p uzayinda tahmin."""
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    kok = float(gec.ly.mean())
    g = hed.tanim.map(gr).values
    k7 = geri_dolgu(hed, oz.k7, oz.ly_all, kok=kok)
    allm = geri_dolgu(hed, oz.ly_all, kok=kok)
    s28 = geri_dolgu(hed, oz.s28, oz.ly_all, kok=kok)
    gd = gec.copy()
    gd["dw"] = gd.tarih.dt.dayofweek
    gd["dev"] = gd.ly - gd.groupby("tanim").ly.transform("mean")
    dwt = gd[gd.tanim.map(oz.maxt) >= 1].groupby(["tanim", "dw"]).dev.mean()
    dw = np.nan_to_num(
        pd.Series(
            dwt.reindex(
                pd.MultiIndex.from_arrays([hed.tanim.values, hed.tarih.dt.dayofweek.values])
            ).values
        ).values.astype(float)
    )
    p = np.where(g == "B_bayat", 0.7 * s28 + 0.3 * allm, 0.75 * k7 + 0.25 * allm) + 0.5 * dw
    if mevsim is not None and lam > 0:
        d = pd.Series(hed.tarih.dt.month.values).map(mevsim).fillna(0).values
        p = p + lam * np.where(np.isin(g, ["A_tum_sifir", "C_son28_sifir", "D_son7_sifir"]), 0, d)
    for gg, cc in [("A_tum_sifir", 0.6), ("C_son28_sifir", 1.1), ("D_son7_sifir", 1.3)]:
        p = np.where(g == gg, cc, p)
    return p, g


print("== NIHAI KESTIRIMCI (mevsim duzeltmesi YOK) - tum kesimler, SICAK RMSLE ==")
son = {}
for kesim in KESIMLER + ["2025-03-31", "2025-12-31"]:
    uf = 3 if kesim == "2025-12-31" else 4
    gec, hed = hazirla(tr, kesim, uf)
    p, g = tahmin(gec, hed, kesim)
    y = hed.ly.values
    E = g == "E_normal"
    # taban karsilastirmalari
    oz = ozellik(gec, kesim)
    kok = float(gec.ly.mean())
    b28 = geri_dolgu(hed, oz.k28, oz.ly_all, kok=kok)
    b7 = geri_dolgu(hed, oz.k7, oz.ly_all, kok=kok)
    r28 = float(np.sqrt(((y - b28) ** 2).mean()))
    r7 = float(np.sqrt(((y - b7) ** 2).mean()))
    r = float(np.sqrt(((y - p) ** 2).mean()))
    son[kesim] = {
        "n": int(len(y)),
        "taban_son28": r28,
        "taban_son7": r7,
        "nihai": r,
        "nihai_E": float(np.sqrt(((y[E] - p[E]) ** 2).mean())),
    }
    print(
        f"{kesim}: n={len(y):7,d} | son28 {r28:.4f} | son7 {r7:.4f} | NIHAI {r:.4f} (kazanc son28'e gore {r28 - r:+.4f})"
    )
json_yaz("nihai_kestirimci", son)
print(
    f"\n4 ana kesim ort: son28 {np.mean([son[k]['taban_son28'] for k in KESIMLER]):.4f} -> "
    f"NIHAI {np.mean([son[k]['nihai'] for k in KESIMLER]):.4f}"
)

# ---- 2026 icin mevsim kaymalari ----
print("\n== 2026 Nis-Tem MEVSIM KAYMALARI ==")
tr2 = tr.copy()
tr2["ly"] = np.log1p(tr2.tuketim)
tr2["ym"] = tr2.tarih.dt.to_period("M")
sel = tr2[tr2.groupby("tanim").tarih.transform("nunique") >= 120]
sel = sel[sel.groupby("tanim").tuketim.transform("max") >= 1]
sel["dev"] = sel.ly - sel.groupby("tanim").ly.transform("mean")
prof = sel.groupby("ym").dev.mean()
P = {str(a): float(v) for a, v in prof.items()}
drift = np.mean(
    [prof[pd.Period(f"2026-{m:02d}")] - prof[pd.Period(f"2025-{m:02d}")] for m in [1, 2, 3]]
)
pmar26 = prof[pd.Period("2026-03")]
mevsim = {m: float(prof[pd.Period(f"2025-{m:02d}")] + drift - 0.75 * pmar26) for m in [4, 5, 6, 7]}
print(f"  yil-uzeri drift (Oca-Mar 2026 - 2025) = {drift:+.4f}; P[2026-03]={pmar26:+.4f}")
for m in [4, 5, 6, 7]:
    print(f"  ay {m}: kayma {mevsim[m]:+.4f}")
mevsim_driftsiz = {m: float(prof[pd.Period(f"2025-{m:02d}")] - 0.75 * pmar26) for m in [4, 5, 6, 7]}
print("  (drift eklenmeden:", {m: round(v, 3) for m, v in mevsim_driftsiz.items()}, ")")

# ---- takvim ikizinde formulun dogrulanmasi ----
print("\n== FORMUL DOGRULAMA: kesim 2025-03-31, formul-tabanli mevsim duzeltmesi ==")
gec, hed = hazirla(tr, "2025-03-31", 4)
p0, g = tahmin(gec, hed, "2025-03-31")
y = hed.ly.values
pmar25 = prof[pd.Period("2025-03")]
allP = float(np.mean([prof[pd.Period(f"2025-{m:02d}")] for m in [1, 2, 3]]))
mev25 = {
    m: float(prof[pd.Period(f"2025-{m:02d}")] - (0.75 * pmar25 + 0.25 * allP)) for m in [4, 5, 6, 7]
}
print("  formulun ongordugu kaymalar:", {m: round(v, 3) for m, v in mev25.items()})
gercek = {
    m: float(
        (
            y[(g == "E_normal") & (hed.tarih.dt.month.values == m)]
            - p0[(g == "E_normal") & (hed.tarih.dt.month.values == m)]
        ).mean()
    )
    for m in [4, 5, 6, 7]
}
print("  gercek artik ortalamalari:  ", {m: round(v, 3) for m, v in gercek.items()})
r0 = float(np.sqrt(((y - p0) ** 2).mean()))
for lam in [0, 0.5, 0.75, 1.0]:
    p2, _ = tahmin(gec, hed, "2025-03-31", mevsim=mev25, lam=lam)
    rr = float(np.sqrt(((y - p2) ** 2).mean()))
    print(f"  lam={lam:.2f}: SICAK RMSLE {rr:.4f} ({rr - r0:+.4f})")
json_yaz(
    "mevsim_2026",
    {
        "ay_profili": P,
        "drift": float(drift),
        "kayma_2026": mevsim,
        "kayma_2026_driftsiz": mevsim_driftsiz,
        "ikiz_formul": mev25,
        "ikiz_gercek": gercek,
    },
)
