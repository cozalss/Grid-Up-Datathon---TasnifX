"""BIRLESIK sicak kestirimci + LOO (bir kesimde ayarla, digerinde dogrula)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()
CACHE = {}
for kesim in KESIMLER:
    gec, hed = hazirla(tr, kesim)
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    kok = float(gec.ly.mean())
    g = hed.tanim.map(gr).values
    y = hed.ly.values
    C = {
        c: geri_dolgu(hed, oz[c], oz.ly_all, kok=kok)
        for c in ["k3", "k7", "k14", "k28", "k91", "ly_all", "s7", "s28", "s56", "s91"]
    }
    gd = gec.copy()
    gd["dw"] = gd.tarih.dt.dayofweek
    gd["dev"] = gd.ly - gd.groupby("tanim").ly.transform("mean")
    dwt = gd[gd.tanim.map(oz.maxt) >= 1].groupby(["tanim", "dw"]).dev.mean()
    hdw = hed.tarih.dt.dayofweek.values
    C["dw"] = np.nan_to_num(
        pd.Series(
            dwt.reindex(pd.MultiIndex.from_arrays([hed.tanim.values, hdw])).values
        ).values.astype(float)
    )
    CACHE[kesim] = dict(C=C, g=g, y=y, n=len(y))


def kur(kesim, wE, wB, cA, cC, cD, wdw):
    d = CACHE[kesim]
    C = d["C"]
    g = d["g"]
    p = (
        np.where(
            g == "B_bayat",
            sum(w * C[c] for c, w in wB.items()),
            sum(w * C[c] for c, w in wE.items()),
        )
        + wdw * C["dw"]
    )
    for gg, cc in [("A_tum_sifir", cA), ("C_son28_sifir", cC), ("D_son7_sifir", cD)]:
        p = np.where(g == gg, cc, p)
    return p


def mse(kesim, **kw):
    d = CACHE[kesim]
    return float(((kur(kesim, **kw) - d["y"]) ** 2).mean())


TABAN = dict(wE={"k7": 1.0}, wB={"k7": 1.0}, cA=0.0, cC=0.0, cD=0.0, wdw=0.0)
E1 = {"k7": 0.8, "ly_all": 0.2}
B1 = {"s28": 0.7, "ly_all": 0.3}
ADIMLAR = [
    ("0 taban k7", dict(TABAN)),
    ("1 +E harman .8k7+.2all", {**TABAN, "wE": E1}),
    ("2 +B .7s28+.3all", {**TABAN, "wE": E1, "wB": B1}),
    ("3 +gun etkisi .5", {**TABAN, "wE": E1, "wB": B1, "wdw": 0.5}),
    ("4 +sifir sbt A.5 C1 D1", {"wE": E1, "wB": B1, "wdw": 0.5, "cA": 0.5, "cC": 1.0, "cD": 1.0}),
]
print("== ADIM ADIM (tam SICAK RMSLE) ==")
tab = {}
for ad, kw in ADIMLAR:
    rs = [np.sqrt(mse(k, **kw)) for k in KESIMLER]
    tab[ad] = {**{k: float(r) for k, r in zip(KESIMLER, rs)}, "ort": float(np.mean(rs))}
    print(f"{ad:26s} " + " ".join(f"{r:.4f}" for r in rs) + f" | ort {np.mean(rs):.4f}")
json_yaz("birlesik_adimlar", tab)

# --- sabitler AYRISTIRILABILIR: her grup icin ayri SSE ---
print("\n== LOO: sifir sabitleri (grup bazli, havuz SSE) ==")
IZG = np.arange(0, 3.01, 0.125)
Y = {}
for k in KESIMLER:
    d = CACHE[k]
    Y[k] = {gg: d["y"][d["g"] == gg] for gg in ["A_tum_sifir", "C_son28_sifir", "D_son7_sifir"]}
loo = {}
for test in KESIMLER:
    egit = [k for k in KESIMLER if k != test]
    sec = {}
    for gg in ["A_tum_sifir", "C_son28_sifir", "D_son7_sifir"]:
        yy = np.concatenate([Y[k][gg] for k in egit])
        sse = [((c - yy) ** 2).mean() for c in IZG]
        sec[gg] = float(IZG[int(np.argmin(sse))])
    kw = dict(
        wE=E1,
        wB=B1,
        wdw=0.5,
        cA=sec["A_tum_sifir"],
        cC=sec["C_son28_sifir"],
        cD=sec["D_son7_sifir"],
    )
    r = np.sqrt(mse(test, **kw))
    r0 = np.sqrt(mse(test, wE=E1, wB=B1, wdw=0.5, cA=0, cC=0, cD=0))
    loo[test] = {
        **{f"c{g[0]}": v for g, v in sec.items()},
        "rmsle": float(r),
        "c0_rmsle": float(r0),
        "kazanc": float(r0 - r),
    }
    print(
        f"  test {test}: cA={sec['A_tum_sifir']:.3f} cC={sec['C_son28_sifir']:.3f} cD={sec['D_son7_sifir']:.3f} -> {r:.4f} (c=0: {r0:.4f}, kazanc {r0 - r:+.4f})"
    )
json_yaz("loo_sifir_sabitleri", loo)

print("\n== LOO: E taban harmani a*k7+b*k28+c*ly_all ==")
IZ = np.arange(0, 1.01, 0.1)
loo2 = {}
onceki = {k: {} for k in KESIMLER}
for a in IZ:
    for b in IZ:
        if a + b > 1.0001:
            continue
        c = 1 - a - b
        wE = {"k7": a, "k28": b, "ly_all": c}
        for k in KESIMLER:
            onceki[k][(round(a, 2), round(b, 2))] = mse(
                k, wE=wE, wB=B1, wdw=0.5, cA=0.5, cC=1.0, cD=1.0
            )
for test in KESIMLER:
    egit = [k for k in KESIMLER if k != test]
    key = min(onceki[test].keys(), key=lambda kk: np.mean([onceki[k][kk] for k in egit]))
    loo2[test] = {
        "k7": key[0],
        "k28": key[1],
        "ly_all": round(1 - key[0] - key[1], 2),
        "rmsle": float(np.sqrt(onceki[test][key])),
    }
    print(
        f"  test {test}: k7={key[0]:.1f} k28={key[1]:.1f} all={1 - key[0] - key[1]:.1f} -> {np.sqrt(onceki[test][key]):.4f}"
    )
json_yaz("loo_E_agirlik", loo2)

print("\n== LOO: gun etkisi agirligi ==")
gw = {}
for w in [0, 0.25, 0.5, 0.75, 1.0]:
    gw[w] = {k: mse(k, wE=E1, wB=B1, wdw=w, cA=0.5, cC=1.0, cD=1.0) for k in KESIMLER}
for test in KESIMLER:
    egit = [k for k in KESIMLER if k != test]
    b = min(gw.keys(), key=lambda w: np.mean([gw[w][k] for k in egit]))
    print(
        f"  test {test}: en iyi wdw={b} -> {np.sqrt(gw[b][test]):.4f} (wdw=0: {np.sqrt(gw[0][test]):.4f})"
    )
json_yaz("loo_gun", {str(w): {k: float(np.sqrt(v)) for k, v in d.items()} for w, d in gw.items()})
