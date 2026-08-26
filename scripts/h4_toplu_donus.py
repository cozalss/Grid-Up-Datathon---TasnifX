"""H4 -- 2026-05-11 TOPLU DONUS: son_islem_olay.py'nin ESKI gun-0 / IC BOSLUK
kaymalari PARTI BUYUKLUGUNU AYIRT ETMIYOR. Dogumda ayirt ediyor (100+ -> -0,106,
<100 -> -0,604) ama donuste ETMIYOR (hepsine -0,526 / -0,558 / -0,529).

Testte 2026-05-11 tek gunde:
    1.634 SOGUK trafo DOGUYOR (parti 100+ -> -0,106 uygulaniyor)
      896 SICAK trafo DONUYOR (parti ayrimi YOK -> -0,505/-0,556 uygulaniyor)
      129 IC BOSLUK donusu    (parti ayrimi YOK -> -0,528 uygulaniyor)

Ayni gun, ayni toplu-katilim olayi. Dogum kolu "gun TAM" diyor, donus kolu
"gun KISMI" diyor. Ikisi birden dogru olamaz.

OLCUM: train'de DONUS gunlerini parti buyuklugune gore ayir, 100+ kovasi var mi?

    uv run python scripts/h4_toplu_donus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "scripts"))

import eksen2_kesin as ek  # noqa: E402

TR_BAS = pd.Timestamp("2025-01-01")
TR_SON = pd.Timestamp("2026-03-31")
IK0, IK1 = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")


def ozet(x, ad, gen=46):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return f"{ad:<{gen}} n={len(x):>5}  --"
    kk = []
    for K in (0, 1, 5, 10, 25, 50):
        if len(x) > K:
            v = x if K == 0 else x.drop(x.abs().sort_values(ascending=False).index[:K])
            kk.append(f"K{K}={v.mean():+.3f}")
    return (
        f"{ad:<{gen}} n={len(x):>5,} ort {x.mean():>+8.4f} "
        f"sh {x.std(ddof=1) / np.sqrt(len(x)):>6.4f} med {x.median():>+8.4f}  ["
        + " ".join(kk)
        + "]"
    )


tr = pd.read_csv(KOK / "data/raw/train.csv", dtype={"tanim": str}, parse_dates=["tarih"])
tr["r"] = np.log1p(tr["tuketim"].to_numpy("float64")) - np.log1p(tr["guc"].to_numpy("float64"))
T, bT = ek.hazirla(tr[["tanim", "tarih", "r", "tuketim"]], "r")
T = ek.yerel(T)
print(f"train panel gun std = {bT.std():.4f}")

print("=" * 118)
print("1) TRAIN -- DONUS GUNU, PARTI BUYUKLUGU KOVALARI (panel duzeltilmis)")
print("=" * 118)
T_don = T[T["bosluk"] > 0].copy()
T_don["parti_n"] = T_don["tarih"].map(T_don["tarih"].value_counts())
print(f"toplam donus olayi = {len(T_don):,}, essiz gun = {T_don['tarih'].nunique()}")
print(f"en buyuk donus partileri: {T_don['tarih'].value_counts().head(8).to_dict()}")
print()
pk = pd.cut(
    T_don["parti_n"], [0, 4, 19, 49, 99, 5000], labels=["1-4", "5-19", "20-49", "50-99", "100+"]
)
print(ozet(T_don["sap_i"], "DONUS gunu -- HEPSI"))
for k, x in T_don.groupby(pk, observed=True):
    print(ozet(x["sap_i"], f"   parti {k}"))
print()
ik = T_don[(T_don["tarih"] >= IK0) & (T_don["tarih"] <= IK1)]
print(ozet(ik["sap_i"], "IKIZ PENCERE (2025-04..07) -- HEPSI"))
for k, x in ik.groupby(pd.cut(ik["parti_n"], [0, 19, 5000], labels=["<20", "20+"]), observed=True):
    print(ozet(x["sap_i"], f"   IKIZ parti {k}"))

print()
print("=" * 118)
print("2) TRAIN -- EN BUYUK DONUS GUNLERI TEK TEK")
print("=" * 118)
print(f"   {'tarih':<12} {'n':>5} {'panel b':>9} {'DUZELT sap':>11} {'HAM sap':>10}")
for t, x in T_don[T_don["parti_n"] >= 20].groupby("tarih"):
    print(
        f"   {str(t.date()):<12} {int(x['sap_i'].notna().sum()):>5} "
        f"{float(x['b'].mean()):>+9.4f} {float(x['sap_i'].mean()):>+11.4f} "
        f"{float(x['bosluk'].median()):>+10.1f}"
    )

print()
print("=" * 118)
print("3) TRAIN -- AYNI GUN HEM DOGUM HEM DONUS OLAN GUNLER (toplu katilim imzasi)")
print("=" * 118)
T_dog = T[(T["yas"] == 0) & (T["ilk"] > TR_BAS)].copy()
T_dog["parti_n"] = T_dog["ilk"].map(T_dog["ilk"].value_counts())
dog_gun = T_dog.groupby("ilk").size()
don_gun = T_don.groupby("tarih").size()
ortak = pd.DataFrame({"dogum": dog_gun, "donus": don_gun}).dropna()
ortak["toplam"] = ortak["dogum"] + ortak["donus"]
ortak = ortak.sort_values("toplam", ascending=False)
print(ortak.head(12).to_string())
print()
print("   Bu gunlerde DOGUM ve DONUS dususu:")
print(f"   {'tarih':<12} {'n_dog':>6} {'D_dog':>9} {'n_don':>6} {'D_don':>9}")
for t in ortak.head(12).index:
    a = T_dog[T_dog["ilk"] == t]["sap_i"]
    b = T_don[T_don["tarih"] == t]["sap_i"]
    print(
        f"   {str(pd.Timestamp(t).date()):<12} {int(a.notna().sum()):>6} "
        f"{a.mean():>+9.4f} {int(b.notna().sum()):>6} {b.mean():>+9.4f}"
    )

print()
print("=" * 118)
print("4) 2026-05-11 dMSE HESABI -- uretimde ne uygulaniyor, dogrusu ne olabilir")
print("=" * 118)
te = pd.read_csv(KOK / "data/raw/test.csv", dtype={"tanim": str}, parse_dates=["tarih"])
trs = tr.groupby("tanim")["tarih"].max()
m = te.sort_values(["tanim", "tarih"], kind="mergesort").copy()
g = m.groupby("tanim", observed=True)
m["ilk"] = g["tarih"].transform("min")
m["yas"] = (m["tarih"] - m["ilk"]).dt.days
m["bosluk"] = ((m["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
m["tr_son"] = m["tanim"].map(trs)
m["yeni"] = m["tr_son"].isna()
m["gb"] = (m["tarih"] - m["tr_son"]).dt.days - 1.0
ilk = m["yas"] == 0
gr = {
    "ESKI gun-0 bosluk 1-60g": (ilk & ~m["yeni"] & m["gb"].between(1, 60), -0.5259, -0.0205),
    "ESKI gun-0 bosluk 60+g": (ilk & ~m["yeni"] & (m["gb"] > 60), -0.5576, -0.0020),
    "IC BOSLUK donusu": (m["bosluk"] > 0, -0.5289, -0.0011),
}
N = len(te)
print(
    f"{'grup':<26} {'toplam':>7} {'05-11':>7} {'pay':>7} {'kayma':>8} {'dMSE_toplam':>12} {'dMSE_0511':>11}"
)
top_0511 = 0.0
for ad, (msk, dg, dv) in gr.items():
    k = 0.6 * (dg - dv)
    n_t = int(msk.sum())
    n_5 = int((msk & (m["tarih"] == "2026-05-11")).sum())
    b = dg - dv
    # dMSE = p*(k^2 - 2*k*b), b = gercek yanlilik
    d_t = (n_t / N) * (k**2 - 2 * k * b)
    d_5 = (n_5 / N) * (k**2 - 2 * k * b)
    top_0511 += d_5
    print(
        f"{ad:<26} {n_t:>7,} {n_5:>7,} {n_5 / max(n_t, 1):>6.1%} {k:>+8.4f} "
        f"{d_t:>+12.6f} {d_5:>+11.6f}"
    )
print(f"\n05-11'e bagli TOPLAM beklenen kazanc (uretim varsayimiyla) = {top_0511:+.6f}")
# Alternatif: 05-11 partisi TOPLU KATILIM ise gercek dusus ~ dogum 100+ (-0,106)
print("\nEGER 2026-05-11 toplu katilim ise (gercek dusus dogumdaki gibi ~-0,106):")
ters = 0.0
for ad, (msk, dg, dv) in gr.items():
    k = 0.6 * (dg - dv)
    n_5 = int((msk & (m["tarih"] == "2026-05-11")).sum())
    b_alt = -0.1060 - dv  # gercek yanlilik toplu katilim varsayimiyla
    d_5 = (n_5 / N) * (k**2 - 2 * k * b_alt)
    ters += d_5
    print(f"  {ad:<26} n={n_5:>5,} kayma={k:+.4f} b_alt={b_alt:+.4f} dMSE={d_5:+.7f}")
print(f"  TOPLAM = {ters:+.6f}   (uretim varsayimiyla {top_0511:+.6f})")
print(f"  RISK BANDI (iki senaryo farki) = {ters - top_0511:+.6f}")
