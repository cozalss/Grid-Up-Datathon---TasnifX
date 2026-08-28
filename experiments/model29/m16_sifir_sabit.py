"""EKSEN 4b: sifir-durumlu trafolar icin optimal sabit; ve TEST'teki bilesim."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *

tr = yukle()
KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("experiments/model29/x"))))
te = pd.read_csv("data/raw/test.csv", parse_dates=["tarih"])
print("TEST:", te.tarih.min(), te.tarih.max(), len(te), te.tanim.nunique())
gorulen = set(tr.tanim)
te_sicak = te[te.tanim.isin(gorulen)]
print(
    f"test sicak satir {len(te_sicak):,} (%{100 * len(te_sicak) / len(te):.1f}), trafo {te_sicak.tanim.nunique():,}"
)
k = pd.Timestamp("2026-03-31")
gg = tr.groupby("tanim")
oz = pd.DataFrame({"maxt": gg.tuketim.max(), "sonT": gg.tarih.max(), "nsat": gg.size()})
oz["bosluk"] = (k - oz.sonT).dt.days
for W in [7, 28, 56]:
    oz[f"max{W}"] = tr[tr.tarih > k - pd.Timedelta(days=W)].groupby("tanim").tuketim.max()
t = te_sicak.join(oz, on="tanim")


def grup(r):
    if r.maxt < 1:
        return "A_tum_gecmis_sifir"
    if r.bosluk > 14:
        return "B_bayat"
    if not (r.max28 >= 1):
        return "C_son28_sifir"
    if not (r.max7 >= 1):
        return "D_son7_sifir"
    return "E_normal"


t["grup"] = t.apply(grup, axis=1)
print("\nTEST'te sicak grup dagilimi:")
print(t.groupby("grup").agg(satir=("tanim", "size"), trafo=("tanim", "nunique")))

# --- optimal sabit taramasi: her kesimde, "sifir durumlu" satirlar icin
print("\n=== sifir-durumlu (A/C/D) satirlar icin sabit c taramasi ===")
CS = np.arange(0, 4.01, 0.25)
tab = {}
for kesim in KESIMLER:
    kk = pd.Timestamp(kesim)
    gec, hed = hazirla(tr, kesim)
    gg = gec.groupby("tanim")
    oz = pd.DataFrame(
        {"maxt": gg.tuketim.max(), "sonT": gg.tarih.max(), "nsat": gg.size(), "tumly": gg.ly.mean()}
    )
    oz["bosluk"] = (kk - oz.sonT).dt.days
    for W in [7, 28, 56]:
        oz[f"max{W}"] = gec[gec.tarih > kk - pd.Timedelta(days=W)].groupby("tanim").tuketim.max()
    h = hed.join(oz, on="tanim")
    h["grup"] = h.apply(grup, axis=1)
    tab[kesim] = {}
    for gr in ["A_tum_gecmis_sifir", "C_son28_sifir", "D_son7_sifir"]:
        s = h[h.grup == gr]
        if len(s) == 0:
            continue
        e = [float(np.sqrt(((c - s.ly) ** 2).mean())) for c in CS]
        tab[kesim][gr] = {
            "n": len(s),
            "c_egrisi": dict(zip([f"{c:.2f}" for c in CS], e)),
            "en_iyi_c": float(CS[int(np.argmin(e))]),
            "c0_rmsle": e[0],
            "en_iyi_rmsle": float(min(e)),
        }
        print(
            f"{kesim} {gr:20s} n={len(s):6,d} c=0 -> {e[0]:.3f} | en iyi c={CS[int(np.argmin(e))]:.2f} -> {min(e):.3f}"
        )
    # tum A/C/D birlikte
    s = h[h.grup.isin(["A_tum_gecmis_sifir", "C_son28_sifir", "D_son7_sifir"])]
    e = [float(np.sqrt(((c - s.ly) ** 2).mean())) for c in CS]
    tab[kesim]["ACD"] = {
        "n": len(s),
        "c_egrisi": dict(zip([f"{c:.2f}" for c in CS], e)),
        "en_iyi_c": float(CS[int(np.argmin(e))]),
    }
    print(
        f"{kesim} {'ACD-birlesik':20s} n={len(s):6,d} c=0 -> {e[0]:.3f} | en iyi c={CS[int(np.argmin(e))]:.2f} -> {min(e):.3f}"
    )
json_yaz("eksen4b_sabit", tab)
json_yaz(
    "test_grup_dagilimi",
    {
        g: {"satir": int(v[0]), "trafo": int(v[1])}
        for g, v in t.groupby("grup")
        .agg(satir=("tanim", "size"), trafo=("tanim", "nunique"))
        .iterrows()
    },
)
