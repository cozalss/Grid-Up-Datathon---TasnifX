"""EKSEN 2e -- MEKANIZMA AYRIMI: "sistem kesinti gunu" mu, "geriye dolgu
katilimi" mi?

Iki rakip mekanizma ayni gorunumu uretir ama testte ZIT recete gerektirir:

  (M1) KISMI GUN. Trafo o gun gercekten yarim gun olculmus (enerjilendirme,
       devreden cikma, veri hatti kopmasi). O gun DUSUK. Testte de dusuk olur.
  (M2) GERIYE DOLGU KATILIMI. Trafo zaten olculuyordu, sadece veri setine o
       gun eklendi. Ilk gun TAM bir gundur, dusus YOKTUR.

AYIRT EDICI: M1 bir SISTEM olayiysa, o gun PANELIN GERI KALANI da (kesintisiz
devam eden trafolar) dusmus olmali. M2'de panel etkilenmez.

Bu betik her buyuk olay gunu icin PANEL GUN ETKISINI (trafo etkisi cikarilmis,
yalnizca o gun kesintisiz devam eden trafolar) hesaplar ve olay gruplarinin
dususuyle yan yana koyar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TR_BAS = pd.Timestamp("2025-01-01")
TR_SON = pd.Timestamp("2026-03-31")


def main():
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    tr["r"] = np.log1p(tr["tuketim"].to_numpy("float64")) - np.log1p(tr["guc"].to_numpy("float64"))
    d = tr.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("tanim", observed=True)
    d["ilk"] = g["tarih"].transform("min")
    d["son"] = g["tarih"].transform("max")
    d["bosluk"] = ((d["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
    d["yas"] = (d["tarih"] - d["ilk"]).dt.days
    d["kalan"] = (d["son"] - d["tarih"]).dt.days

    # ---- PANEL GUN ETKISI: yalnizca o gun OLAYSIZ olan trafolar ----
    olaysiz = (d["yas"] > 2) & (d["kalan"] > 2) & (d["bosluk"] == 0)
    p = d[olaysiz].copy()
    p["c"] = p["r"] - p.groupby("tanim", observed=True)["r"].transform("mean")
    b = p.groupby("tarih")["c"].mean()
    b = b - b.mean()
    n_gun = p.groupby("tarih").size()
    print(
        f"panel gun etkisi: {len(b)} gun, gunluk std {b.std():.4f}, "
        f"gun basina ort {n_gun.mean():,.0f} trafo"
    )

    # ---- olay sayilari ----
    ilk = d.groupby("tanim", observed=True)["ilk"].first()
    son = d.groupby("tanim", observed=True)["son"].first()
    dogum = ilk[ilk > TR_BAS].value_counts()
    olum = son[son < TR_SON].value_counts()
    donus = d.loc[d["bosluk"] > 0, "tarih"].value_counts()

    # ---- olay gruplarinin YEREL dususu (ileri ref 7 gun) ----
    n = len(d)
    tan = d["tanim"].to_numpy()
    rv = d["r"].to_numpy("float64")
    bs = d["bosluk"].to_numpy()
    ileri = np.full(n, np.nan)
    geri = np.full(n, np.nan)
    for i in range(n):
        t = tan[i]
        acc = 0.0
        k = 0
        for j in range(i + 1, min(i + 8, n)):
            if tan[j] != t or bs[j] > 0:
                break
            acc += rv[j]
            k += 1
        if k >= 2:
            ileri[i] = acc / k
        acc = 0.0
        k = 0
        for j in range(i - 1, max(i - 8, -1), -1):
            if tan[j] != t:
                break
            if j + 1 != i and bs[j + 1] > 0:
                break
            acc += rv[j]
            k += 1
        if k >= 2:
            geri[i] = acc / k
    d["sap_i"] = rv - ileri
    d["sap_g"] = rv - geri

    dog = d[(d["yas"] == 0) & (d["ilk"] > TR_BAS)]
    don = d[d["bosluk"] > 0]
    sn = d[(d["kalan"] == 0) & (d["son"] < TR_SON)]

    print("\n" + "=" * 118)
    print("OLAY GUNLERI -- panel gun etkisi vs olay gruplarinin dususu")
    print("=" * 118)
    print(
        f"{'tarih':<12} {'dogum':>6} {'donus':>6} {'olum':>6} | {'PANEL b':>9} {'panel z':>8} | "
        f"{'dogum sap':>10} {'n':>5} | {'donus sap':>10} {'n':>5} | {'olum sap':>10} {'n':>5}"
    )
    tarihler = sorted(
        set(dogum[dogum >= 20].index) | set(olum[olum >= 20].index) | set(donus[donus >= 20].index)
    )
    sd = b.std()
    satirlar = []
    for t in tarihler:
        pb = float(b.get(t, np.nan))
        xd = dog.loc[dog["tarih"] == t, "sap_i"].dropna()
        xr = don.loc[don["tarih"] == t, "sap_i"].dropna()
        xs = sn.loc[sn["tarih"] == t, "sap_g"].dropna()
        satirlar.append(
            (
                t,
                int(dogum.get(t, 0)),
                int(donus.get(t, 0)),
                int(olum.get(t, 0)),
                pb,
                pb / sd,
                xd.mean() if len(xd) else np.nan,
                len(xd),
                xr.mean() if len(xr) else np.nan,
                len(xr),
                xs.mean() if len(xs) else np.nan,
                len(xs),
            )
        )
        print(
            f"{str(t.date()):<12} {int(dogum.get(t, 0)):>6} {int(donus.get(t, 0)):>6} "
            f"{int(olum.get(t, 0)):>6} | {pb:>+9.4f} {pb / sd:>+8.2f} | "
            f"{(xd.mean() if len(xd) else float('nan')):>+10.4f} {len(xd):>5} | "
            f"{(xr.mean() if len(xr) else float('nan')):>+10.4f} {len(xr):>5} | "
            f"{(xs.mean() if len(xs) else float('nan')):>+10.4f} {len(xs):>5}"
        )

    S = pd.DataFrame(
        satirlar,
        columns=[
            "tarih",
            "dogum",
            "donus",
            "olum",
            "panel_b",
            "panel_z",
            "dog_sap",
            "dog_n",
            "don_sap",
            "don_n",
            "son_sap",
            "son_n",
        ],
    )
    print("\n--- panel dusmus mu (z<-1) vs dusmemis, olay dususleri ---")
    for ad, m in [
        ("PANEL DUSMUS (z<-1)", S["panel_z"] < -1),
        ("PANEL NORMAL (z>=-1)", S["panel_z"] >= -1),
    ]:
        x = S[m]
        wd = x["dog_n"].sum()
        wr = x["don_n"].sum()
        print(
            f"{ad:<24} gun={len(x):>3}  dogum sap "
            f"{float((x['dog_sap'] * x['dog_n']).sum() / wd) if wd else float('nan'):>+8.4f} (n={int(wd):>5}) "
            f"| donus sap {float((x['don_sap'] * x['don_n']).sum() / wr) if wr else float('nan'):>+8.4f} (n={int(wr):>5})"
        )
    kor = S[["panel_z", "dog_sap"]].dropna()
    print(
        f"kor(panel_z, dogum sap) = {np.corrcoef(kor['panel_z'], kor['dog_sap'])[0, 1]:+.3f} (n={len(kor)})"
    )
    kor2 = S[["panel_z", "don_sap"]].dropna()
    print(
        f"kor(panel_z, donus sap) = {np.corrcoef(kor2['panel_z'], kor2['don_sap'])[0, 1]:+.3f} (n={len(kor2)})"
    )
    kor3 = S[["dogum", "dog_sap"]].dropna()
    print(
        f"kor(parti buyuklugu, dogum sap) = {np.corrcoef(np.log(kor3['dogum']), kor3['dog_sap'])[0, 1]:+.3f} (n={len(kor3)})"
    )

    # ---- 2026-03-26 donusleri, gevsetilmis ileri ref ----
    print("\n" + "=" * 118)
    print("2026-03-26 (train'in EN BUYUK olay gunu, 05-11'in dogrudan analogu)")
    print("=" * 118)
    for t in [
        pd.Timestamp("2026-03-26"),
        pd.Timestamp("2025-06-17"),
        pd.Timestamp("2025-07-28"),
        pd.Timestamp("2025-11-25"),
        pd.Timestamp("2025-09-10"),
    ]:
        xr = don.loc[don["tarih"] == t]
        xd = dog.loc[dog["tarih"] == t]
        print(
            f"{str(t.date())}: panel b {float(b.get(t, np.nan)):+.4f} "
            f"(z {float(b.get(t, np.nan)) / sd:+.2f})  "
            f"donus n={len(xr)} olculen={xr['sap_i'].notna().sum()} "
            f"sap {xr['sap_i'].mean():+.4f} | geri-ref sap {xr['sap_g'].mean():+.4f} "
            f"(n={xr['sap_g'].notna().sum()})  "
            f"dogum n={len(xd)} sap {xd['sap_i'].mean():+.4f}"
        )

    # ---- TESTTE 2026-05-11 var mi olum? ----
    print("\n" + "=" * 118)
    print("TEST 2026-05-11 OLAY PROFILI")
    print("=" * 118)
    e = te.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    eg = e.groupby("tanim", observed=True)
    e["ilk"] = eg["tarih"].transform("min")
    e["son"] = eg["tarih"].transform("max")
    e["bosluk"] = ((e["tarih"] - eg["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
    ei = e.groupby("tanim", observed=True)["ilk"].first().value_counts()
    es = e.groupby("tanim", observed=True)["son"].first()
    es = es[es < pd.Timestamp("2026-07-31")].value_counts()
    ed = e.loc[e["bosluk"] > 0, "tarih"].value_counts()
    print(f"{'tarih':<12} {'ilk gun':>8} {'son gun':>8} {'ic donus':>9}")
    for t in sorted(set(ei[ei >= 20].index) | set(es[es >= 5].index) | set(ed[ed >= 20].index)):
        print(
            f"{str(t.date()):<12} {int(ei.get(t, 0)):>8} {int(es.get(t, 0)):>8} {int(ed.get(t, 0)):>9}"
        )
    print(
        "\ntest 2026-05-11: hicbir trafo BITMIYOR ve ic-donus yok -> "
        "olum/kesinti degil, KATILIM olayi."
    )
    # test ic bosluk tarihleri
    print(
        f"test ic-donus tarihlerinin ilk 10'u: "
        f"{ {str(k.date()): int(v) for k, v in ed.head(10).items()} }"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
