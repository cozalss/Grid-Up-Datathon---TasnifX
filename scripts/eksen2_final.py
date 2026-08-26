"""EKSEN 2 FINAL -- YEREL referansla olcum, train ve testte AYNI tasarim.

ONCEKI OLCUMDEKI HATA
---------------------
v55'in "uyguladigi sapma" trafonun kendi yas 8..28 gunlerine gore olculuyordu.
Testte bu referans olay gununden 3 AY uzakta olabiliyor (ornek: 07-07'de biten
bir trafonun yas 8..28'i NISAN); mevsimsel yukselis olay etkisiyle karisiyor.
Bu yuzden SON GUN'de v55 sapmasi +0,34 gorunuyordu -- model bir sey yapmiyor,
sadece temmuz nisandan yuksek.

DUZELTME: her olay icin YEREL referans (komsu 1..7 gun), train ve testte
BIRE BIR ayni tasarim. Boylece fark dogrudan karsilastirilabilir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TR_BAS = pd.Timestamp("2025-01-01")
TR_SON = pd.Timestamp("2026-03-31")
TE_SON = pd.Timestamp("2026-07-31")
N_TEST = 714688
MSE0 = 1.03207
IK0, IK1 = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")


def ozet(x, ad, gen=52):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return f"{ad:<{gen}} n={len(x):>5}  --"
    kk = []
    for K in (1, 5, 10, 25):
        if len(x) > K:
            v = x.drop(x.abs().sort_values(ascending=False).index[:K])
            kk.append(f"K{K}={v.mean():+.3f}")
    return (
        f"{ad:<{gen}} n={len(x):>5,} ort {x.mean():>+8.4f} "
        f"sh {x.std(ddof=1) / np.sqrt(len(x)):>6.4f} med {x.median():>+8.4f}  ["
        + " ".join(kk)
        + "]"
    )


def panel(df, deger):
    """Sirali panelden olay olculeri cikarir. df: tanim,tarih,deger sirali."""
    d = df.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("tanim", observed=True)
    d["ilk"] = g["tarih"].transform("min")
    d["son"] = g["tarih"].transform("max")
    d["yas"] = (d["tarih"] - d["ilk"]).dt.days
    d["kalan"] = (d["son"] - d["tarih"]).dt.days
    d["bosluk"] = ((d["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
    return d


def yerel_olc(d, deger, pencere=7, min_n=3):
    """Her satir icin: ileri yerel ref (sonraki `pencere` bitisik gun) ve
    geri yerel ref (onceki `pencere` bitisik gun). Bosluk gorunce durur."""
    n = len(d)
    tan = d["tanim"].to_numpy()
    v = d[deger].to_numpy("float64")
    bos = d["bosluk"].to_numpy()
    ileri = np.full(n, np.nan)
    geri = np.full(n, np.nan)
    for i in range(n):
        t = tan[i]
        acc, k = 0.0, 0
        for j in range(i + 1, min(i + 1 + pencere, n)):
            if tan[j] != t or bos[j] > 0:
                break
            acc += v[j]
            k += 1
        if k >= min_n:
            ileri[i] = acc / k
        acc, k = 0.0, 0
        for j in range(i - 1, max(i - 1 - pencere, -1), -1):
            if tan[j] != t:
                break
            if j + 1 != i and bos[j + 1] > 0:
                break
            acc += v[j]
            k += 1
        if k >= min_n:
            geri[i] = acc / k
    d = d.copy()
    d["ref_ileri"] = ileri
    d["ref_geri"] = geri
    d["sap_ileri"] = v - ileri
    d["sap_geri"] = v - geri
    return d


def main():
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    sub = pd.read_csv(KOK / "submissions/tuketim_v55_gunolcek.csv", encoding="utf-8")
    tr["r"] = np.log1p(tr["tuketim"].to_numpy("float64")) - np.log1p(tr["guc"].to_numpy("float64"))
    te = te.merge(sub, on="id", how="left", validate="one_to_one")
    te["r"] = np.log1p(te["tuketim"].to_numpy("float64")) - np.log1p(te["guc"].to_numpy("float64"))

    T = yerel_olc(panel(tr[["tanim", "tarih", "r", "tuketim"]], "r"), "r")
    E = yerel_olc(panel(te[["tanim", "tarih", "r"]], "r"), "r")
    tr_son_map = tr.groupby("tanim", observed=True)["tarih"].max()
    E["tr_son"] = E["tanim"].map(tr_son_map)
    E["yeni"] = E["tr_son"].isna()

    # parti buyuklukleri
    T_dog = T[(T["yas"] == 0) & (T["ilk"] > TR_BAS)].copy()
    T_dog["parti_n"] = T_dog["ilk"].map(T_dog["ilk"].value_counts())
    T_don = T[T["bosluk"] > 0].copy()
    T_don["parti_n"] = T_don["tarih"].map(T_don["tarih"].value_counts())
    T_son = T[(T["kalan"] == 0) & (T["son"] < TR_SON)].copy()
    T_son["parti_n"] = T_son["son"].map(T_son["son"].value_counts())
    # son gun icin: onceki 7 gunde sifir orani (olu trafolari ayirmak icin)
    olu = (
        T[(T["kalan"].between(1, 14))]
        .groupby("tanim", observed=True)["tuketim"]
        .apply(lambda s: float((s <= 0).mean()))
    )
    T_son["olu"] = T_son["tanim"].map(olu).fillna(0.0)

    print("=" * 108)
    print("A) TRAIN -- YEREL referansla (komsu 7 gun, en az 3) olay dususleri")
    print("=" * 108)
    print(ozet(T_dog["sap_ileri"], "DOGUM gun-0 (ileri ref)"))
    pk = pd.cut(T_dog["parti_n"], [0, 4, 19, 99, 3000], labels=["1-4", "5-19", "20-99", "100+"])
    for k, x in T_dog.groupby(pk, observed=True):
        print(ozet(x["sap_ileri"], f"   parti {k}"))
    print(
        ozet(T_dog[(T_dog["ilk"] >= IK0) & (T_dog["ilk"] <= IK1)]["sap_ileri"], "   IKIZ PENCERE")
    )
    for kk, lb in [
        ((T_dog["parti_n"] >= 100), "parti 100+"),
        ((T_dog["parti_n"] < 100), "parti <100"),
    ]:
        s = T_dog[kk & (T_dog["ilk"] >= IK0) & (T_dog["ilk"] <= IK1)]
        print(ozet(s["sap_ileri"], f"   IKIZ PENCERE {lb}"))

    print()
    print(ozet(T_don["sap_ileri"], "DONUS gunu (ileri ref)"))
    print(ozet(T_don["sap_geri"], "DONUS gunu (GERI ref = bosluk oncesi seviye)"))
    bk = pd.cut(T_don["bosluk"], [0, 7, 60, 3000], labels=["1-7g", "8-60g", "60+g"])
    pk2 = np.where(T_don["parti_n"] >= 20, "TOPLU", "TEKIL")
    for (a, b), x in T_don.groupby([bk, pk2], observed=True):
        print(ozet(x["sap_ileri"], f"   bosluk {a} {b}"))
    print(
        ozet(
            T_don[(T_don["tarih"] >= IK0) & (T_don["tarih"] <= IK1)]["sap_ileri"], "   IKIZ PENCERE"
        )
    )

    print()
    print(ozet(T_son["sap_geri"], "SON gun (geri ref)"))
    print(ozet(T_son[T_son["olu"] < 0.5]["sap_geri"], "   CANLI trafolar"))
    print(ozet(T_son[T_son["olu"] >= 0.5]["sap_geri"], "   OLU trafolar"))
    canli = T_son[T_son["olu"] < 0.5].copy()
    canli["pk3"] = np.where(canli["parti_n"] >= 20, "TOPLU", "TEKIL")
    for k, x in canli.groupby("pk3", observed=True):
        print(ozet(x["sap_geri"], f"   CANLI {k} olum tarihi"))
    ik = T_son[(T_son["son"] >= IK0) & (T_son["son"] <= IK1) & (T_son["olu"] < 0.5)]
    print(ozet(ik["sap_geri"], "   IKIZ PENCERE CANLI"))

    # kontrol: olay OLMAYAN satirlarda yerel sapma ~0 mi?
    normal = T[(T["yas"] > 3) & (T["kalan"] > 3) & (T["bosluk"] == 0)]
    print(
        ozet(
            normal["sap_ileri"].sample(50000, random_state=0),
            "KONTROL normal satir (ileri ref) ~0 olmali",
        )
    )
    print(
        ozet(
            normal["sap_geri"].sample(50000, random_state=0),
            "KONTROL normal satir (geri ref) ~0 olmali",
        )
    )

    print("\n" + "=" * 108)
    print("B) TEST -- v55'in AYNI TASARIMLA olculen sapmasi")
    print("=" * 108)
    E_dog = E[E["yas"] == 0].copy()
    E_dog["parti_n"] = E_dog["ilk"].map(E_dog["ilk"].value_counts())
    E_dog["gercek_bosluk"] = np.where(
        E_dog["yeni"], np.nan, (E_dog["tarih"] - E_dog["tr_son"]).dt.days - 1
    )
    E_don = E[E["bosluk"] > 0].copy()
    E_son = E[(E["kalan"] == 0) & (E["son"] < TE_SON)].copy()

    gruplar = []

    def ekle(ad, alt, alan, n_over=None):
        x = alt[alan].dropna()
        n = len(alt) if n_over is None else n_over
        mu = float(x.mean()) if len(x) else np.nan
        sh = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan
        print(f"{ad:<52} satir={n:>6,} olculen={len(x):>6,} v55sap {mu:>+8.4f} sh {sh:>6.4f}")
        gruplar.append((ad, n, mu))
        return mu

    v_yeni_b = ekle(
        "YENI trafo gun-0, parti 100+",
        E_dog[E_dog["yeni"] & (E_dog["parti_n"] >= 100)],
        "sap_ileri",
    )
    v_yeni_k = ekle(
        "YENI trafo gun-0, parti <100", E_dog[E_dog["yeni"] & (E_dog["parti_n"] < 100)], "sap_ileri"
    )
    v_esk_kes = ekle(
        "ESKI trafo gun-0, KESINTISIZ",
        E_dog[(~E_dog["yeni"]) & (E_dog["gercek_bosluk"] <= 0)],
        "sap_ileri",
    )
    v_esk_k = ekle(
        "ESKI trafo gun-0, bosluk 1-60g",
        E_dog[(~E_dog["yeni"]) & E_dog["gercek_bosluk"].between(1, 60)],
        "sap_ileri",
    )
    v_esk_u = ekle(
        "ESKI trafo gun-0, bosluk 60+g",
        E_dog[(~E_dog["yeni"]) & (E_dog["gercek_bosluk"] > 60)],
        "sap_ileri",
    )
    v_ic = ekle("IC BOSLUK donusu", E_don, "sap_ileri")
    v_sg = ekle("SON gun (07-31 disi)", E_son, "sap_geri")

    # test son-gun trafolarinin CANLI mi olu mu oldugu (v55 tahmini uzerinden)
    e_olu = E[E["kalan"].between(1, 14)].groupby("tanim", observed=True)["r"].mean()
    E_son["v_seviye"] = E_son["tanim"].map(e_olu)
    print(
        f"   son-gun trafolarinin v55 seviyesi: medyan r {E_son['v_seviye'].median():+.3f}"
        f"  (tum test medyani {E['r'].median():+.3f})"
    )
    print(f"   son-gun tarihleri: {E_son['son'].dt.month.value_counts().sort_index().to_dict()}")

    print("\n" + "=" * 108)
    print("C) dMSE DEFTERI  (b = D_gercek - D_v55 ; dMSE = p*(s^2-2s)*b^2)")
    print("=" * 108)
    D = {}
    D["dogum100"] = float(T_dog[T_dog["parti_n"] >= 100]["sap_ileri"].mean())
    D["dogumkucuk"] = float(T_dog[T_dog["parti_n"] < 100]["sap_ileri"].mean())
    D["don_kisa"] = float(T_don[T_don["bosluk"].between(1, 60)]["sap_ileri"].mean())
    D["don_uzun"] = float(T_don[T_don["bosluk"] > 60]["sap_ileri"].mean())
    D["don_hepsi"] = float(T_don["sap_ileri"].mean())
    D["son_canli"] = float(T_son[T_son["olu"] < 0.5]["sap_geri"].mean())
    D["son_ikiz"] = float(ik["sap_geri"].mean())
    for k, v in D.items():
        print(f"   D[{k}] = {v:+.4f}")

    n_yeni_b = int((E_dog["yeni"] & (E_dog["parti_n"] >= 100)).sum())
    n_yeni_k = int((E_dog["yeni"] & (E_dog["parti_n"] < 100)).sum())
    n_esk_k = int(((~E_dog["yeni"]) & E_dog["gercek_bosluk"].between(1, 60)).sum())
    n_esk_u = int(((~E_dog["yeni"]) & (E_dog["gercek_bosluk"] > 60)).sum())
    n_ic, n_sg = len(E_don), len(E_son)

    kayit = [
        ("YENI gun-0 parti 100+", n_yeni_b, D["dogum100"], v_yeni_b),
        ("YENI gun-0 parti <100", n_yeni_k, D["dogumkucuk"], v_yeni_k),
        ("ESKI gun-0 bosluk 1-60g", n_esk_k, D["don_kisa"], v_esk_k),
        ("ESKI gun-0 bosluk 60+g", n_esk_u, D["don_uzun"], v_esk_u),
        ("IC BOSLUK donusu", n_ic, D["don_hepsi"], v_ic),
        ("SON gun (ihtiyatli: ikiz)", n_sg, D["son_ikiz"], v_sg),
        ("SON gun (tam: canli)", 0, D["son_canli"], v_sg),
    ]
    for s in (1.0, 0.6, 0.5):
        tot = 0.0
        print(f"\n--- shrink s={s:.1f} ---")
        print(
            f"{'grup':<28} {'n':>7} {'p':>9} {'D_ger':>8} {'D_v55':>8} {'b':>8} "
            f"{'kayma':>8} {'dMSE':>11}"
        )
        for ad, n, dg, dv in kayit:
            if n == 0:
                continue
            p = n / N_TEST
            b = dg - dv
            dm = p * (s * s - 2 * s) * b * b
            tot += dm
            print(
                f"{ad:<28} {n:>7,} {p:>9.6f} {dg:>+8.4f} {dv:>+8.4f} {b:>+8.4f} "
                f"{s * b:>+8.4f} {dm:>11.6f}"
            )
        print(f"{'TOPLAM':<28} {'':>7} {'':>9} {'':>8} {'':>8} {'':>8} {'':>8} {tot:>11.6f}")
        print(
            f"   RMSLE {np.sqrt(MSE0):.5f} -> {np.sqrt(MSE0 + tot):.5f} "
            f"({np.sqrt(MSE0 + tot) - np.sqrt(MSE0):+.5f})"
        )

    # ---- RECETE: uygulanacak kayma vektoru ----
    print("\n" + "=" * 108)
    print("D) RECETE -- id bazinda uygulanacak log-uzayi kaymasi (s=0.6)")
    print("=" * 108)
    s = 0.6
    kayma = pd.Series(0.0, index=E.index)
    tanim_tarih = E.set_index(["tanim", "tarih"]).index

    def isaretle(mask, b):
        kayma[mask] = s * b

    m_yeni_b = (
        (E["yas"] == 0)
        & E["yeni"]
        & E["tanim"].isin(set(E_dog.loc[E_dog["yeni"] & (E_dog["parti_n"] >= 100), "tanim"]))
    )
    m_yeni_k = (
        (E["yas"] == 0)
        & E["yeni"]
        & E["tanim"].isin(set(E_dog.loc[E_dog["yeni"] & (E_dog["parti_n"] < 100), "tanim"]))
    )
    m_esk_k = (E["yas"] == 0) & E["tanim"].isin(
        set(E_dog.loc[(~E_dog["yeni"]) & E_dog["gercek_bosluk"].between(1, 60), "tanim"])
    )
    m_esk_u = (E["yas"] == 0) & E["tanim"].isin(
        set(E_dog.loc[(~E_dog["yeni"]) & (E_dog["gercek_bosluk"] > 60), "tanim"])
    )
    m_ic = E["bosluk"] > 0
    m_sg = (E["kalan"] == 0) & (E["son"] < TE_SON)
    for m, b, ad in [
        (m_yeni_b, D["dogum100"] - v_yeni_b, "YENI 100+"),
        (m_yeni_k, D["dogumkucuk"] - v_yeni_k, "YENI <100"),
        (m_esk_k, D["don_kisa"] - v_esk_k, "ESKI 1-60g"),
        (m_esk_u, D["don_uzun"] - v_esk_u, "ESKI 60+g"),
        (m_ic, D["don_hepsi"] - v_ic, "IC BOSLUK"),
        (m_sg, D["son_ikiz"] - v_sg, "SON GUN"),
    ]:
        isaretle(m, b)
        print(f"   {ad:<12} satir {int(m.sum()):>6,}  kayma {s * b:+.4f}")
    print(
        f"   TOPLAM dokunulan satir: {int((kayma != 0).sum()):,} "
        f"({(kayma != 0).mean() * 100:.3f}% of test)"
    )
    E["kayma"] = kayma
    E[["tanim", "tarih", "kayma"]][E["kayma"] != 0].to_csv(
        KOK / "data/interim/eksen2_kayma.csv", index=False
    )
    print("   yazildi: data/interim/eksen2_kayma.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
