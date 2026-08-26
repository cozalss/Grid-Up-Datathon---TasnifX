"""EKSEN 2 KESIN -- PANEL GUN ETKISI CIKARILMIS olay olcumu.

DUZELTILEN IKINCI HATA
----------------------
Yerel referans (sonraki 7 gun) panel gun etkisini KALDIRMAZ. Olay gunu
tesaduefen sicak/soguk bir gunse, olculen dusus o kadar kayar. Ornek:
2025-07-28'de panel gun etkisi +0,4722 (z=+2,50) -- o gun butun sehir
yuksekti. Orada olculen "dusus yok" (-0,012) aslinda -0,012-0,472 = -0,48'lik
gercek bir dususu gizliyordu.

DOGRU OLCUM:
    sap = (r_olay - ort r_ref) - (b_olay - ort b_ref)
burada b, o gun OLAYSIZ (yas>2, kalan>2, bosluksuz) trafolardan, trafo etkisi
cikarilarak hesaplanan panel gun etkisidir.

Ayni duzeltme TEST tarafinda v55 tahminleri icin de yapilir (v55'in KENDI
panel gun etkisiyle). Boylece D_gercek ve D_v55 ayni tanima sahip olur.
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
PENC = 7


def ozet(x, ad, gen=50):
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


def hazirla(df, deger):
    d = df.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("tanim", observed=True)
    d["ilk"] = g["tarih"].transform("min")
    d["son"] = g["tarih"].transform("max")
    d["yas"] = (d["tarih"] - d["ilk"]).dt.days
    d["kalan"] = (d["son"] - d["tarih"]).dt.days
    d["bosluk"] = ((d["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
    # PANEL GUN ETKISI -- olaysiz satirlardan, trafo etkisi cikarilmis
    ol = d[(d["yas"] > 2) & (d["kalan"] > 2) & (d["bosluk"] == 0)].copy()
    ol["c"] = ol[deger] - ol.groupby("tanim", observed=True)[deger].transform("mean")
    b = ol.groupby("tarih")["c"].mean()
    b = b - b.mean()
    d["b"] = d["tarih"].map(b)
    # panel etkisi CIKARILMIS deger
    d["rd"] = d[deger] - d["b"]
    return d, b


def yerel(d, alan="rd", pencere=PENC, min_n=3):
    n = len(d)
    tan = d["tanim"].to_numpy()
    v = d[alan].to_numpy("float64")
    bs = d["bosluk"].to_numpy()
    ile = np.full(n, np.nan)
    ger = np.full(n, np.nan)
    for i in range(n):
        t = tan[i]
        acc, k = 0.0, 0
        for j in range(i + 1, min(i + 1 + pencere, n)):
            if tan[j] != t or bs[j] > 0:
                break
            if not np.isnan(v[j]):
                acc += v[j]
                k += 1
        if k >= min_n:
            ile[i] = acc / k
        acc, k = 0.0, 0
        for j in range(i - 1, max(i - 1 - pencere, -1), -1):
            if tan[j] != t:
                break
            if j + 1 != i and bs[j + 1] > 0:
                break
            if not np.isnan(v[j]):
                acc += v[j]
                k += 1
        if k >= min_n:
            ger[i] = acc / k
    d = d.copy()
    d["sap_i"] = v - ile
    d["sap_g"] = v - ger
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

    T, bT = hazirla(tr[["tanim", "tarih", "r", "tuketim"]], "r")
    T = yerel(T)
    E, bE = hazirla(te[["tanim", "tarih", "r"]], "r")
    E = yerel(E)
    E["tr_son"] = E["tanim"].map(tr.groupby("tanim", observed=True)["tarih"].max())
    E["yeni"] = E["tr_son"].isna()
    print(f"train panel gun std {bT.std():.4f} | test(v55) panel gun std {bE.std():.4f}")

    # ---- KONTROL: olaysiz satirlarda panel-duzeltilmis yerel sapma 0 mi ----
    nor = T[(T["yas"] > 3) & (T["kalan"] > 3) & (T["bosluk"] == 0)]
    print(
        ozet(nor["sap_i"].sample(50000, random_state=0), "KONTROL olaysiz satir sap_i (0 olmali)")
    )
    print(
        ozet(nor["sap_g"].sample(50000, random_state=0), "KONTROL olaysiz satir sap_g (0 olmali)")
    )

    # ================= TRAIN OLAYLARI =================
    T_dog = T[(T["yas"] == 0) & (T["ilk"] > TR_BAS)].copy()
    T_dog["parti_n"] = T_dog["ilk"].map(T_dog["ilk"].value_counts())
    T_don = T[T["bosluk"] > 0].copy()
    T_don["parti_n"] = T_don["tarih"].map(T_don["tarih"].value_counts())
    T_son = T[(T["kalan"] == 0) & (T["son"] < TR_SON)].copy()
    olu = (
        T[T["kalan"].between(1, 14)]
        .groupby("tanim", observed=True)["tuketim"]
        .apply(lambda s: float((s <= 0).mean()))
    )
    T_son["olu"] = T_son["tanim"].map(olu).fillna(0.0)

    print("\n" + "=" * 112)
    print("A) TRAIN -- PANEL DUZELTILMIS olay dususleri")
    print("=" * 112)
    print(ozet(T_dog["sap_i"], "DOGUM gun-0"))
    pk = pd.cut(T_dog["parti_n"], [0, 4, 19, 99, 3000], labels=["1-4", "5-19", "20-99", "100+"])
    for k, x in T_dog.groupby(pk, observed=True):
        print(ozet(x["sap_i"], f"   parti {k}"))
    ikd = T_dog[(T_dog["ilk"] >= IK0) & (T_dog["ilk"] <= IK1)]
    print(ozet(ikd["sap_i"], "   IKIZ PENCERE (2025-04..07)"))
    print(ozet(ikd[ikd["parti_n"] >= 100]["sap_i"], "   IKIZ PENCERE parti 100+"))
    print(ozet(ikd[ikd["parti_n"] < 100]["sap_i"], "   IKIZ PENCERE parti <100"))
    print("\n   -- 90+ partiler tek tek (panel duzeltmesi ONCE/SONRA) --")
    print(f"   {'tarih':<12} {'n':>5} {'ham sap':>9} {'panel b-bref':>13} {'DUZELT sap':>11}")
    for t, x in T_dog[T_dog["parti_n"] >= 90].groupby("ilk"):
        ham = float((x["r"] - (x["r"] - x["sap_i"] + x["b"] - x["b"])).mean()) if False else np.nan
        d_i = float(x["sap_i"].mean())
        print(
            f"   {str(t.date()):<12} {int(x['sap_i'].notna().sum()):>5} "
            f"{'':>9} {float(x['b'].mean()):>+13.4f} {d_i:>+11.4f}"
        )

    print()
    print(ozet(T_don["sap_i"], "DONUS gunu"))
    bk = pd.cut(T_don["bosluk"], [0, 7, 60, 3000], labels=["1-7g", "8-60g", "60+g"])
    T_don["_pk"] = np.where(T_don["parti_n"] >= 20, "TOPLU", "TEKIL")
    for (a, b_), x in T_don.groupby([bk, "_pk"], observed=True):
        print(ozet(x["sap_i"], f"   bosluk {a} {b_}"))
    print(
        ozet(T_don[(T_don["tarih"] >= IK0) & (T_don["tarih"] <= IK1)]["sap_i"], "   IKIZ PENCERE")
    )

    print()
    print(ozet(T_son["sap_g"], "SON gun"))
    print(ozet(T_son[T_son["olu"] < 0.5]["sap_g"], "   CANLI"))
    iks = T_son[(T_son["son"] >= IK0) & (T_son["son"] <= IK1) & (T_son["olu"] < 0.5)]
    print(ozet(iks["sap_g"], "   IKIZ PENCERE CANLI"))

    # ================= TEST =================
    print("\n" + "=" * 112)
    print("B) TEST -- v55'in AYNI (panel duzeltilmis) tasarimla olculen sapmasi")
    print("=" * 112)
    E_dog = E[E["yas"] == 0].copy()
    E_dog["parti_n"] = E_dog["ilk"].map(E_dog["ilk"].value_counts())
    E_dog["gb"] = np.where(E_dog["yeni"], np.nan, (E_dog["tarih"] - E_dog["tr_son"]).dt.days - 1)
    E_don = E[E["bosluk"] > 0].copy()
    E_son = E[(E["kalan"] == 0) & (E["son"] < TE_SON)].copy()
    print(
        f"   v55 panel gun etkisi 2026-05-11: {float(bE.get(pd.Timestamp('2026-05-11'), np.nan)):+.4f}"
    )

    def ekle(ad, alt, alan):
        x = alt[alan].dropna()
        print(
            f"{ad:<50} satir={len(alt):>6,} olculen={len(x):>6,} "
            f"v55 {x.mean():>+8.4f} sh {x.std(ddof=1) / np.sqrt(len(x)):>6.4f}"
        )
        return len(alt), float(x.mean())

    n1, v1 = ekle(
        "YENI gun-0, parti 100+ (05-11)", E_dog[E_dog["yeni"] & (E_dog["parti_n"] >= 100)], "sap_i"
    )
    n2, v2 = ekle(
        "YENI gun-0, parti <100", E_dog[E_dog["yeni"] & (E_dog["parti_n"] < 100)], "sap_i"
    )
    n3, v3 = ekle(
        "ESKI gun-0, bosluk 1-60g", E_dog[(~E_dog["yeni"]) & E_dog["gb"].between(1, 60)], "sap_i"
    )
    n4, v4 = ekle("ESKI gun-0, bosluk 60+g", E_dog[(~E_dog["yeni"]) & (E_dog["gb"] > 60)], "sap_i")
    n5, v5 = ekle("IC BOSLUK donusu", E_don, "sap_i")
    n6, v6 = ekle("SON gun (07-31 disi)", E_son, "sap_g")
    _, vk = ekle(
        "ESKI gun-0 KESINTISIZ (kontrol, ~0)", E_dog[(~E_dog["yeni"]) & (E_dog["gb"] <= 0)], "sap_i"
    )

    # ================= dMSE =================
    print("\n" + "=" * 112)
    print("C) dMSE DEFTERI")
    print("=" * 112)
    D = {
        "dog100": float(T_dog[T_dog["parti_n"] >= 100]["sap_i"].mean()),
        "dog100_ikiz": float(ikd[ikd["parti_n"] >= 100]["sap_i"].mean()),
        "dogkucuk": float(T_dog[T_dog["parti_n"] < 100]["sap_i"].mean()),
        "don_kisa": float(T_don[T_don["bosluk"].between(1, 60)]["sap_i"].mean()),
        "don_uzun": float(T_don[T_don["bosluk"] > 60]["sap_i"].mean()),
        "don_hepsi": float(T_don["sap_i"].mean()),
        "don_ikiz": float(T_don[(T_don["tarih"] >= IK0) & (T_don["tarih"] <= IK1)]["sap_i"].mean()),
        "son_canli": float(T_son[T_son["olu"] < 0.5]["sap_g"].mean()),
        "son_ikiz": float(iks["sap_g"].mean()),
    }
    for k, v in D.items():
        print(f"   D[{k}] = {v:+.4f}")

    senaryolar = {
        "MERKEZ (tam ornek)": [
            ("YENI gun-0 parti 100+", n1, D["dog100"], v1),
            ("YENI gun-0 parti <100", n2, D["dogkucuk"], v2),
            ("ESKI gun-0 bosluk 1-60g", n3, D["don_kisa"], v3),
            ("ESKI gun-0 bosluk 60+g", n4, D["don_uzun"], v4),
            ("IC BOSLUK donusu", n5, D["don_hepsi"], v5),
            ("SON gun", n6, D["son_canli"], v6),
        ],
        "IHTIYATLI (mevsimsel ikiz)": [
            ("YENI gun-0 parti 100+", n1, D["dog100_ikiz"], v1),
            ("YENI gun-0 parti <100", n2, D["dogkucuk"], v2),
            ("ESKI gun-0 bosluk 1-60g", n3, D["don_ikiz"], v3),
            ("ESKI gun-0 bosluk 60+g", n4, D["don_ikiz"], v4),
            ("IC BOSLUK donusu", n5, D["don_ikiz"], v5),
            ("SON gun", n6, D["son_ikiz"], v6),
        ],
    }
    for ad, kayit in senaryolar.items():
        for s in (1.0, 0.6):
            tot = 0.0
            print(f"\n--- {ad}  shrink s={s:.1f} ---")
            print(
                f"{'grup':<26} {'n':>6} {'p':>9} {'D_ger':>8} {'D_v55':>8} {'b':>8} "
                f"{'kayma':>8} {'dMSE':>11}"
            )
            for g_, n, dg, dv in kayit:
                p = n / N_TEST
                b_ = dg - dv
                dm = p * (s * s - 2 * s) * b_ * b_
                tot += dm
                print(
                    f"{g_:<26} {n:>6,} {p:>9.6f} {dg:>+8.4f} {dv:>+8.4f} {b_:>+8.4f} "
                    f"{s * b_:>+8.4f} {dm:>11.6f}"
                )
            print(f"{'TOPLAM':<26} {'':>6} {'':>9} {'':>8} {'':>8} {'':>8} {'':>8} {tot:>11.6f}")
            print(
                f"   RMSLE {np.sqrt(MSE0):.5f} -> {np.sqrt(MSE0 + tot):.5f} "
                f"({np.sqrt(MSE0 + tot) - np.sqrt(MSE0):+.5f})"
            )

    # ---- RECETE dosyasi (MERKEZ, s=0.6) ----
    s = 0.6
    kay = pd.Series(0.0, index=E.index)
    tset = lambda q: set(q["tanim"])
    m1 = (E["yas"] == 0) & E["tanim"].isin(tset(E_dog[E_dog["yeni"] & (E_dog["parti_n"] >= 100)]))
    m2 = (E["yas"] == 0) & E["tanim"].isin(tset(E_dog[E_dog["yeni"] & (E_dog["parti_n"] < 100)]))
    m3 = (E["yas"] == 0) & E["tanim"].isin(
        tset(E_dog[(~E_dog["yeni"]) & E_dog["gb"].between(1, 60)])
    )
    m4 = (E["yas"] == 0) & E["tanim"].isin(tset(E_dog[(~E_dog["yeni"]) & (E_dog["gb"] > 60)]))
    m5 = E["bosluk"] > 0
    m6 = (E["kalan"] == 0) & (E["son"] < TE_SON)
    print("\n" + "=" * 112)
    print("D) RECETE (MERKEZ, s=0.6) -- log-uzayi additif kayma")
    print("=" * 112)
    for m, b_, ad in [
        (m1, D["dog100"] - v1, "YENI 100+"),
        (m2, D["dogkucuk"] - v2, "YENI <100"),
        (m3, D["don_kisa"] - v3, "ESKI 1-60g"),
        (m4, D["don_uzun"] - v4, "ESKI 60+g"),
        (m5, D["don_hepsi"] - v5, "IC BOSLUK"),
        (m6, D["son_canli"] - v6, "SON GUN"),
    ]:
        kay[m] = s * b_
        print(f"   {ad:<12} satir {int(m.sum()):>6,}  kayma {s * b_:>+8.4f}")
    print(f"   dokunulan satir {int((kay != 0).sum()):,} ({(kay != 0).mean() * 100:.3f}%)")
    E["kayma"] = kay
    out = E.loc[E["kayma"] != 0, ["tanim", "tarih", "kayma"]].copy()
    out["id"] = out["tanim"] + "_" + out["tarih"].dt.strftime("%Y-%m-%d")
    out[["id", "kayma"]].to_csv(KOK / "data/interim/eksen2_kayma.csv", index=False)
    print("   yazildi: data/interim/eksen2_kayma.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
