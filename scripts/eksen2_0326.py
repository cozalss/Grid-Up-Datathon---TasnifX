"""EKSEN 2d -- 2026-03-26 KOHORTU: 2026-05-11'in en yakin ve EN TAZE analogu.

2026-03-26'da 329 dogum + 193 donus var; 2026-03-27'de 509 olum. Testte
2026-05-11'de 1.326 dogum + 896 donus var ve o 896'nin 493'unun train son gunu
tam olarak 2026-03-27. Yani AYNI trafolar, ayni sistem olayi.

Bu betik:
  1. 2026-03-26 dogumlarini GERIYE degil ILERIYE (yas 1..5) referansla olcer.
  2. 2026-03-26 donuslerini GERIYE (bosluk oncesi 21 gun) referansla olcer.
     Geri referans, train sonunda da calisir ve modelin gordugu seyle ayni.
  3. Butun donusleri GERI referansla yeniden olcer (saglamlik).
  4. 03-26/03-27 iki gunluk trafolarin gun profilini cikarir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TR_BAS = pd.Timestamp("2025-01-01")
TR_SON = pd.Timestamp("2026-03-31")


def ozet(x, ad):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return f"{ad:<52} n={len(x):>5}  --"
    kk = []
    for K in (1, 5, 10, 25):
        if len(x) > K:
            v = x.drop(x.abs().sort_values(ascending=False).index[:K])
            kk.append(f"K{K}={v.mean():+.3f}")
    return (
        f"{ad:<52} n={len(x):>5,} ort {x.mean():>+8.4f} "
        f"sh {x.std(ddof=1) / np.sqrt(len(x)):>6.4f} med {x.median():>+8.4f}  ["
        + " ".join(kk)
        + "]"
    )


def main():
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    tr["r"] = np.log1p(tr["tuketim"].to_numpy("float64")) - np.log1p(tr["guc"].to_numpy("float64"))
    tr = tr.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    g = tr.groupby("tanim", observed=True)
    d = tr
    d["ilk_tarih"] = g["tarih"].transform("min")
    d["son_tarih"] = g["tarih"].transform("max")
    d["yas"] = (d["tarih"] - d["ilk_tarih"]).dt.days
    d["bosluk"] = ((d["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
    tanim_arr = d["tanim"].to_numpy()
    r_arr = d["r"].to_numpy()
    bos_arr = d["bosluk"].to_numpy()
    tar_arr = d["tarih"].to_numpy()
    tuk_arr = d["tuketim"].to_numpy()

    # ---------- 1) 2026-03-26 DOGUMLARI, ileri referans yas 1..5 ----------
    print("=" * 104)
    print("1) 2026-03-26 DOGUMLARI -- ileri referans (yas 1..5, en az 2 gun, arada bosluk yok)")
    print("=" * 104)
    ilk = d.groupby("tanim", observed=True)["ilk_tarih"].first()
    dog_say = ilk[ilk > TR_BAS].value_counts()

    def ileri_ref(i, bas, bit, min_n):
        t = tanim_arr[i]
        v = []
        for j in range(i + bas, i + bit + 1):
            if j >= len(r_arr) or tanim_arr[j] != t or bos_arr[j] > 0:
                break
            v.append(r_arr[j])
        return float(np.mean(v)) if len(v) >= min_n else np.nan

    dog_idx = np.flatnonzero(
        (d["yas"].to_numpy() == 0) & (d["ilk_tarih"].to_numpy() > np.datetime64(TR_BAS))
    )
    kayit = []
    for i in dog_idx:
        t = pd.Timestamp(tar_arr[i])
        kayit.append(
            (
                tanim_arr[i],
                t,
                int(dog_say.get(t, 0)),
                r_arr[i] - ileri_ref(i, 1, 5, 2),
                r_arr[i] - ileri_ref(i, 8, 28, 5),
                tuk_arr[i] <= 0,
            )
        )
    dg = pd.DataFrame(kayit, columns=["tanim", "tarih", "parti_n", "sap15", "sap828", "sifir"])
    print(ozet(dg["sap15"], "TUM dogumlar (ileri ref 1..5)"))
    print(ozet(dg["sap828"], "TUM dogumlar (ileri ref 8..28) -- karsilastirma"))
    print("\n-- parti buyuklugune gore (ileri ref 1..5, HER partiyi kapsar) --")
    pk = pd.cut(
        dg["parti_n"], [0, 4, 19, 99, 199, 2000], labels=["1-4", "5-19", "20-99", "100-199", "200+"]
    )
    for k, x in dg.groupby(pk, observed=True):
        print(ozet(x["sap15"], f"  parti {k}"))
    print("\n-- 80+ trafolu partiler tek tek --")
    print(f"{'tarih':<12} {'parti_n':>8} {'n_olculen':>10} {'sap15':>10} {'sh':>8} {'sifir%':>8}")
    for t, x in dg[dg["parti_n"] >= 80].groupby("tarih"):
        v = x["sap15"].dropna()
        if len(v) < 3:
            print(f"{str(t.date()):<12} {int(x['parti_n'].iloc[0]):>8} {len(v):>10} {'--':>10}")
            continue
        print(
            f"{str(t.date()):<12} {int(x['parti_n'].iloc[0]):>8} {len(v):>10} "
            f"{v.mean():>+10.4f} {v.std(ddof=1) / np.sqrt(len(v)):>8.4f} {x['sifir'].mean() * 100:>7.2f}%"
        )

    # ---------- 2) DONUSLER, GERI referans ----------
    print("\n" + "=" * 104)
    print("2) BOSLUK DONUSLERI -- GERI referans (bosluktan onceki 21 gun, en az 5)")
    print("   Geri referans train sonunda da calisir ve modelin gordugu gecmisle ayni.")
    print("=" * 104)

    def geri_ref(i, n=21, min_n=5):
        t = tanim_arr[i]
        v = []
        for j in range(i - 1, i - 1 - n, -1):
            if j < 0 or tanim_arr[j] != t or bos_arr[j + 1] > 0:
                break
            v.append(r_arr[j])
        return float(np.mean(v)) if len(v) >= min_n else np.nan

    don_idx = np.flatnonzero(bos_arr > 0)
    don_say = pd.Series(tar_arr[don_idx]).value_counts()
    kayit2 = []
    for i in don_idx:
        t = pd.Timestamp(tar_arr[i])
        gr = geri_ref(i)
        il = ileri_ref(i, 1, 21, 5)
        kayit2.append(
            (
                tanim_arr[i],
                t,
                bos_arr[i],
                int(don_say.get(t, 0)),
                r_arr[i] - gr,
                r_arr[i] - il,
                (ileri_ref(i, 1, 21, 5) - gr),
                tuk_arr[i] <= 0,
            )
        )
    bd = pd.DataFrame(
        kayit2,
        columns=[
            "tanim",
            "tarih",
            "bosluk",
            "parti_n",
            "geri_sap",
            "ileri_sap",
            "seviye_kayma",
            "sifir",
        ],
    )
    print(ozet(bd["geri_sap"], "TUM donusler (GERI ref)"))
    print(ozet(bd["ileri_sap"], "TUM donusler (ILERI ref)"))
    print(ozet(bd["seviye_kayma"], "bosluk sonrasi SEVIYE kaymasi (ileri-geri)"))
    print("\n-- parti buyuklugune gore (GERI ref) --")
    pk2 = pd.cut(bd["parti_n"], [0, 4, 19, 99, 3000], labels=["1-4", "5-19", "20-99", "100+"])
    for k, x in bd.groupby(pk2, observed=True):
        print(ozet(x["geri_sap"], f"  parti {k}"))
    print("\n-- 2026-05-11 kohortunun train ANALOGLARI --")
    for t in [
        pd.Timestamp("2026-03-26"),
        pd.Timestamp("2025-06-17"),
        pd.Timestamp("2026-02-03"),
        pd.Timestamp("2026-01-12"),
        pd.Timestamp("2026-02-19"),
        pd.Timestamp("2026-02-18"),
    ]:
        x = bd[bd["tarih"] == t]
        print(
            ozet(
                x["geri_sap"], f"  {str(t.date())} donusleri (bosluk ort {x['bosluk'].mean():.0f}g)"
            )
        )
    ik = bd[(bd["tarih"] >= "2025-04-01") & (bd["tarih"] <= "2025-07-31")]
    print(ozet(ik["geri_sap"], "  IKIZ PENCERE (2025-04..07)"))

    # ---------- 3) IKI GUNLUK TRAFOLAR (03-26 dogup 03-27 biten) ----------
    print("\n" + "=" * 104)
    print("3) 2026-03-26'da DOGUP 2026-03-27'de BITEN trafolar (testte 05-11'de donuyorlar)")
    print("=" * 104)
    son = d.groupby("tanim", observed=True)["son_tarih"].first()
    iki = set(ilk[(ilk == "2026-03-26")].index) & set(son[son == "2026-03-27"].index)
    print(f"iki gunluk trafo sayisi: {len(iki):,}")
    x = d[d["tanim"].isin(iki)]
    for t, s in x.groupby("tarih"):
        print(
            f"  {str(t.date())}  n={len(s):,}  ort r {s['r'].mean():+.4f}  "
            f"medyan r {s['r'].median():+.4f}  sifir% {(s['tuketim'] <= 0).mean() * 100:.2f}"
        )
    p = x.pivot_table(index="tanim", columns="tarih", values="r")
    if p.shape[1] == 2:
        fark = p.iloc[:, 1] - p.iloc[:, 0]
        print(ozet(fark, "  gun2 - gun1 farki (kismi ilk gun ise >0 olmali)"))
    # bu trafolarin testteki ilk gunu
    te_ilk = te.groupby("tanim", observed=True)["tarih"].min()
    var = te_ilk[te_ilk.index.isin(iki)]
    print(f"  bunlarin testteki ilk gunu: {var.value_counts().head(3).to_dict()}")
    # karsilastirma: ayni gun dogup DEVAM eden trafolar
    devam = set(ilk[(ilk == "2026-03-26")].index) - iki
    y = d[d["tanim"].isin(devam)]
    print(f"\n2026-03-26'da dogup DEVAM eden trafo: {len(devam):,}")
    for t, s in y.groupby("tarih"):
        print(
            f"  {str(t.date())}  n={len(s):,}  ort r {s['r'].mean():+.4f}  "
            f"medyan r {s['r'].median():+.4f}  sifir% {(s['tuketim'] <= 0).mean() * 100:.2f}"
        )

    # ---------- 4) 2026-03-27 OLUMLERI, geri referans ----------
    print("\n" + "=" * 104)
    print("4) 2026-03-27 OLUMLERI (509) -- geri referans")
    print("=" * 104)
    olu_idx = np.flatnonzero(
        (d["tarih"].to_numpy() == np.datetime64("2026-03-27"))
        & (d["son_tarih"].to_numpy() == np.datetime64("2026-03-27"))
    )
    kayit3 = []
    for i in olu_idx:
        kayit3.append(
            (
                tanim_arr[i],
                r_arr[i] - geri_ref(i, 21, 5),
                r_arr[i] - geri_ref(i, 21, 1),
                tuk_arr[i] <= 0,
            )
        )
    od = pd.DataFrame(kayit3, columns=["tanim", "sap5", "sap1", "sifir"])
    print(f"olculen olum satiri: {len(od):,}  sifir% {od['sifir'].mean() * 100:.2f}")
    print(ozet(od["sap5"], "  son gun sapmasi (geri ref, >=5 gun gecmis)"))
    print(ozet(od["sap1"], "  son gun sapmasi (geri ref, >=1 gun gecmis)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
