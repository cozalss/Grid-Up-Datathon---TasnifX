"""EKSEN 2b -- PARTI HETEROJENLIGI + TEST TARAFI SAYIMI + v55 uzerinde dMSE.

Sorular:
  1. TOPLU dogum partilerinin bir kisminda gun-0 dususu YOK (2025-07-28 +0,10,
     2025-09-10 +0,12). Bu partiler ayni zamanda TOPLU OLUM tarihi mi?
     Yani "raporlama gocu" mu?
  2. TOPLU DONUS tarihleri (ayni gun >=20 trafo boslukten donuyor) var mi,
     ve orada donus gunu dusuk mu? 2026-05-11 kohortu icin belirleyici.
  3. Testte kac satir etkileniyor, v55 o satirlarda ne yaziyor, dMSE ne?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TR_BAS = pd.Timestamp("2025-01-01")
TR_SON = pd.Timestamp("2026-03-31")
TE_BAS = pd.Timestamp("2026-04-01")
TE_SON = pd.Timestamp("2026-07-31")
ESIK = 20


def yukle():
    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    tr["r"] = np.log1p(tr["tuketim"].to_numpy("float64")) - np.log1p(tr["guc"].to_numpy("float64"))
    tr = tr.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    te = te.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    return tr, te


def olaylar(tr):
    g = tr.groupby("tanim", observed=True)
    d = tr.copy()
    d["ilk_tarih"] = g["tarih"].transform("min")
    d["son_tarih"] = g["tarih"].transform("max")
    d["yas"] = (d["tarih"] - d["ilk_tarih"]).dt.days
    d["kalan"] = (d["son_tarih"] - d["tarih"]).dt.days
    onc = g["tarih"].shift(1)
    d["bosluk"] = ((d["tarih"] - onc).dt.days - 1.0).fillna(-1.0)
    snr = g["tarih"].shift(-1)
    d["ileri_bosluk"] = ((snr - d["tarih"]).dt.days - 1.0).fillna(-1.0)
    d["_sira"] = g.cumcount()
    return d


def ileri_ref(d, i, tanim_arr, r_arr, bosluk_arr, n=21, min_n=5):
    t = tanim_arr[i]
    vals = []
    for j in range(i + 1, i + 1 + n):
        if j >= len(r_arr) or tanim_arr[j] != t or bosluk_arr[j] > 0:
            break
        vals.append(r_arr[j])
    return float(np.mean(vals)) if len(vals) >= min_n else np.nan


def geri_ref(d, i, tanim_arr, r_arr, bosluk_arr, n=21, min_n=5):
    t = tanim_arr[i]
    vals = []
    for j in range(i - 1, i - 1 - n, -1):
        if j < 0 or tanim_arr[j] != t or bosluk_arr[j + 1] > 0:
            break
        vals.append(r_arr[j])
    return float(np.mean(vals)) if len(vals) >= min_n else np.nan


def sat(x, ad):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return f"{ad:<40} n=0"
    return (
        f"{ad:<40} n={len(x):>6,} ort {x.mean():+.4f} "
        f"sh {x.std(ddof=1) / np.sqrt(len(x)):.4f} medyan {x.median():+.4f}"
    )


def main():
    tr, te = yukle()
    d = olaylar(tr)
    tanim_arr = d["tanim"].to_numpy()
    r_arr = d["r"].to_numpy()
    bosluk_arr = d["bosluk"].to_numpy()
    tuk_arr = d["tuketim"].to_numpy()

    # ---- olay tarihi tipolojisi ----
    ilk = d.groupby("tanim", observed=True)["ilk_tarih"].first()
    son = d.groupby("tanim", observed=True)["son_tarih"].first()
    dogum_say = ilk[ilk > TR_BAS].value_counts()
    olum_say = son[son < TR_SON].value_counts()
    donus_say = d.loc[d["bosluk"] > 0, "tarih"].value_counts()
    cikis_say = d.loc[d["ileri_bosluk"] > 0, "tarih"].value_counts()

    print("=== 1) OLAY TARIHI TIPOLOJISI (>=20 olay olan gunler) ===")
    tarihler = sorted(
        set(dogum_say[dogum_say >= ESIK].index)
        | set(olum_say[olum_say >= ESIK].index)
        | set(donus_say[donus_say >= ESIK].index)
    )
    print(f"{'tarih':<12} {'dogum':>7} {'olum':>7} {'donus':>7} {'cikis':>7}")
    for t in tarihler:
        print(
            f"{str(t.date()):<12} {int(dogum_say.get(t, 0)):>7} {int(olum_say.get(t, 0)):>7} "
            f"{int(donus_say.get(t, 0)):>7} {int(cikis_say.get(t, 0)):>7}"
        )

    # ---- gun-0 dususu: SAF dogum partisi vs GOC partisi ----
    dd = d[d["tanim"].isin(ilk[ilk > TR_BAS].index)].copy()
    ref = dd[(dd["yas"] >= 8) & (dd["yas"] <= 28)].groupby("tanim", observed=True)["r"]
    ref = ref.mean()[ref.size() >= 5]
    dd["ref"] = dd["tanim"].map(ref)
    g0 = dd[(dd["yas"] == 0) & dd["ref"].notna()].copy()
    g0["sapma"] = g0["r"] - g0["ref"]
    # GOC tarihi = ayni gun hem >=20 dogum hem >=20 olum/cikis
    goc = {
        t
        for t in tarihler
        if dogum_say.get(t, 0) >= ESIK
        and (olum_say.get(t, 0) >= ESIK or cikis_say.get(t, 0) >= ESIK)
    }
    saf = {t for t in dogum_say[dogum_say >= ESIK].index if t not in goc}
    print(f"\nGOC partisi tarihleri (dogum+olum ayni gun): {sorted(str(x.date()) for x in goc)}")
    g0["sinif"] = np.where(
        g0["ilk_tarih"].isin(goc), "GOC", np.where(g0["ilk_tarih"].isin(saf), "SAF_PARTI", "TEKIL")
    )
    print()
    for s, x in g0.groupby("sinif"):
        print(sat(x["sapma"], f"gun-0 sapma [{s}]"))
        print(
            f"     -> trafo {x['tanim'].nunique():,}  sifir% {(x['tuketim'] <= 0).mean() * 100:.2f}"
            f"  K=10 atilinca {x['sapma'].drop(x['sapma'].abs().sort_values(ascending=False).index[:10]).mean():+.4f}"
        )

    # ---- 2) TOPLU DONUS vs TEKIL DONUS ----
    print("\n=== 2) BOSLUK DONUSU: TOPLU tarih vs TEKIL tarih ===")
    idx = np.flatnonzero(bosluk_arr > 0)
    rows = []
    for i in idx:
        rf = ileri_ref(d, i, tanim_arr, r_arr, bosluk_arr)
        if np.isnan(rf):
            continue
        rows.append(
            (tanim_arr[i], d["tarih"].iloc[i], bosluk_arr[i], r_arr[i] - rf, tuk_arr[i] <= 0)
        )
    bd = pd.DataFrame(rows, columns=["tanim", "tarih", "bosluk", "sapma", "sifir"])
    toplu_donus = set(donus_say[donus_say >= ESIK].index)
    bd["sinif"] = np.where(bd["tarih"].isin(toplu_donus), "TOPLU_DONUS", "TEKIL_DONUS")
    for s, x in bd.groupby("sinif"):
        print(sat(x["sapma"], f"donus gunu sapma [{s}]"))
        print(
            f"     -> sifir% {x['sifir'].mean() * 100:.2f}  "
            f"K=10 atilinca {x['sapma'].drop(x['sapma'].abs().sort_values(ascending=False).index[:10]).mean():+.4f}"
        )
    print("\n--- en yogun donus tarihleri ---")
    print(f"{'tarih':<12} {'n':>5} {'ort sapma':>11} {'sh':>8} {'ort bosluk':>11}")
    for t, x in bd[bd["tarih"].isin(toplu_donus)].groupby("tarih"):
        if len(x) < 5:
            continue
        print(
            f"{str(t.date()):<12} {len(x):>5} {x['sapma'].mean():>+11.4f} "
            f"{x['sapma'].std(ddof=1) / np.sqrt(len(x)):>8.4f} {x['bosluk'].mean():>11.1f}"
        )

    # ---- 3) SON GUN: olum tarihine gore ----
    print("\n=== 3) SON GUN: olum tarihine gore ===")
    ddo = d[d["tanim"].isin(son[son < TR_SON].index)].copy()
    rf2 = ddo[(ddo["kalan"] >= 8) & (ddo["kalan"] <= 28)].groupby("tanim", observed=True)["r"]
    rf2 = rf2.mean()[rf2.size() >= 5]
    ddo["ref"] = ddo["tanim"].map(rf2)
    s0 = ddo[(ddo["kalan"] == 0) & ddo["ref"].notna()].copy()
    s0["sapma"] = s0["r"] - s0["ref"]
    print(f"{'tarih':<12} {'n':>5} {'ort sapma':>11} {'sh':>8} {'sifir%':>8}")
    for t, x in s0.groupby("son_tarih"):
        if len(x) < 5:
            continue
        print(
            f"{str(t.date()):<12} {len(x):>5} {x['sapma'].mean():>+11.4f} "
            f"{x['sapma'].std(ddof=1) / np.sqrt(len(x)):>8.4f} {(x['tuketim'] <= 0).mean() * 100:>7.2f}%"
        )
    kucuk = s0[~s0["son_tarih"].isin(olum_say[olum_say >= ESIK].index)]
    print(sat(kucuk["sapma"], "TEKIL olum tarihleri toplam"))
    buyuk = s0[s0["son_tarih"].isin(olum_say[olum_say >= ESIK].index)]
    print(sat(buyuk["sapma"], "TOPLU olum tarihleri toplam"))

    # ---- 4) MEVSIMSEL IKIZ: 2025-04-01..07-31 penceresindeki olaylar ----
    print("\n=== 4) MEVSIMSEL IKIZ (olay tarihi 2025-04-01..2025-07-31) ===")
    ik0, ik1 = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")
    m = g0[(g0["ilk_tarih"] >= ik0) & (g0["ilk_tarih"] <= ik1)]
    print(sat(m["sapma"], "gun-0 sapma [ikiz pencere]"))
    for s, x in m.groupby("sinif"):
        print(sat(x["sapma"], f"   gun-0 [{s}] ikiz"))
    mb = bd[(bd["tarih"] >= ik0) & (bd["tarih"] <= ik1)]
    print(sat(mb["sapma"], "donus gunu sapma [ikiz pencere]"))
    ms = s0[(s0["son_tarih"] >= ik0) & (s0["son_tarih"] <= ik1)]
    print(sat(ms["sapma"], "son gun sapma [ikiz pencere]"))

    # ================= TEST TARAFI =================
    print("\n=== 5) TEST TARAFI SATIR SAYIMI ===")
    tr_tanim = set(tr["tanim"])
    tg = te.groupby("tanim", observed=True)
    t_ilk = tg["tarih"].min()
    t_son = tg["tarih"].max()
    t_gun = tg["tarih"].size()
    yeni = ~t_ilk.index.isin(tr_tanim)
    print(f"test trafo {len(t_ilk):,}  YENI {int(yeni.sum()):,}  ESKI {int((~yeni).sum()):,}")
    print("\ntest ILK gun dagilimi (ilk 8):")
    for t, n in t_ilk.value_counts().head(8).items():
        alt = t_ilk[t_ilk == t]
        ny = int((~alt.index.isin(tr_tanim)).sum())
        print(f"  {str(t.date())}  toplam {n:>5}  YENI {ny:>5}  ESKI {n - ny:>5}")
    print("\ntest SON gun dagilimi (ilk 8):")
    for t, n in t_son.value_counts().head(8).items():
        print(f"  {str(t.date())}  {n:,}")

    # test ic bosluklari
    te["_onc"] = tg["tarih"].shift(1)
    te["_bosluk"] = ((te["tarih"] - te["_onc"]).dt.days - 1.0).fillna(-1.0)
    te_donus = te[te["_bosluk"] > 0]
    print(
        f"\ntest ic bosluk donus satiri: {len(te_donus):,} ({te_donus['tanim'].nunique():,} trafo)"
    )
    print(f"  toplam bos gun: {int(te_donus['_bosluk'].sum()):,}")

    # ---- v55 uzerinde ----
    sub = pd.read_csv(KOK / "submissions/tuketim_v55_gunolcek.csv", encoding="utf-8")
    mm = te.merge(sub, on="id", how="left", validate="one_to_one")
    mm["r"] = np.log1p(mm["tuketim"].to_numpy("float64")) - np.log1p(mm["guc"].to_numpy("float64"))
    mm = mm.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    mg = mm.groupby("tanim", observed=True)
    mm["t_ilk"] = mg["tarih"].transform("min")
    mm["t_son"] = mg["tarih"].transform("max")
    mm["t_yas"] = (mm["tarih"] - mm["t_ilk"]).dt.days
    mm["t_kalan"] = (mm["t_son"] - mm["tarih"]).dt.days
    mm["yeni"] = ~mm["tanim"].isin(tr_tanim)
    # v55'in kendi kararli seviyesi: yas 8..28
    vref = mm[(mm["t_yas"] >= 8) & (mm["t_yas"] <= 28)].groupby("tanim", observed=True)["r"].mean()
    mm["vref"] = mm["tanim"].map(vref)
    mm["vsapma"] = mm["r"] - mm["vref"]

    print("\n=== 6) v55 MODELI test ilk-gununde NE YAZIYOR? (kendi yas 8..28 referansina gore) ===")
    print(f"{'yas':>4} {'n':>7} {'YENI ort':>10} {'n':>7} {'ESKI ort':>10} {'n':>7}")
    for y in range(0, 6):
        s = mm[(mm["t_yas"] == y) & mm["vref"].notna()]
        sy, se = s[s["yeni"]], s[~s["yeni"]]
        print(
            f"{y:>4} {len(s):>7,} "
            f"{sy['vsapma'].mean() if len(sy) else float('nan'):>+10.4f} {len(sy):>7,} "
            f"{se['vsapma'].mean() if len(se) else float('nan'):>+10.4f} {len(se):>7,}"
        )

    # ilk gun satirlari, sinif bazinda
    ilkgun = mm[mm["t_yas"] == 0].copy()
    ilkgun["ilkever"] = ilkgun["yeni"]
    print(f"\ntest ilk-gun satiri toplam {len(ilkgun):,}")
    print(f"  ILK-EVER (yeni trafo)      {int(ilkgun['ilkever'].sum()):,}")
    print(f"  ESKI trafo (donus/devam)   {int((~ilkgun['ilkever']).sum()):,}")
    esk = ilkgun[~ilkgun["ilkever"]].copy()
    tr_son_map = tr.groupby("tanim", observed=True)["tarih"].max()
    esk["tr_son"] = esk["tanim"].map(tr_son_map)
    esk["bosluk"] = (esk["tarih"] - esk["tr_son"]).dt.days - 1
    print("   ESKI trafolarin bosluk dagilimi:")
    for k, n in esk["bosluk"].value_counts().head(8).items():
        print(f"     bosluk {int(k):>4} gun -> {n:,}")
    kesintisiz = int((esk["bosluk"] <= 0).sum())
    print(f"   bosluksuz (train 03-31'de bitip 04-01'de devam): {kesintisiz:,}")
    print(f"   GERCEK DONUS (bosluk>0): {int((esk['bosluk'] > 0).sum()):,}")

    # test son gun (07-31 disi)
    songun = mm[(mm["t_kalan"] == 0) & (mm["t_son"] < TE_SON)]
    print(f"\ntest SON-gun satiri (07-31 disi): {len(songun):,} trafo")
    print(
        f"  bunlarin v55 sapmasi: {songun['vsapma'].mean():+.4f} (n={songun['vsapma'].notna().sum():,})"
    )

    # test bosluk donusu
    mm["_onc"] = mg["tarih"].shift(1)
    mm["_bosluk"] = ((mm["tarih"] - mm["_onc"]).dt.days - 1.0).fillna(-1.0)
    tdon = mm[mm["_bosluk"] > 0]
    print(f"test ic-bosluk donus satiri: {len(tdon):,}")
    print(f"  v55 sapmasi: {tdon['vsapma'].mean():+.4f}")

    mm.to_parquet(KOK / "data/interim/eksen2_v55.parquet")
    print("\nyazildi: data/interim/eksen2_v55.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
