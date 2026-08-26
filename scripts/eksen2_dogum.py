"""EKSEN 2 -- DOGUM GUNU / SON GUN / BOSLUK SONRASI ILK GUN etkisi.

Train tarafinda olcup test tarafina tasiyor. Cikti: sayilar, tablo halinde.
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
PARTI_ESIGI = 20  # ayni gun >= 20 trafo -> TOPLU parti


def yukle():
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        encoding="utf-8",
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        encoding="utf-8",
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tr["r"] = np.log1p(tr["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        tr["guc"].to_numpy(dtype="float64")
    )
    tr = tr.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    te = te.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    return tr, te


def olay_tablosu(tr: pd.DataFrame) -> pd.DataFrame:
    """Her trafo icin gun-indeksi (kacinci kayit gunu) ve olay bayraklari."""
    g = tr.groupby("tanim", observed=True)
    ilk = g["tarih"].transform("min")
    son = g["tarih"].transform("max")
    d = tr.copy()
    d["ilk_tarih"] = ilk
    d["son_tarih"] = son
    d["yas"] = (d["tarih"] - ilk).dt.days
    d["kalan"] = (son - d["tarih"]).dt.days
    # onceki kayit gunu -> bosluk uzunlugu
    onceki = g["tarih"].shift(1)
    d["bosluk"] = (d["tarih"] - onceki).dt.days - 1.0  # 0 = kesintisiz
    d["bosluk"] = d["bosluk"].fillna(-1.0)  # -1 = ilk kayit
    sonraki = g["tarih"].shift(-1)
    d["ileri_bosluk"] = (sonraki - d["tarih"]).dt.days - 1.0
    d["ileri_bosluk"] = d["ileri_bosluk"].fillna(-1.0)
    return d


def kararli_referans(d: pd.DataFrame, alan: str, bas: int, bit: int) -> pd.Series:
    """Trafo bazinda, yas araligi [bas,bit] icindeki r ortalamasi."""
    m = d[(d[alan] >= bas) & (d[alan] <= bit)]
    return m.groupby("tanim", observed=True)["r"].mean()


def ozetle(x: pd.Series, ad: str) -> str:
    if len(x) == 0:
        return f"{ad:<34} n=0"
    return (
        f"{ad:<34} n={len(x):>6,}  ort {x.mean():+.4f}  sh {x.std(ddof=1) / np.sqrt(len(x)):.4f}"
        f"  medyan {x.median():+.4f}  std {x.std(ddof=1):.4f}"
    )


def main() -> int:
    tr, te = yukle()
    d = olay_tablosu(tr)
    print(f"train {len(tr):,} satir  {tr['tanim'].nunique():,} trafo")

    # ================= (a)+(b) DOGUM GUNU =================
    ilk_tarih = d.groupby("tanim", observed=True)["ilk_tarih"].first()
    dogan = ilk_tarih[ilk_tarih > TR_BAS]
    print(f"\n=== (a) DOGUM: train icinde dogan trafo {len(dogan):,} ===")
    parti_say = dogan.value_counts()
    toplu_tarihler = set(parti_say[parti_say >= PARTI_ESIGI].index)
    print(f"TOPLU tarih sayisi (>= {PARTI_ESIGI} trafo): {len(toplu_tarihler)}")
    print(
        f"  TOPLU tarihlerde dogan: {int(parti_say[parti_say >= PARTI_ESIGI].sum()):,}"
        f"  | TEKIL tarihlerde: {int(parti_say[parti_say < PARTI_ESIGI].sum()):,}"
    )

    dd = d[d["tanim"].isin(dogan.index)].copy()
    # kararli referans: yas 8..28 (en az 5 gozlem)
    ref = kararli_referans(dd, "yas", 8, 28)
    n_ref = dd[(dd["yas"] >= 8) & (dd["yas"] <= 28)].groupby("tanim", observed=True).size()
    ref = ref[n_ref >= 5]
    dd["ref"] = dd["tanim"].map(ref)
    dd["sapma"] = dd["r"] - dd["ref"]
    dd["toplu"] = dd["ilk_tarih"].isin(toplu_tarihler)

    print("\n--- yas profili (kararli referans = yas 8..28) ---")
    print(
        f"{'yas':>4} {'n':>7} {'ort sapma':>11} {'sh':>8} {'sifir%':>8} "
        f"{'TOPLU ort':>11} {'n':>7} {'TEKIL ort':>11} {'n':>7}"
    )
    for y in range(0, 10):
        s = dd[(dd["yas"] == y) & dd["ref"].notna()]
        if len(s) == 0:
            continue
        st = s[s["toplu"]]
        sk = s[~s["toplu"]]
        print(
            f"{y:>4} {len(s):>7,} {s['sapma'].mean():>+11.4f} "
            f"{s['sapma'].std(ddof=1) / np.sqrt(len(s)):>8.4f} "
            f"{(s['tuketim'] <= 0).mean() * 100:>7.2f}% "
            f"{st['sapma'].mean() if len(st) else float('nan'):>+11.4f} {len(st):>7,} "
            f"{sk['sapma'].mean() if len(sk) else float('nan'):>+11.4f} {len(sk):>7,}"
        )
    # sifir orani karsilastirma
    g0 = dd[(dd["yas"] == 0) & dd["ref"].notna()]
    gk = dd[(dd["yas"] >= 8) & (dd["yas"] <= 28) & dd["ref"].notna()]
    print(
        f"\nsifir orani  yas0 {(g0['tuketim'] <= 0).mean() * 100:.2f}%"
        f"  |  yas 8..28 {(gk['tuketim'] <= 0).mean() * 100:.2f}%"
    )
    print(
        f"  TOPLU yas0 sifir {(g0[g0['toplu']]['tuketim'] <= 0).mean() * 100:.2f}%"
        f"  | TEKIL yas0 sifir {(g0[~g0['toplu']]['tuketim'] <= 0).mean() * 100:.2f}%"
    )
    print("\n" + ozetle(g0["sapma"].dropna(), "gun-0 sapma TUMU"))
    print(ozetle(g0[g0["toplu"]]["sapma"].dropna(), "gun-0 sapma TOPLU parti"))
    print(ozetle(g0[~g0["toplu"]]["sapma"].dropna(), "gun-0 sapma TEKIL"))

    # sifir olmayanlar
    g0n = g0[g0["tuketim"] > 0]
    print(ozetle(g0n["sapma"].dropna(), "gun-0 sapma (sifir HARIC)"))
    print(ozetle(g0n[g0n["toplu"]]["sapma"].dropna(), "gun-0 sifir haric TOPLU"))
    print(ozetle(g0n[~g0n["toplu"]]["sapma"].dropna(), "gun-0 sifir haric TEKIL"))

    print("\n--- (b) PARTI BAZINDA gun-0 dususu (n>=20) ---")
    print(
        f"{'tarih':<12} {'n':>5} {'ort sapma':>11} {'sh':>8} {'std':>8} {'sifir%':>8} {'medyan':>9}"
    )
    tab = []
    for t, s in g0.groupby("ilk_tarih"):
        s2 = s["sapma"].dropna()
        if len(s2) < 3:
            continue
        tab.append(
            (
                t,
                len(s2),
                s2.mean(),
                s2.std(ddof=1) / np.sqrt(len(s2)),
                s2.std(ddof=1),
                (s["tuketim"] <= 0).mean() * 100,
                s2.median(),
            )
        )
    tab.sort(key=lambda z: -z[1])
    for t, n, m, sh, sd, z, md in tab[:25]:
        print(
            f"{str(t.date()):<12} {n:>5} {m:>+11.4f} {sh:>8.4f} {sd:>8.4f} {z:>7.2f}% {md:>+9.4f}"
        )
    kucuk = [z for z in tab if z[1] < PARTI_ESIGI]
    if kucuk:
        w = np.array([z[1] for z in kucuk], dtype=float)
        mm = np.array([z[2] for z in kucuk])
        print(f"{'TEKIL toplam':<12} {int(w.sum()):>5} {float((w * mm).sum() / w.sum()):>+11.4f}")

    # trafo bazinda ayristirma (KURAL 1): en buyuk K trafo atilinca
    print("\n--- gun-0 etkisi: en buyuk |sapma| K trafo atilinca ---")
    s0 = g0["sapma"].dropna().sort_values()
    print(f"{'K':>4} {'n':>7} {'ort sapma':>11}")
    for K in (0, 1, 5, 10, 25, 50, 100):
        if K == 0:
            v = s0
        else:
            idx = g0["sapma"].dropna().abs().sort_values(ascending=False).index[:K]
            v = g0["sapma"].dropna().drop(idx)
        print(f"{K:>4} {len(v):>7,} {v.mean():>+11.4f}")

    # ================= (d) SON GUN =================
    son_tarih = d.groupby("tanim", observed=True)["son_tarih"].first()
    olen = son_tarih[son_tarih < TR_SON]
    print(f"\n=== (d) SON GUN: train icinde biten trafo {len(olen):,} ===")
    olum_say = olen.value_counts()
    print("en yogun bitis tarihleri:")
    for t, n in olum_say.head(10).items():
        print(f"  {str(t.date())}  {n:,}")
    ddo = d[d["tanim"].isin(olen.index)].copy()
    refs = kararli_referans(ddo, "kalan", 8, 28)
    n_refs = ddo[(ddo["kalan"] >= 8) & (ddo["kalan"] <= 28)].groupby("tanim", observed=True).size()
    refs = refs[n_refs >= 5]
    ddo["ref"] = ddo["tanim"].map(refs)
    ddo["sapma"] = ddo["r"] - ddo["ref"]
    print("\n--- kalan-gun profili (referans = kalan 8..28) ---")
    print(f"{'kalan':>6} {'n':>7} {'ort sapma':>11} {'sh':>8} {'sifir%':>8}")
    for k in range(0, 8):
        s = ddo[(ddo["kalan"] == k) & ddo["ref"].notna()]
        if len(s) == 0:
            continue
        print(
            f"{k:>6} {len(s):>7,} {s['sapma'].mean():>+11.4f} "
            f"{s['sapma'].std(ddof=1) / np.sqrt(len(s)):>8.4f} "
            f"{(s['tuketim'] <= 0).mean() * 100:>7.2f}%"
        )
    sk0 = ddo[(ddo["kalan"] == 0) & ddo["ref"].notna()]
    print("\n" + ozetle(sk0["sapma"].dropna(), "SON gun sapma"))
    print(ozetle(sk0[sk0["tuketim"] > 0]["sapma"].dropna(), "SON gun sapma (sifir haric)"))
    print("\n--- son gun: en buyuk K trafo atilinca ---")
    for K in (0, 1, 5, 10, 25, 50):
        v = sk0["sapma"].dropna()
        if K:
            v = v.drop(v.abs().sort_values(ascending=False).index[:K])
        print(f"{K:>4} {len(v):>7,} {v.mean():>+11.4f}")

    # ================= (e) BOSLUK SONRASI ILK GUN =================
    print("\n=== (e) IC BOSLUK SONRASI ILK GUN ===")
    bos = d[d["bosluk"] > 0].copy()
    print(
        f"train'de bosluk sonrasi ilk gun satiri: {len(bos):,}  ({bos['tanim'].nunique():,} trafo)"
    )
    # referans: bosluktan SONRAKI 1..21. gunler (donus gunu haric)
    d["_sira"] = d.groupby("tanim", observed=True).cumcount()
    bos_sira = d.loc[d["bosluk"] > 0, ["tanim", "_sira", "tarih", "bosluk", "r", "tuketim"]]
    sonuc = []
    ind = d.set_index(["tanim", "_sira"])["r"]
    tuk = d.set_index(["tanim", "_sira"])["tuketim"]
    dmap = {(t, s): i for i, (t, s) in enumerate(zip(d["tanim"], d["_sira"]))}
    r_arr = d["r"].to_numpy()
    tuk_arr = d["tuketim"].to_numpy()
    bosluk_arr = d["bosluk"].to_numpy()
    tanim_arr = d["tanim"].to_numpy()
    for t, s, tarih, bl, rr, tk in bos_sira.itertuples(index=False):
        i = dmap[(t, s)]
        # sonraki 1..21 gun, ayni trafo, ARADA yeni bosluk yoksa
        vals = []
        for j in range(i + 1, i + 22):
            if j >= len(d) or tanim_arr[j] != t:
                break
            if bosluk_arr[j] > 0:
                break
            vals.append(r_arr[j])
        if len(vals) < 5:
            continue
        sonuc.append((t, tarih, bl, rr - float(np.mean(vals)), tk <= 0))
    bd = pd.DataFrame(sonuc, columns=["tanim", "tarih", "bosluk", "sapma", "sifir"])
    print(ozetle(bd["sapma"], "BOSLUK sonrasi ilk gun sapma"))
    print(f"  sifir orani donus gununde {bd['sifir'].mean() * 100:.2f}%")
    print("\n--- bosluk uzunluguna gore ---")
    print(f"{'bosluk':<12} {'n':>7} {'ort sapma':>11} {'sh':>8} {'sifir%':>8}")
    kova = pd.cut(
        bd["bosluk"], [0, 1, 3, 7, 30, 90, 400], labels=["1", "2-3", "4-7", "8-30", "31-90", "90+"]
    )
    for k, s in bd.groupby(kova, observed=True):
        print(
            f"{str(k):<12} {len(s):>7,} {s['sapma'].mean():>+11.4f} "
            f"{s['sapma'].std(ddof=1) / np.sqrt(len(s)):>8.4f} {s['sifir'].mean() * 100:>7.2f}%"
        )
    print("\n--- bosluk donusu: en buyuk K trafo atilinca ---")
    for K in (0, 1, 5, 10, 25, 50):
        v = bd["sapma"]
        if K:
            v = v.drop(v.abs().sort_values(ascending=False).index[:K])
        print(f"{K:>4} {len(v):>7,} {v.mean():>+11.4f}")

    # bosluk ONCESI son gun
    onc = d[d["ileri_bosluk"] > 0].copy()
    sonuc2 = []
    for i in np.flatnonzero(bosluk_arr * 0 + (d["ileri_bosluk"].to_numpy() > 0)):
        t = tanim_arr[i]
        vals = []
        for j in range(i - 1, i - 22, -1):
            if j < 0 or tanim_arr[j] != t:
                break
            if bosluk_arr[j + 1] > 0:
                break
            vals.append(r_arr[j])
        if len(vals) < 5:
            continue
        sonuc2.append((t, r_arr[i] - float(np.mean(vals)), tuk_arr[i] <= 0))
    od = pd.DataFrame(sonuc2, columns=["tanim", "sapma", "sifir"])
    print("\n" + ozetle(od["sapma"], "BOSLUK oncesi son gun sapma"))
    print(f"  sifir orani {od['sifir'].mean() * 100:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
