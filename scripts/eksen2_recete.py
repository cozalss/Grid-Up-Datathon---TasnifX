"""EKSEN 2c -- 2026-05-11 ANALOGU, TEST ALT GRUPLARI, dMSE DEFTERI.

Kurgu:
  b = (train'de olculen GERCEK dusus) - (v55'in o satirlarda UYGULADIGI dusus)
  optimal kayma c* = b, kazanc = -b^2 / satir; shrink ile 2*s*b - s^2*b^2 carpani.
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


def ozet(x, ad, kdrop=(0, 1, 5, 10, 25)):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return f"{ad:<46} n={len(x):>5}  --"
    s = f"{ad:<46} n={len(x):>5,} ort {x.mean():>+8.4f} sh {x.std(ddof=1) / np.sqrt(len(x)):>6.4f} med {x.median():>+8.4f}"
    kk = []
    for K in kdrop[1:]:
        if len(x) > K:
            v = x.drop(x.abs().sort_values(ascending=False).index[:K])
            kk.append(f"K{K}={v.mean():+.3f}")
    return s + "  [" + " ".join(kk) + "]"


def main():
    tr, te = yukle()
    g = tr.groupby("tanim", observed=True)
    d = tr.copy()
    d["ilk_tarih"] = g["tarih"].transform("min")
    d["son_tarih"] = g["tarih"].transform("max")
    d["yas"] = (d["tarih"] - d["ilk_tarih"]).dt.days
    d["kalan"] = (d["son_tarih"] - d["tarih"]).dt.days
    d["bosluk"] = ((d["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
    tanim_arr = d["tanim"].to_numpy()
    r_arr = d["r"].to_numpy()
    bos_arr = d["bosluk"].to_numpy()
    tuk_arr = d["tuketim"].to_numpy()
    tar_arr = d["tarih"].to_numpy()

    def ileri(i, n=21, min_n=3, atla=0):
        t = tanim_arr[i]
        v = []
        for j in range(i + 1 + atla, i + 1 + atla + n):
            if j >= len(r_arr) or tanim_arr[j] != t or bos_arr[j] > 0:
                break
            v.append(r_arr[j])
        return float(np.mean(v)) if len(v) >= min_n else np.nan

    # ============ A) DONUS OLAYLARI, bosluk ve parti buyuklugune gore ============
    print("=" * 100)
    print("A) BOSLUK DONUSU -- 2026-05-11 kohortunun train ANALOGU")
    print("=" * 100)
    idx = np.flatnonzero(bos_arr > 0)
    donus_say = pd.Series(tar_arr[idx]).value_counts()
    rows = []
    for i in idx:
        rf = ileri(i)
        rf1 = ileri(i, atla=1)  # donus gunu +1'den itibaren (gun+1 kontrolu icin)
        if np.isnan(rf):
            continue
        t = pd.Timestamp(tar_arr[i])
        rows.append(
            (
                tanim_arr[i],
                t,
                bos_arr[i],
                r_arr[i] - rf,
                (r_arr[i + 1] - rf1)
                if (i + 1 < len(r_arr) and tanim_arr[i + 1] == tanim_arr[i] and not np.isnan(rf1))
                else np.nan,
                tuk_arr[i] <= 0,
                int(donus_say.get(t, 0)),
            )
        )
    bd = pd.DataFrame(
        rows, columns=["tanim", "tarih", "bosluk", "sapma", "sapma1", "sifir", "parti_n"]
    )
    print(ozet(bd["sapma"], "TUM donusler"))
    print(ozet(bd["sapma1"], "  donus gunu +1 (kontrol: 0 olmali)"))
    print("\n-- bosluk x parti buyuklugu --")
    bk = pd.cut(
        bd["bosluk"], [0, 7, 30, 60, 150, 400], labels=["1-7", "8-30", "31-60", "61-150", "151+"]
    )
    pk = np.where(bd["parti_n"] >= 20, "TOPLU(>=20)", "TEKIL(<20)")
    for (a, b), x in bd.groupby([bk, pk], observed=True):
        print(ozet(x["sapma"], f"  bosluk {a}  {b}"))
    print("\n-- 2026-05-11 ile en yakin yapisal analoglar --")
    print(
        ozet(
            bd[(bd["bosluk"].between(30, 60)) & (bd["parti_n"] >= 20)]["sapma"],
            "  bosluk 31-60 gun & TOPLU  (test: 493 trafo x 44g)",
        )
    )
    print(
        ozet(
            bd[(bd["bosluk"] >= 150) & (bd["parti_n"] >= 20)]["sapma"],
            "  bosluk 151+ gun & TOPLU   (test: ~330 trafo x 240-330g)",
        )
    )
    print(ozet(bd[(bd["bosluk"] >= 150)]["sapma"], "  bosluk 151+ gun (parti farketmez)"))
    # mevsimsel ikiz
    ik = bd[(bd["tarih"] >= "2025-04-01") & (bd["tarih"] <= "2025-07-31")]
    print(ozet(ik["sapma"], "  IKIZ PENCERE (2025-04..07) tum donusler"))
    print(ozet(ik[ik["parti_n"] >= 20]["sapma"], "  IKIZ PENCERE TOPLU donusler"))

    # ============ B) DOGUM, parti buyuklugune gore ============
    print("\n" + "=" * 100)
    print("B) DOGUM GUNU -- parti buyuklugune gore")
    print("=" * 100)
    ilk = d.groupby("tanim", observed=True)["ilk_tarih"].first()
    dogan = ilk[ilk > TR_BAS]
    dog_say = dogan.value_counts()
    dd = d[d["tanim"].isin(dogan.index)].copy()
    rr = dd[(dd["yas"] >= 8) & (dd["yas"] <= 28)].groupby("tanim", observed=True)["r"]
    rr = rr.mean()[rr.size() >= 5]
    dd["ref"] = dd["tanim"].map(rr)
    g0 = dd[(dd["yas"] == 0) & dd["ref"].notna()].copy()
    g0["sapma"] = g0["r"] - g0["ref"]
    g0["parti_n"] = g0["ilk_tarih"].map(dog_say)
    print(ozet(g0["sapma"], "TUM dogumlar"))
    pk2 = pd.cut(
        g0["parti_n"], [0, 1, 4, 19, 99, 400], labels=["1", "2-4", "5-19", "20-99", "100+"]
    )
    for k, x in g0.groupby(pk2, observed=True):
        print(ozet(x["sapma"], f"  parti buyuklugu {k}"))
    ikd = g0[(g0["ilk_tarih"] >= "2025-04-01") & (g0["ilk_tarih"] <= "2025-07-31")]
    print(ozet(ikd["sapma"], "  IKIZ PENCERE (2025-04..07)"))
    print(ozet(ikd[ikd["parti_n"] >= 100]["sapma"], "  IKIZ PENCERE parti 100+"))
    print("\n-- 100+ partiler tek tek --")
    for t, x in g0[g0["parti_n"] >= 100].groupby("ilk_tarih"):
        print(ozet(x["sapma"], f"  {str(t.date())} (n_parti={int(x['parti_n'].iloc[0])})"))

    # ============ C) SON GUN ============
    print("\n" + "=" * 100)
    print("C) SON GUN (devreden cikma / kismi gun)")
    print("=" * 100)
    son = d.groupby("tanim", observed=True)["son_tarih"].first()
    olen = son[son < TR_SON]
    ddo = d[d["tanim"].isin(olen.index)].copy()
    r2 = ddo[(ddo["kalan"] >= 8) & (ddo["kalan"] <= 28)].groupby("tanim", observed=True)["r"]
    r2 = r2.mean()[r2.size() >= 5]
    ddo["ref"] = ddo["tanim"].map(r2)
    s0 = ddo[(ddo["kalan"] == 0) & ddo["ref"].notna()].copy()
    s0["sapma"] = s0["r"] - s0["ref"]
    # HAYATTA olan trafolari ayikla: son gunde zaten olu (sifir kuyruklu) olanlar
    olu_kuyruk = (
        ddo[(ddo["kalan"] >= 1) & (ddo["kalan"] <= 14)]
        .groupby("tanim", observed=True)["tuketim"]
        .apply(lambda s: float((s <= 0).mean()))
    )
    s0["olu_kuyruk"] = s0["tanim"].map(olu_kuyruk)
    print(ozet(s0["sapma"], "TUM son gunler"))
    print(
        ozet(s0[s0["olu_kuyruk"] < 0.5]["sapma"], "  CANLI trafolar (onceki 14g sifir orani<0.5)")
    )
    print(ozet(s0[s0["olu_kuyruk"] >= 0.5]["sapma"], "  OLU trafolar"))
    iks = s0[(s0["son_tarih"] >= "2025-04-01") & (s0["son_tarih"] <= "2025-07-31")]
    print(ozet(iks["sapma"], "  IKIZ PENCERE"))
    print(ozet(iks[iks["olu_kuyruk"] < 0.5]["sapma"], "  IKIZ PENCERE CANLI"))

    # ============ D) TEST ALT GRUPLARI + v55 ============
    print("\n" + "=" * 100)
    print("D) TEST ALT GRUPLARI ve v55'in UYGULADIGI SAPMA")
    print("=" * 100)
    sub = pd.read_csv(KOK / "submissions/tuketim_v55_gunolcek.csv", encoding="utf-8")
    m = te.merge(sub, on="id", how="left", validate="one_to_one")
    m["r"] = np.log1p(m["tuketim"].to_numpy("float64")) - np.log1p(m["guc"].to_numpy("float64"))
    m = m.sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    mg = m.groupby("tanim", observed=True)
    m["t_ilk"] = mg["tarih"].transform("min")
    m["t_son"] = mg["tarih"].transform("max")
    m["t_yas"] = (m["tarih"] - m["t_ilk"]).dt.days
    m["t_kalan"] = (m["t_son"] - m["tarih"]).dt.days
    m["t_bosluk"] = ((m["tarih"] - mg["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
    vr = m[(m["t_yas"] >= 8) & (m["t_yas"] <= 28)].groupby("tanim", observed=True)["r"]
    vr = vr.mean()[vr.size() >= 5]
    m["vref"] = m["tanim"].map(vr)
    m["vsapma"] = m["r"] - m["vref"]
    tr_son_map = tr.groupby("tanim", observed=True)["tarih"].max()
    m["tr_son"] = m["tanim"].map(tr_son_map)
    m["yeni"] = m["tr_son"].isna()

    ig = m[m["t_yas"] == 0].copy()
    ig["ilk_parti_n"] = ig["t_ilk"].map(ig["t_ilk"].value_counts())
    ig["gercek_bosluk"] = np.where(ig["yeni"], np.nan, (ig["tarih"] - ig["tr_son"]).dt.days - 1)
    print(f"{'grup':<44} {'satir':>7} {'v55 sapma':>11} {'sh':>7}")

    def gv(mask, ad):
        x = ig.loc[mask, "vsapma"].dropna()
        n = int(mask.sum())
        print(
            f"{ad:<44} {n:>7,} {x.mean() if len(x) else float('nan'):>+11.4f} "
            f"{(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else float('nan'):>7.4f}"
        )
        return n, (x.mean() if len(x) else np.nan)

    n_yeni_toplu, v_yeni_toplu = gv(
        ig["yeni"] & (ig["ilk_parti_n"] >= 100), "YENI trafo, parti 100+ (05-11 dahil)"
    )
    n_yeni_orta, v_yeni_orta = gv(
        ig["yeni"] & ig["ilk_parti_n"].between(20, 99), "YENI trafo, parti 20-99"
    )
    n_yeni_tekil, v_yeni_tekil = gv(ig["yeni"] & (ig["ilk_parti_n"] < 20), "YENI trafo, parti <20")
    n_esk_kes, v_esk_kes = gv(
        (~ig["yeni"]) & (ig["gercek_bosluk"] <= 0), "ESKI trafo, KESINTISIZ (bosluk<=0)"
    )
    n_esk_k, v_esk_k = gv(
        (~ig["yeni"]) & ig["gercek_bosluk"].between(1, 60), "ESKI trafo, bosluk 1-60 gun"
    )
    n_esk_u, v_esk_u = gv((~ig["yeni"]) & (ig["gercek_bosluk"] > 60), "ESKI trafo, bosluk 60+ gun")

    ic = m[m["t_bosluk"] > 0]
    n_ic = len(ic)
    v_ic = float(ic["vsapma"].dropna().mean())
    print(f"{'IC BOSLUK donusu':<44} {n_ic:>7,} {v_ic:>+11.4f}")
    sg = m[(m["t_kalan"] == 0) & (m["t_son"] < TE_SON)]
    n_sg = len(sg)
    v_sg = float(sg["vsapma"].dropna().mean())
    print(f"{'SON gun (07-31 disi)':<44} {n_sg:>7,} {v_sg:>+11.4f}")
    # son gun: olu mu canli mi
    sg_olu = (
        m[(m["t_kalan"].between(1, 14)) & (m["t_son"] < TE_SON)]
        .groupby("tanim", observed=True)["tuketim"]
        .apply(lambda s: float((s <= 1.0).mean()))
    )
    print(
        f"   son-gun trafolarinin v55 tahmininde ~sifir orani: {float((sg_olu >= 0.5).mean()) * 100:.1f}%"
    )

    # ============ E) dMSE DEFTERI ============
    print("\n" + "=" * 100)
    print("E) dMSE DEFTERI")
    print("=" * 100)
    # gercek dususler (train, ikiz pencere ile capraz kontrol edilmis)
    D_dogum_buyuk = float(g0[g0["parti_n"] >= 100]["sapma"].mean())
    D_dogum_orta = float(g0[g0["parti_n"].between(20, 99)]["sapma"].mean())
    D_dogum_tekil = float(g0[g0["parti_n"] < 20]["sapma"].mean())
    D_don_kisa = float(bd[bd["bosluk"].between(1, 60)]["sapma"].mean())
    D_don_uzun = float(bd[bd["bosluk"] > 60]["sapma"].mean())
    D_ic = float(bd[bd["bosluk"].between(1, 60)]["sapma"].mean())
    D_son = float(s0[s0["olu_kuyruk"] < 0.5]["sapma"].mean())

    kayitlar = [
        ("YENI trafo parti 100+", n_yeni_toplu, D_dogum_buyuk, v_yeni_toplu),
        ("YENI trafo parti 20-99", n_yeni_orta, D_dogum_orta, v_yeni_orta),
        ("YENI trafo parti <20", n_yeni_tekil, D_dogum_tekil, v_yeni_tekil),
        ("ESKI bosluk 1-60g", n_esk_k, D_don_kisa, v_esk_k),
        ("ESKI bosluk 60+g", n_esk_u, D_don_uzun, v_esk_u),
        ("IC BOSLUK donusu", n_ic, D_ic, v_ic),
        ("SON gun (07-31 disi)", n_sg, D_son, v_sg),
    ]
    print(
        f"{'grup':<26} {'n':>7} {'p':>9} {'D_gercek':>9} {'D_v55':>8} {'b':>8} "
        f"{'c*=b':>7} {'dMSE(c*)':>10} {'dMSE(0.6b)':>11}"
    )
    tot_opt = 0.0
    tot_shr = 0.0
    for ad, n, dg, dv in kayitlar:
        p = n / N_TEST
        b = dg - dv
        dm_opt = p * (-(b**2))
        s = 0.6
        dm_shr = p * (s * s * b * b - 2 * s * b * b)
        tot_opt += dm_opt
        tot_shr += dm_shr
        print(
            f"{ad:<26} {n:>7,} {p:>9.6f} {dg:>+9.4f} {dv:>+8.4f} {b:>+8.4f} "
            f"{b:>+7.3f} {dm_opt:>10.6f} {dm_shr:>11.6f}"
        )
    print(
        f"{'TOPLAM':<26} {'':>7} {'':>9} {'':>9} {'':>8} {'':>8} {'':>7} "
        f"{tot_opt:>10.6f} {tot_shr:>11.6f}"
    )
    print(
        f"\nRMSLE  simdi {np.sqrt(MSE0):.5f}"
        f"  -> optimal {np.sqrt(MSE0 + tot_opt):.5f} ({np.sqrt(MSE0 + tot_opt) - np.sqrt(MSE0):+.5f})"
        f"  -> 0.6 shrink {np.sqrt(MSE0 + tot_shr):.5f} ({np.sqrt(MSE0 + tot_shr) - np.sqrt(MSE0):+.5f})"
    )
    print(f"1. sira 1.00635 icin gereken dMSE = {1.00635**2 - MSE0:+.5f}")

    m.to_csv(KOK / "data/interim/eksen2_v55.csv.gz", index=False, compression="gzip")
    print("\nyazildi: data/interim/eksen2_v55.csv.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
