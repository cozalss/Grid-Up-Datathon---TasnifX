"""EKSEN 1 -- TEMIZ COKLU-KESME BACKTEST: trafo yanliligi b_i tasinabilir mi?

Uretim onbelleginde tek TEMIZ (yalniz-gecmis) fold vardi: kis26. Bu betik
bagimsiz bir VEKIL model kurar ve BES temiz kesme uretir; boylece b_i
kestiricisinin kesmeler arasi tasinip tasinmadigi ADIL olcuebilir.

KURGU
-----
Hedef: ofs = log1p(tuketim) - log1p(guc).

Kesme C icin:
  * oznitelikler YALNIZCA C'den onceki gunlerden (as-of) hesaplanir,
  * hedef penceresi (C, C+122gun],
  * vekil model, C'den ONCE biten etiket pencerelerine sahip ic bloklarda
    egitilir (yuvarlanan koken). Hicbir asamada gelecege bakilmaz.

Ic egitim bloklari: ay sonu kesmeleri c' (c' + 122 gun <= C). Bu kosulu
saglayan blok yoksa (erken kesmeler) etiket penceresi C'de kirpilir ve bu
durum raporda acikca "KIRPIK EGITIM" olarak isaretlenir.

Calistirma:
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/eksen1_coklu_kesme.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

KOK = Path(__file__).resolve().parents[1]
HAM = KOK / "data" / "raw"
CIKTI = KOK / "data" / "interim" / "eksen1_kesme"
CIKTI.mkdir(parents=True, exist_ok=True)

RASGELE = 1000
EGITIM_TAVAN = 900_000  # kesmeler arasi vekil kalitesini esitlemek icin
UFUK = 122
SICAK_PAY = 0.7784  # testte sicak rejimin satir payi
TABAN_MSE = 1.03207  # mevcut gonderimin MSE'si (RMSLE 1,01591^2)

# ---------------------------------------------------------------- veri yukle


def veri_yukle():
    d = pd.read_csv(HAM / "train.csv", parse_dates=["tarih"])
    d = d.sort_values(["tanim", "tarih"], kind="stable").reset_index(drop=True)
    d["ofs"] = np.log1p(d["tuketim"].to_numpy()) - np.log1p(d["guc"].to_numpy())
    return d


def panel_kur(d):
    """Yogun (trafo x gun) matrisleri. 5.344 x 455 -> ~10 MB, bellek sorunu yok."""
    trafolar = np.sort(d["tanim"].unique())
    gunler = pd.date_range(d["tarih"].min(), d["tarih"].max(), freq="D")
    t_ix = pd.Series(np.arange(len(trafolar)), index=trafolar)
    g_ix = pd.Series(np.arange(len(gunler)), index=gunler)

    ti = t_ix.reindex(d["tanim"]).to_numpy()
    gi = g_ix.reindex(d["tarih"]).to_numpy()

    n_t, n_g = len(trafolar), len(gunler)
    M = np.full((n_t, n_g), np.nan, dtype=np.float32)
    M[ti, gi] = d["ofs"].to_numpy(dtype=np.float32)
    Z = np.full((n_t, n_g), np.nan, dtype=np.float32)
    Z[ti, gi] = (d["tuketim"].to_numpy() == 0).astype(np.float32)

    ilk = d.groupby("tanim", sort=True).first()
    guc = ilk["guc"].reindex(trafolar).to_numpy(dtype=np.float64)
    lok = ilk["lokasyon"].reindex(trafolar).astype(str)

    return dict(
        trafolar=trafolar,
        gunler=gunler,
        t_ix=t_ix,
        g_ix=g_ix,
        ti=ti,
        gi=gi,
        M=M,
        Z=Z,
        guc=guc,
        lok=lok,
        n_t=n_t,
        n_g=n_g,
    )


def kategoriler(P):
    """Trafo bazli kategorik kodlar -- kesmeden bagimsiz, sizinti yok."""
    lok = P["lok"]
    parca = lok.str.split(">")
    il = parca.str[0]
    bolge = parca.apply(lambda x: x[1] if len(x) >= 3 else x[0])
    ilce = parca.str[-1]
    tan = pd.Series(P["trafolar"].astype(str), index=lok.index)
    kolonlar = {"il": il, "bolge": bolge, "ilce": ilce}
    for k in (2, 3, 4, 5):
        kolonlar[f"on{k}"] = tan.str[:k]
    kodlar, adlar = [], []
    for ad, s in kolonlar.items():
        kodlar.append(pd.factorize(s.to_numpy())[0].astype(np.int32))
        adlar.append(ad)
    return np.column_stack(kodlar), adlar


# ------------------------------------------------------- as-of oznitelikleri

PENCERELER = [7, 14, 28, 56, 112, 224, 365]


def kumulatifler(P):
    M, Z = P["M"], P["Z"]
    gozlem = ~np.isnan(M)
    Mf = np.where(gozlem, M, 0.0).astype(np.float64)
    Zf = np.where(gozlem, np.nan_to_num(Z), 0.0).astype(np.float64)
    S = np.cumsum(Mf, axis=1)
    S2 = np.cumsum(Mf * Mf, axis=1)
    N = np.cumsum(gozlem, axis=1, dtype=np.float64)
    ZS = np.cumsum(Zf, axis=1)
    # son gozlem gunu (kumulatif) ve ilk gozlem gunu
    kol = np.arange(P["n_g"])[None, :]
    son_ix = np.maximum.accumulate(np.where(gozlem, kol, -1), axis=1)
    ilk_ix = np.where(gozlem.any(axis=1), np.argmax(gozlem, axis=1), 10**6)
    return dict(S=S, S2=S2, N=N, ZS=ZS, son_ix=son_ix, ilk_ix=ilk_ix, gozlem=gozlem)


def _dilim(C, k, a, b):
    """[a,b] kapali sutun araligi toplamlari (a<0 ise 0'a kirp)."""
    a = max(a, 0)
    if b < a:
        z = np.zeros(C["S"].shape[0])
        return z, z, z, z
    on = (
        (C["S"][:, a - 1], C["S2"][:, a - 1], C["N"][:, a - 1], C["ZS"][:, a - 1])
        if a > 0
        else (0.0, 0.0, 0.0, 0.0)
    )
    return (
        C["S"][:, b] - on[0],
        C["S2"][:, b] - on[1],
        C["N"][:, b] - on[2],
        C["ZS"][:, b] - on[3],
    )


def asof_oznitelik(P, C, k):
    """k sutununa (kesme gunu, dahil) kadar trafo bazli oznitelikler."""
    n_t = P["n_t"]
    ad, kol = [], []

    def ekle(isim, v):
        ad.append(isim)
        kol.append(np.asarray(v, dtype=np.float32))

    ekle("log_guc", np.log1p(P["guc"]))

    for L in PENCERELER:
        s, s2, n, zs = _dilim(C, k, k - L + 1, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            ort = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            var = np.where(n > 1, s2 / np.maximum(n, 1) - ort**2, np.nan)
            std = np.sqrt(np.maximum(var, 0))
            sifir = np.where(n > 0, zs / np.maximum(n, 1), np.nan)
        ekle(f"ort_{L}", ort)
        ekle(f"std_{L}", std)
        ekle(f"say_{L}", n)
        ekle(f"doluluk_{L}", n / float(L))
        ekle(f"sifir_{L}", sifir)

    # seviye / egim
    def pencere_ort(a, b):
        s, _, n, _ = _dilim(C, k, a, b)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    o30 = pencere_ort(k - 29, k)
    o90 = pencere_ort(k - 89, k)
    o30_90 = pencere_ort(k - 89, k - 30)
    o90_180 = pencere_ort(k - 179, k - 90)
    s_all, _, n_all, z_all = _dilim(C, k, 0, k)
    seviye_uzun = np.where(n_all > 0, s_all / np.maximum(n_all, 1), np.nan)
    ekle("seviye_90", o90)
    ekle("seviye_uzun", seviye_uzun)
    ekle("egim_30_90", o30 - o30_90)
    ekle("egim_90_180", o90 - o90_180)
    ekle("sifir_oran", np.where(n_all > 0, z_all / np.maximum(n_all, 1), np.nan))

    ilk = C["ilk_ix"]
    son = C["son_ix"][:, k]
    omur = np.where(ilk <= k, k - ilk + 1, np.nan)
    bayat = np.where(son >= 0, k - son, np.nan)
    ekle("omur", omur)
    ekle("bayat", bayat)
    ekle("doluluk_omur", np.where(omur > 0, n_all / np.maximum(omur, 1), np.nan))

    # haftagunu profili (son 112 gun), genel ortalamaya gore sapma
    a = max(k - 111, 0)
    dilim = P["M"][:, a : k + 1]
    hg = P["gunler"][a : k + 1].dayofweek.to_numpy()
    with np.errstate(invalid="ignore"):
        genel = np.nanmean(dilim, axis=1)
    for w in range(7):
        m = hg == w
        if m.sum() == 0:
            ekle(f"hg_{w}", np.full(n_t, np.nan))
            continue
        with np.errstate(invalid="ignore"):
            ortw = np.nanmean(dilim[:, m], axis=1)
        ekle(f"hg_{w}", ortw - genel)

    X = np.column_stack(kol).astype(np.float32)
    sicak = C["ilk_ix"] <= k
    return X, ad, sicak


# ------------------------------------------------------------ satir uretimi


def blok_satirlari(P, k_kesme, k_bas, k_son, sicak):
    """(k_bas..k_son) sutun araligindaki gercek gozlemler; yalniz sicak trafolar."""
    gi, ti = P["gi"], P["ti"]
    m = (gi >= k_bas) & (gi <= k_son) & sicak[ti]
    idx = np.flatnonzero(m)
    return idx


def satir_matrisi(P, Xt, kat, idx, k_kesme):
    ti = P["ti"][idx]
    gi = P["gi"][idx]
    g = P["gunler"][gi]
    takvim = np.column_stack(
        [
            g.month.to_numpy(),
            g.dayofweek.to_numpy(),
            g.dayofyear.to_numpy(),
            (gi - k_kesme).astype(np.float32),
        ]
    ).astype(np.float32)
    X = np.hstack([Xt[ti], takvim, kat[ti].astype(np.float32)])
    return X, ti, gi


# ------------------------------------------------------------- vekil model

AY_SONLARI = pd.to_datetime(
    [
        "2025-01-31",
        "2025-02-28",
        "2025-03-31",
        "2025-04-30",
        "2025-05-31",
        "2025-06-30",
        "2025-07-31",
        "2025-08-31",
        "2025-09-30",
        "2025-10-31",
    ]
)


def egitim_bloklari(C_ts, son_gun):
    """C icin ic egitim bloklari: (kesme, etiket_bas, etiket_son, kirpik?)."""
    tam = [c for c in AY_SONLARI if c + pd.Timedelta(days=UFUK) <= C_ts]
    if len(tam) >= 2:
        return [
            (c, c + pd.Timedelta(days=1), c + pd.Timedelta(days=UFUK), False) for c in tam
        ], False
    # kirpik geri dusus
    kirpik = []
    for c in AY_SONLARI:
        if c >= C_ts:
            continue
        son = min(c + pd.Timedelta(days=UFUK), C_ts)
        if (son - c).days >= 28:
            kirpik.append((c, c + pd.Timedelta(days=1), son, True))
    return kirpik, True


LGB_VEKIL = dict(
    objective="regression",
    learning_rate=0.06,
    num_leaves=96,
    min_data_in_leaf=60,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=600,
    verbose=-1,
    n_jobs=8,
    seed=RASGELE,
)


def kesme_kos(P, C, kat, kat_ad, C_ts, ofs_tum):
    g_ix = P["g_ix"]
    k = int(g_ix.loc[C_ts])
    hedef_bas = k + 1
    hedef_son = min(k + UFUK, P["n_g"] - 1)

    Xt_hedef, oz_ad, sicak_hedef = asof_oznitelik(P, C, k)

    bloklar, kirpik_mi = egitim_bloklari(C_ts, P["gunler"][-1])
    parcalar_X, parcalar_y = [], []
    for c, eb, es, _ in bloklar:
        kc = int(g_ix.loc[c])
        Xtc, _, sicak_c = asof_oznitelik(P, C, kc)
        idx = blok_satirlari(P, kc, int(g_ix.loc[eb]), int(g_ix.loc[es]), sicak_c)
        Xr, _, _ = satir_matrisi(P, Xtc, kat, idx, kc)
        parcalar_X.append(Xr)
        parcalar_y.append(ofs_tum[idx])
    Xtr = np.vstack(parcalar_X)
    ytr = np.concatenate(parcalar_y)
    del parcalar_X, parcalar_y
    if len(ytr) > EGITIM_TAVAN:
        rs = np.random.default_rng(RASGELE)
        sec = rs.choice(len(ytr), EGITIM_TAVAN, replace=False)
        Xtr, ytr = Xtr[sec], ytr[sec]

    kol_ad = oz_ad + ["ay", "haftagunu", "yilin_gunu", "ufuk"] + kat_ad
    kat_ix = list(range(len(kol_ad) - len(kat_ad), len(kol_ad)))

    t0 = time.time()
    model = lgb.LGBMRegressor(**LGB_VEKIL)
    model.fit(Xtr, ytr, feature_name=kol_ad, categorical_feature=kat_ix)
    sure = time.time() - t0

    idx_h = blok_satirlari(P, k, hedef_bas, hedef_son, sicak_hedef)
    Xh, ti_h, gi_h = satir_matrisi(P, Xt_hedef, kat, idx_h, k)
    tahmin = model.predict(Xh).astype(np.float64)
    gercek = ofs_tum[idx_h].astype(np.float64)

    return dict(
        k=k,
        C=C_ts,
        Xt=Xt_hedef,
        oz_ad=oz_ad,
        sicak=sicak_hedef,
        ti=ti_h,
        gi=gi_h,
        tahmin=tahmin,
        gercek=gercek,
        n_egitim=len(ytr),
        bloklar=bloklar,
        kirpik=kirpik_mi,
        sure=sure,
        mse=float(np.mean((gercek - tahmin) ** 2)),
    )


# ------------------------------------------------------------- b_i hesaplari


def b_tablosu(R, n_t):
    """Trafo bazli yanlilik b_i = ort(gercek) - ort(tahmin)."""
    art = R["gercek"] - R["tahmin"]
    say = np.bincount(R["ti"], minlength=n_t)
    top = np.bincount(R["ti"], weights=art, minlength=n_t)
    var = np.where(say > 0, top / np.maximum(say, 1), np.nan)
    return var, say


def b_ozet(b, say):
    m = say > 0
    w = say[m].astype(float)
    bb = b[m]
    return dict(
        n_trafo=int(m.sum()),
        agirlikli_ort=float(np.sum(w * bb) / w.sum()),
        duz_ort=float(np.mean(bb)),
        std=float(np.std(bb)),
        medyan=float(np.median(bb)),
        pozitif_pay=float(np.mean(bb > 0)),
    )


B_OZ_9 = [
    "log_guc",
    "egim_90_180",
    "egim_30_90",
    "seviye_90",
    "seviye_uzun",
    "omur",
    "bayat",
    "sifir_oran",
]  # + ilce (kategorik) = 9

LGB_B = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=40,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=300,
    verbose=-1,
    n_jobs=-1,
    seed=RASGELE,
)


def b_ozellik_matrisi(R, kat, kat_ad, sade):
    oz_ad = R["oz_ad"]
    if sade:
        ix = [oz_ad.index(a) for a in B_OZ_9]
        X = R["Xt"][:, ix]
        ad = list(B_OZ_9)
        kix = [kat_ad.index("ilce")]
        X = np.hstack([X, kat[:, kix].astype(np.float32)])
        ad += ["ilce"]
        kat_ix = [len(ad) - 1]
    else:
        X = np.hstack([R["Xt"], kat.astype(np.float32)])
        ad = list(oz_ad) + list(kat_ad)
        kat_ix = list(range(len(oz_ad), len(ad)))
    return X, ad, kat_ix


def b_modeli_egit(X, ad, kat_ix, b, say, maske):
    m = maske & (say > 0)
    mdl = lgb.LGBMRegressor(**LGB_B)
    mdl.fit(
        X[m], b[m], sample_weight=say[m].astype(float), feature_name=ad, categorical_feature=kat_ix
    )
    return mdl


def kazanc(gercek, tahmin, duzeltme, alfa):
    e0 = gercek - tahmin
    e1 = e0 - alfa * duzeltme
    return float(np.mean(e0**2) - np.mean(e1**2))


def kirpma_tablosu(R, duzeltme_satir, alfa, n_t, Klar=(0, 1, 5, 10, 25, 50)):
    e0 = R["gercek"] - R["tahmin"]
    e1 = e0 - alfa * duzeltme_satir
    d = e0**2 - e1**2  # satir bazli kazanc
    ti = R["ti"]
    kat_top = np.bincount(ti, weights=d, minlength=n_t)
    say = np.bincount(ti, minlength=n_t)
    sira = np.argsort(-kat_top)
    top_d, top_n = d.sum(), float(len(d))
    out = {}
    at_d, at_n = 0.0, 0.0
    onceki = 0
    for K in Klar:
        for j in range(onceki, K):
            t = sira[j]
            at_d += kat_top[t]
            at_n += say[t]
        onceki = K
        kalan_n = top_n - at_n
        out[K] = float((top_d - at_d) / kalan_n) if kalan_n > 0 else float("nan")
    return out


# ----------------------------------------------------------------- ana akis


def main():
    t0 = time.time()
    d = veri_yukle()
    P = panel_kur(d)
    kat, kat_ad = kategoriler(P)
    C = kumulatifler(P)
    ofs_tum = d["ofs"].to_numpy(dtype=np.float64)
    n_t = P["n_t"]
    print(f"[veri] {len(d):,} satir  {n_t} trafo  {P['n_g']} gun  ({time.time() - t0:.0f}s)")

    KESMELER = ["2025-03-31", "2025-05-31", "2025-07-31", "2025-09-30", "2025-11-30"]
    R = {}
    for ks in KESMELER:
        C_ts = pd.Timestamp(ks)
        onb = CIKTI / f"kesme_{ks}.npz"
        if onb.exists():
            z = np.load(onb, allow_pickle=True)
            r = dict(
                k=int(z["k"]),
                C=C_ts,
                Xt=z["Xt"],
                oz_ad=list(z["oz_ad"]),
                sicak=z["sicak"],
                ti=z["ti"],
                gi=z["gi"],
                tahmin=z["tahmin"],
                gercek=z["gercek"],
                n_egitim=int(z["n_egitim"]),
                bloklar=[(pd.Timestamp(b), None, None, None) for b in z["blok_kesme"]],
                kirpik=bool(z["kirpik"]),
                sure=0.0,
                mse=float(z["mse"]),
            )
        else:
            r = kesme_kos(P, C, kat, kat_ad, C_ts, ofs_tum)
            np.savez_compressed(
                onb,
                k=r["k"],
                Xt=r["Xt"],
                oz_ad=np.array(r["oz_ad"]),
                sicak=r["sicak"],
                ti=r["ti"],
                gi=r["gi"],
                tahmin=r["tahmin"],
                gercek=r["gercek"],
                n_egitim=r["n_egitim"],
                mse=r["mse"],
                kirpik=r["kirpik"],
                blok_kesme=np.array([str(b[0].date()) for b in r["bloklar"]]),
            )
        R[ks] = r
        bl = ", ".join(b[0].strftime("%m-%d") for b in r["bloklar"])
        print(
            f"[kesme {ks}] egitim {r['n_egitim']:,} satir ({len(r['bloklar'])} blok: {bl})"
            f"{'  <<KIRPIK EGITIM>>' if r['kirpik'] else ''}"
        )
        print(
            f"    hedef {len(r['gercek']):,} satir  sicak trafo {int(r['sicak'].sum())}"
            f"  vekil MSE {r['mse']:.5f}  ({r['sure']:.0f}s)"
        )

    # ---- (c) b_i ozetleri
    print("\n=== (c) KESME BAZINDA TRAFO YANLILIGI b_i ===")
    print(
        f"{'kesme':<12}{'n_trafo':>9}{'agir.ort':>10}{'duz.ort':>9}{'std':>8}"
        f"{'medyan':>9}{'poz%':>7}{'TAVAN':>9}{'SABIT':>9}"
    )
    B, SAY, OZ = {}, {}, {}
    for ks in KESMELER:
        r = R[ks]
        b, say = b_tablosu(r, n_t)
        B[ks], SAY[ks] = b, say
        oz = b_ozet(b, say)
        OZ[ks] = oz
        bs = np.nan_to_num(b)
        tavan = kazanc(r["gercek"], r["tahmin"], bs[r["ti"]], 1.0)
        sabit = kazanc(r["gercek"], r["tahmin"], np.full(len(r["ti"]), oz["agirlikli_ort"]), 1.0)
        oz["tavan"], oz["sabit"] = tavan, sabit
        print(
            f"{ks:<12}{oz['n_trafo']:>9}{oz['agirlikli_ort']:>10.4f}"
            f"{oz['duz_ort']:>9.4f}{oz['std']:>8.4f}{oz['medyan']:>9.4f}"
            f"{oz['pozitif_pay'] * 100:>7.1f}{tavan:>9.5f}{sabit:>9.5f}"
        )

    # ---- kesme ici sizintisiz (GroupKFold) referans
    print("\n=== KESME ICI SIZINTISIZ (5 kat GroupKFold trafo) -- referans ===")
    print(f"{'kesme':<12}{'sade9_kazanc':>14}{'tam_kazanc':>12}{'sabit':>10}")
    IC = {}
    for ks in KESMELER:
        r = R[ks]
        b, say = b_tablosu(r, n_t)
        satir = {}
        for sade in (True, False):
            X, ad, kix = b_ozellik_matrisi(r, kat, kat_ad, sade)
            gecerli = np.flatnonzero(say > 0)
            bhat = np.full(n_t, 0.0)
            gkf = GroupKFold(n_splits=5)
            for tr, te in gkf.split(gecerli, groups=gecerli):
                mtr = np.zeros(n_t, bool)
                mtr[gecerli[tr]] = True
                mdl = b_modeli_egit(X, ad, kix, b, say, mtr)
                bhat[gecerli[te]] = mdl.predict(X[gecerli[te]])
            satir["sade" if sade else "tam"] = kazanc(r["gercek"], r["tahmin"], bhat[r["ti"]], 1.0)
        IC[ks] = satir
        print(f"{ks:<12}{satir['sade']:>14.5f}{satir['tam']:>12.5f}{OZ[ks]['sabit']:>10.5f}")

    # ---- (d) ASIL TEST: kesmeler arasi transfer
    ALFALAR = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("\n=== (d) KESMELER ARASI TRANSFER (C1'de uydur -> C2'ye uygula) ===")
    sonuc = []
    for sade in (True, False):
        etiket = "SADE-9" if sade else "TAM"
        print(f"\n--- oznitelik seti: {etiket} ---")
        print(
            f"{'C1(uydur)':<12}{'C2(sina)':<12}{'kor':>7}"
            + "".join(f"{'a=' + str(a):>9}" for a in ALFALAR)
            + f"{'SABIT@1':>10}{'GECTI?':>8}"
        )
        for k1 in KESMELER:
            r1 = R[k1]
            X1, ad, kix = b_ozellik_matrisi(r1, kat, kat_ad, sade)
            mtr = SAY[k1] > 0
            mdl = b_modeli_egit(X1, ad, kix, B[k1], SAY[k1], mtr)
            delta1 = OZ[k1]["agirlikli_ort"]
            for k2 in KESMELER:
                if k2 == k1:
                    continue
                r2 = R[k2]
                X2, _, _ = b_ozellik_matrisi(r2, kat, kat_ad, sade)
                bhat = mdl.predict(X2)
                gec = SAY[k2] > 0
                kor = float(np.corrcoef(bhat[gec], B[k2][gec])[0, 1])
                satir_duz = bhat[r2["ti"]]
                kz = [kazanc(r2["gercek"], r2["tahmin"], satir_duz, a) for a in ALFALAR]
                sabit_kz = kazanc(r2["gercek"], r2["tahmin"], np.full(len(r2["ti"]), delta1), 1.0)
                en_iyi = max(kz)
                gecti = "EVET" if en_iyi > sabit_kz else "hayir"
                print(
                    f"{k1:<12}{k2:<12}{kor:>7.3f}"
                    + "".join(f"{v:>9.5f}" for v in kz)
                    + f"{sabit_kz:>10.5f}{gecti:>8}"
                )
                sonuc.append(
                    dict(
                        set=etiket,
                        C1=k1,
                        C2=k2,
                        kor=kor,
                        kazanc=dict(zip(map(str, ALFALAR), kz)),
                        sabit=sabit_kz,
                        gecti=gecti == "EVET",
                    )
                )

    # ---- (f) kirpma tablolari (SADE-9, alfa=1 ve en iyi alfa)
    print("\n=== (f) KIRPMA TABLOSU -- en buyuk K trafo atilinca satir basi kazanc ===")
    print("(SADE-9 kestirici, alfa=1,0)")
    print(f"{'C1':<12}{'C2':<12}" + "".join(f"{'K=' + str(K):>10}" for K in (0, 1, 5, 10, 25, 50)))
    for k1 in KESMELER:
        r1 = R[k1]
        X1, ad, kix = b_ozellik_matrisi(r1, kat, kat_ad, True)
        mdl = b_modeli_egit(X1, ad, kix, B[k1], SAY[k1], SAY[k1] > 0)
        for k2 in KESMELER:
            if k2 == k1:
                continue
            r2 = R[k2]
            X2, _, _ = b_ozellik_matrisi(r2, kat, kat_ad, True)
            bhat = mdl.predict(X2)
            tab = kirpma_tablosu(r2, bhat[r2["ti"]], 1.0, n_t)
            print(f"{k1:<12}{k2:<12}" + "".join(f"{tab[K]:>10.5f}" for K in (0, 1, 5, 10, 25, 50)))

    # ---- sabit delta: kesmeler arasi tutarlilik
    print("\n=== SABIT DELTA -- kesmeler arasi ===")
    deltalar = np.array([OZ[k]["agirlikli_ort"] for k in KESMELER])
    print("  kesme deltalari:", ", ".join(f"{k}={v:+.4f}" for k, v in zip(KESMELER, deltalar)))
    print(
        f"  ortalama {deltalar.mean():+.4f}  std {deltalar.std(ddof=1):.4f}  "
        f"medyan {np.median(deltalar):+.4f}"
    )
    print(f"  yaz-ikizi (2025-03-31) deltasi: {OZ['2025-03-31']['agirlikli_ort']:+.4f}")
    print("\n  -- her C1 deltasi her C2'de: satir basi kazanc --")
    print(f"{'delta kaynagi':<16}" + "".join(f"{k:>13}" for k in KESMELER))
    for k1 in KESMELER:
        dd = OZ[k1]["agirlikli_ort"]
        hucre = []
        for k2 in KESMELER:
            r2 = R[k2]
            hucre.append(kazanc(r2["gercek"], r2["tahmin"], np.full(len(r2["ti"]), dd), 1.0))
        print(f"{k1:<16}" + "".join(f"{v:>13.5f}" for v in hucre))

    with open(CIKTI / "sonuc.json", "w", encoding="utf-8") as f:
        json.dump(
            dict(
                ozet={k: OZ[k] for k in KESMELER},
                ic=IC,
                transfer=sonuc,
                deltalar=dict(zip(KESMELER, deltalar.tolist())),
            ),
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"\n[bitti] {time.time() - t0:.0f}s  -> {CIKTI / 'sonuc.json'}")


if __name__ == "__main__":
    main()
