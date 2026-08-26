# ruff: noqa
"""H1 EK -- URETIM GERCEKCI karsilastirma: uniform c=1,335 vs iki/uc bant.

``eksen6_gun_bant.py`` ORAKUL tavanini olctu (her blok kendi etiketli
optimumunu kullaniyor). Bu betik URETIMIN gercekten yapabilecegi seyi olcer:
c degerleri ETIKETSIZ CAPADAN gelir, dogrulama bloklarinda uygulanir.

UC BILESEN (fizik ayrimi)
-------------------------
    b_gun  =  h_gun (HAFTALIK: haftagunu ortalamalari, 7 sdf)
            + s_gun (DUSUK FREK: b-h'nin merkezli MA-W'si)
            + e_gun (KALAN: hava kaynakli duzensiz salinim)

Neden bu ayrim: her bilesenin ETIKETSIZ CAPASI FARKLI hizalama ister.
  * s_gun  -> gun-of-year hizasi (2025 Nis-Tem gercek vs 2026 Nis-Tem tahmin)
  * h_gun  -> HAFTAGUNU hizasi (gun-of-year hizasi 2025->2026'da haftayi
              bir gun kaydirir, bu yuzden capa korelasyonu sifira coker)
  * e_gun  -> hicbir hizalama yok (yil-disi hava); capa YOK, c=1 birakilir.

    python scripts/eksen6_gun_bant_uretim.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")
AGIRLIK = (3.0, 1.0, 1.0, 1.4)
ETIKET = "uretim"
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907
LB_RMSLE = 1.01591
W_MA = 15
C_URETIM = 1.335
#: LB 2026-08-25: formul %11 yuksekti (1,492 tahmin / 1,332 cozulen).
LB_KALIBRE = 0.893
CAPA_BASI, CAPA_SONU = "2025-04-01", "2025-07-31"


def blend(blok: str, tohum: int) -> np.ndarray:
    pay = sum(AGIRLIK)
    return (
        sum(
            w * np.load(DIZIN / f"{blok}_{tohum}_{a}_{ETIKET}.npy").astype("float64")
            for a, w in zip(AILELER, AGIRLIK, strict=True)
        )
        / pay
    )


def agirlikli_mse(e: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(w, e * e) / w.sum())


def gun_etkisi_kod(trafo_kod, gun_kod, r, n_gun):  # noqa: ANN001
    nt = np.bincount(trafo_kod)
    mt = np.bincount(trafo_kod, weights=r) / nt
    c = r - mt[trafo_kod]
    nd = np.bincount(gun_kod, minlength=n_gun).astype("float64")
    b = np.bincount(gun_kod, weights=c, minlength=n_gun) / np.maximum(nd, 1.0)
    return b - b.mean()


def uc_bilesen(b: np.ndarray, hafta: np.ndarray, W: int = W_MA):
    """b -> (h haftalik, s dusuk frekans, e kalan). Ucu de merkezli, toplami b."""
    h = np.zeros_like(b)
    for k in range(7):
        msk = hafta == k
        if msk.any():
            h[msk] = b[msk].mean()
    h = h - h.mean()
    kalan = b - h
    s = pd.Series(kalan).rolling(W, center=True, min_periods=1).mean().to_numpy()
    s = s - s.mean()
    e = kalan - s
    return h, s, e - e.mean()


def satir_bazi(u_gun, gun_kod):  # noqa: ANN001
    x = u_gun[gun_kod]
    return x - x.mean()


def kapali_form(e, U, w):  # noqa: ANN001
    n = len(U)
    A = np.empty((n, n))
    bv = np.empty(n)
    for i in range(n):
        bv[i] = float(np.dot(w, e * U[i]))
        for j in range(n):
            A[i, j] = float(np.dot(w, U[i] * U[j]))
    return np.linalg.solve(A, bv)


def uygula(r, U, a):  # noqa: ANN001
    out = r.copy()
    for u, ak in zip(U, a, strict=True):
        out = out + ak * u
    return out


# ------------------------------------------------------------------ CAPA


def capa_bilesenleri(sub_ad: str):
    """2025 Nis-Tem GERCEK vs 2026 Nis-Tem TAHMIN -- bilesen bazinda ETIKETSIZ c."""
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    sicak_kume = set(tr["tanim"].unique())

    g = tr[(tr["tarih"] >= CAPA_BASI) & (tr["tarih"] <= CAPA_SONU) & (tr["tuketim"] > 0)]
    x = pd.DataFrame({"t": g["tanim"].to_numpy(), "gg": g["tarih"].to_numpy()})
    tam = x.groupby("t")["gg"].nunique()
    tam = set(tam[tam >= 0.9 * x["gg"].nunique()].index)
    gs = g[np.isin(g["tanim"].to_numpy(), list(tam))]
    rg = np.log1p(gs["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        gs["guc"].to_numpy(dtype="float64")
    )
    tk, _ = pd.factorize(gs["tanim"].to_numpy())
    gk, gun_g = pd.factorize(gs["tarih"].to_numpy(), sort=True)
    b_g = gun_etkisi_kod(tk.astype("int64"), gk.astype("int64"), rg, len(gun_g))
    dt_g = pd.to_datetime(pd.Index(gun_g))

    sub = pd.read_csv(KOK / "submissions" / sub_ad, encoding="utf-8")
    m = te.merge(sub, on="id", how="left", validate="one_to_one")
    m = m[m["tanim"].isin(sicak_kume)]
    rt = np.log1p(np.clip(m["tuketim"].to_numpy(dtype="float64"), 0.0, None)) - np.log1p(
        m["guc"].to_numpy(dtype="float64")
    )
    tk2, _ = pd.factorize(m["tanim"].to_numpy())
    gk2, gun_t = pd.factorize(m["tarih"].to_numpy(), sort=True)
    b_t = gun_etkisi_kod(tk2.astype("int64"), gk2.astype("int64"), rt, len(gun_t))
    dt_t = pd.to_datetime(pd.Index(gun_t))

    hg, sg, eg = uc_bilesen(b_g, dt_g.dayofweek.to_numpy())
    ht, st, et = uc_bilesen(b_t, dt_t.dayofweek.to_numpy())

    print(f"\n  CAPA KAYNAGI: {sub_ad}   ({len(m):,} sicak test satiri)")
    print(f"    2025 gercek: {len(dt_g)} gun, {len(tam):,} tam panel trafosu")
    print(f"    2026 tahmin: {len(dt_t)} gun")
    print(
        f"    genlik (sd)      HAFTA  gercek {hg.std():.4f} tahmin {ht.std():.4f}   "
        f"DUSUK gercek {sg.std():.4f} tahmin {st.std():.4f}   "
        f"KALAN gercek {eg.std():.4f} tahmin {et.std():.4f}"
    )

    # HAFTALIK: haftagunu hizasi
    wg = np.array([hg[dt_g.dayofweek.to_numpy() == k][0] for k in range(7)])
    wt = np.array([ht[dt_t.dayofweek.to_numpy() == k][0] for k in range(7)])
    kor_h = float(np.corrcoef(wg, wt)[0, 1])
    c_h = kor_h * wg.std() / wt.std()
    # ayni bileseni gun-of-year hizasiyla dene -- neden coktugu gorulsun
    sh_g = pd.Series(hg, index=dt_g.dayofyear)
    sh_t = pd.Series(ht, index=dt_t.dayofyear)
    ortak = sh_g.index.intersection(sh_t.index)
    kor_h_doy = float(np.corrcoef(sh_g[ortak], sh_t[ortak])[0, 1])

    # DUSUK: gun-of-year hizasi
    ss_g = pd.Series(sg, index=dt_g.dayofyear)
    ss_t = pd.Series(st, index=dt_t.dayofyear)
    kor_s = float(np.corrcoef(ss_g[ortak], ss_t[ortak])[0, 1])
    c_s = kor_s * sg.std() / st.std()

    # KALAN: gun-of-year hizasi (beklenti: sifir -> capa YOK)
    se_g = pd.Series(eg, index=dt_g.dayofyear)
    se_t = pd.Series(et, index=dt_t.dayofyear)
    kor_e = float(np.corrcoef(se_g[ortak], se_t[ortak])[0, 1])
    c_e = kor_e * eg.std() / et.std()

    print(f"\n    {'bilesen':10}{'hizalama':>14}{'kor':>9}{'oran':>9}{'c_capa':>9}")
    print(f"    {'HAFTALIK':10}{'haftagunu':>14}{kor_h:+9.3f}{wg.std() / wt.std():9.3f}{c_h:9.3f}")
    print(f"    {'':10}{'gun-of-year':>14}{kor_h_doy:+9.3f}{'':>9}{'(coker)':>9}")
    print(f"    {'DUSUK':10}{'gun-of-year':>14}{kor_s:+9.3f}{sg.std() / st.std():9.3f}{c_s:9.3f}")
    print(f"    {'KALAN':10}{'gun-of-year':>14}{kor_e:+9.3f}{eg.std() / et.std():9.3f}{c_e:9.3f}")
    print(f"    2025 haftagunu gercek (Pzt..Paz): {np.round(wg, 4)}")
    print(f"    2026 haftagunu tahmin (Pzt..Paz): {np.round(wt, 4)}")

    # TEST tahmin bileseninin VARYANS PAYLARI -- kazanc tavani buradan gelir
    tot = float(np.var(b_t))
    print(
        f"    TEST tahmin gun ekseni varyans payi: HAFTA %{100 * np.var(ht) / tot:.1f}  "
        f"DUSUK %{100 * np.var(st) / tot:.1f}  KALAN %{100 * np.var(et) / tot:.1f}"
    )
    return {"c_hafta": c_h, "c_dusuk": c_s, "c_kalan": c_e, "kor_hafta": kor_h}


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    t0 = time.time()
    print("=" * 112)
    print("H1 EK -- URETIM GERCEKCI: uniform c=1,335 vs uc bant (HAFTA/DUSUK/KALAN)")
    print("=" * 112)

    capa = capa_bilesenleri("tuketim_v50_ham30.csv")
    print("\n  NOT: v67 sampiyonda gun ekseni ZATEN 1,335 ile carpilmistir; capa v50 ham")
    print("       uzerinden alinir (son_islem_gunolcek.py'nin kendi capa kaynagi).")

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    V: dict[str, dict] = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        dg = dogrulama[~soguk]
        w, tani = ol.test_agirliklari(dg, te_s, guc_kenar, eksenler=("bayatlik",))
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        gun = pd.to_datetime(dg["tarih"])
        gun_kod, gunler = pd.factorize(gun, sort=True)
        trafo_kod, _ = pd.factorize(dg["tanim"].to_numpy())
        V[b.ad] = {
            "w": w,
            "g": np.log1p(np.clip(gercek[~soguk], 0.0, None)) - lg,
            "r": {t: blend(b.ad, t) - lg for t in TOHUMLAR},
            "kod": gun_kod.astype("int64"),
            "trafo": trafo_kod.astype("int64"),
            "gunler": pd.DatetimeIndex(gunler),
        }
        V[b.ad]["r"]["bag"] = np.mean([blend(b.ad, t) for t in TOHUMLAR], axis=0) - lg

    def bilesenler(blok: str, tohum) -> tuple[list[np.ndarray], np.ndarray]:  # noqa: ANN001
        v = V[blok]
        ng = len(v["gunler"])
        bm = gun_etkisi_kod(v["trafo"], v["kod"], v["r"][tohum], ng)
        h, s, e = uc_bilesen(bm, v["gunler"].dayofweek.to_numpy())
        return [satir_bazi(u, v["kod"]) for u in (h, s, e)], bm

    # -------------------------------------------------- A) etiketli bilesen c'leri
    print("\n" + "-" * 112)
    print("A) ETIKETLI BILESEN OPTIMUMLARI (kapali form, es zamanli), k=3 torbali")
    print("-" * 112)
    print(
        f"  {'blok':8}{'c_hafta':>10}{'c_dusuk':>10}{'c_kalan':>10}"
        f"{'| model varyans payi  hafta / dusuk / kalan':>46}"
    )
    etiketli = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        U, bm = bilesenler(b.ad, "bag")
        e = v["g"] - v["r"]["bag"]
        a = kapali_form(e, U, v["w"])
        etiketli[b.ad] = 1.0 + a
        h, s, ee = uc_bilesen(bm, v["gunler"].dayofweek.to_numpy())
        tot = float(np.var(bm))
        print(
            f"  {b.ad:8}{1 + a[0]:10.3f}{1 + a[1]:10.3f}{1 + a[2]:10.3f}"
            f"{'':6}%{100 * np.var(h) / tot:5.1f} / %{100 * np.var(s) / tot:5.1f}"
            f" / %{100 * np.var(ee) / tot:5.1f}"
        )

    # -------------------------------------------------- B) URETIM ADAYLARI
    print("\n" + "-" * 112)
    print("B) URETIM ADAYLARI -- dRMSLE_sicak (NEGATIF = kazanc), (blok,tohum) eslenik")
    print("   TABAN = uniform c=1,335 (suanki sampiyon).  genele dMSE = dRMSLE x 0,5357 x 2R")
    print("-" * 112)

    c_h_lb = 1.0 + LB_KALIBRE * (capa["c_hafta"] - 1.0)
    c_s_lb = 1.0 + LB_KALIBRE * (capa["c_dusuk"] - 1.0)
    adaylar: dict[str, tuple[float, float, float]] = {
        "uniform 1,335 (TABAN)": (C_URETIM, C_URETIM, C_URETIM),
        "uniform 1,000 (son islem YOK)": (1.0, 1.0, 1.0),
        "capa ham (h,s,e)": (capa["c_hafta"], capa["c_dusuk"], 1.0),
        "capa LB-kalibreli": (c_h_lb, c_s_lb, 1.0),
        "dusuk=1,335 hafta=1 kalan=1": (1.0, C_URETIM, 1.0),
        "dusuk=1,335 hafta=1,335 kalan=1": (C_URETIM, C_URETIM, 1.0),
        "dusuk=1,55 hafta=1,0 kalan=1": (1.0, 1.55, 1.0),
        "dusuk=1,55 hafta=1,335 kalan=1": (C_URETIM, 1.55, 1.0),
    }
    print(f"  {'aday':34}{'c_h':>7}{'c_s':>7}{'c_e':>7}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print(f"{'HAVUZ':>11}{'SH':>10}{'t':>7}{'iyi/N':>8}{'genele dMSE':>14}")

    taban_skor: dict[tuple[str, int], float] = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            U, _ = bilesenler(b.ad, t)
            rr = uygula(v["r"][t], U, np.array([C_URETIM - 1.0] * 3))
            taban_skor[(b.ad, t)] = np.sqrt(agirlikli_mse(v["g"] - rr, v["w"]))

    sonuc = {}
    for ad, (ch, cs, ce) in adaylar.items():
        satir, farklar = {}, []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            U, _ = bilesenler(b.ad, "bag")
            rb = uygula(v["r"]["bag"], U, np.array([ch - 1.0, cs - 1.0, ce - 1.0]))
            r0 = uygula(v["r"]["bag"], U, np.array([C_URETIM - 1.0] * 3))
            satir[b.ad] = np.sqrt(agirlikli_mse(v["g"] - rb, v["w"])) - np.sqrt(
                agirlikli_mse(v["g"] - r0, v["w"])
            )
            for t in TOHUMLAR:
                Ut, _ = bilesenler(b.ad, t)
                rt = uygula(v["r"][t], Ut, np.array([ch - 1.0, cs - 1.0, ce - 1.0]))
                farklar.append(np.sqrt(agirlikli_mse(v["g"] - rt, v["w"])) - taban_skor[(b.ad, t)])
        f = np.array(farklar)
        sh = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0
        tt = float(f.mean() / sh) if sh > 0 else 0.0
        dg = 2 * LB_RMSLE * f.mean() * SICAK_KATSAYI
        sonuc[ad] = {"blok": satir, "ort": float(f.mean()), "sh": sh, "t": tt, "dmse": dg}
        print(f"  {ad:34}{ch:7.3f}{cs:7.3f}{ce:7.3f}", end="")
        for b in tm.BLOKLAR:
            print(f"{satir[b.ad]:+11.5f}", end="")
        print(f"{f.mean():+11.5f}{sh:10.5f}{tt:+7.2f}{int((f < 0).sum()):>4}/{len(f)}{dg:+14.6f}")

    # -------------------------------------------------- C) yaz25 c_s taramasi
    print("\n" + "-" * 112)
    print("C) yaz25 (MEVSIMSEL IKIZ) c_dusuk taramasi, c_hafta=1,335 / c_kalan=1,0 sabit")
    print("   TABAN yine uniform 1,335.  Sadece IKIZ blok -- gecirgenlik iddiasi DEGIL.")
    print("-" * 112)
    v = V["yaz25"]
    U, _ = bilesenler("yaz25", "bag")
    r0 = uygula(v["r"]["bag"], U, np.array([C_URETIM - 1.0] * 3))
    s0 = np.sqrt(agirlikli_mse(v["g"] - r0, v["w"]))
    print(f"  {'c_dusuk':>9}{'yaz25 dRMSLE':>15}{'guz25 dRMSLE':>15}")
    for cs in (1.0, 1.2, 1.335, 1.5, 1.6, 1.75, 2.0, 2.5, 3.0):
        satir = []
        for b in ("yaz25", "guz25"):
            vv = V[b]
            UU, _ = bilesenler(b, "bag")
            rb = uygula(vv["r"]["bag"], UU, np.array([C_URETIM - 1.0, cs - 1.0, 0.0]))
            rz = uygula(vv["r"]["bag"], UU, np.array([C_URETIM - 1.0] * 3))
            satir.append(
                np.sqrt(agirlikli_mse(vv["g"] - rb, vv["w"]))
                - np.sqrt(agirlikli_mse(vv["g"] - rz, vv["w"]))
            )
        print(f"  {cs:9.3f}{satir[0]:+15.5f}{satir[1]:+15.5f}")
    _ = s0

    print(f"\n  toplam sure {time.time() - t0:.0f} sn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
