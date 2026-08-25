# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""IDDIA CURUTME: "sicak hatanin yarisindan fazlasi TRAFO SEVIYESI ve o seviye tasinmiyor".

Dort saldiri:
  A) KIRPILMIS AYRISIM  -- TRAFO payi en buyuk K trafo atilinca ayakta kaliyor mu?
  B) GURULTU DUZELTMESI -- a_i'nin plug-in karesi kestirim gurultusunu tasiyor.
     Bolunmus-yari capraz carpimi yansiz. Rastgele bolme (ust sinir) + ZAMAN
     bolmesi (kalici/tasinabilir kisim).
  C) SEYRELME DUZELTMESI -- bloklar arasi kor, guvenilirlikle bolununce ne oluyor?
  D) ILERI YONLU TASIMA  -- blogun ILK yarisindan kestirilen a_i, IKINCI yarisina
     uygulanir. Uretimde mevcut olan tek protokol budur (kis26 testten 1 gun once
     bitiyor). Bloklar arasi 4-12 aylik sicrama degil.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_sicak_artik as sa  # noqa: E402
import olcut  # noqa: E402
import tuketim_model as tm  # noqa: E402

TOHUMLAR = (1000, 1001, 1002)


def wmean(x, w):
    return float(np.dot(w, x) / w.sum())


def ayrisim(e, w, trafo, gun):
    """artik11 ile birebir ayni sirali ayrisim. Doner: (mse, paylar, a, bd, eps)."""
    ws = pd.Series(w)
    mu = wmean(e, w)
    e0 = pd.Series(e - mu)
    num = (e0 * ws).groupby(trafo).transform("sum")
    den = ws.groupby(trafo).transform("sum")
    a = num / den
    r1 = e0 - a
    num2 = (r1 * ws).groupby(gun).transform("sum")
    den2 = ws.groupby(gun).transform("sum")
    bd = num2 / den2
    eps = r1 - bd
    mse = wmean(e**2, w)
    pay = lambda x: wmean(np.asarray(x) ** 2, w) / mse * 100.0  # noqa: E731
    return mse, (mu**2 / mse * 100.0, pay(a), pay(bd), pay(eps)), a.to_numpy()


def wgrup_ort(x, w, kod):
    """Grup bazinda agirlikli ortalama -> Series(index=grup)."""
    s = pd.Series(np.asarray(x) * w).groupby(kod).sum()
    t = pd.Series(w).groupby(kod).sum()
    return s / t


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    rng = np.random.default_rng(7)

    V = {}
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        v["trafo"] = pd.Series(dg["tanim"].to_numpy())
        v["gun"] = pd.Series(pd.to_datetime(dg["tarih"]).values.astype("datetime64[D]"))
        v["w"], v["tani"] = olcut.test_agirliklari(dg, tsicak, gk)
        v["e"] = np.asarray(v["g"] - v["r"], dtype="float64")
        V[b.ad] = v

    # ------------------------------------------------------------------ A
    print("=" * 78)
    print("A) KIRPILMIS AYRISIM  (ayni w, yalnizca satir kisitlamasi)")
    print(
        f"  {'blok':7}{'K':>4}{'kalan_tr':>9}{'MSE':>9}{'sabit':>7}{'TRAFO':>7}{'GUN':>7}{'ETKIL':>7}"
    )
    for b in tm.BLOKLAR:
        v = V[b.ad]
        w, e, tr, gn = v["w"], v["e"], v["trafo"], v["gun"]
        pay = pd.Series((e**2) * w).groupby(tr).sum().sort_values(ascending=False)
        srt = pay.index.to_numpy()
        for K in (0, 1, 5, 10, 25, 50, 100):
            msk = ~tr.isin(set(srt[:K])).to_numpy()
            mse, p, _ = ayrisim(
                e[msk], w[msk], tr[msk].reset_index(drop=True), gn[msk].reset_index(drop=True)
            )
            print(
                f"  {b.ad:7}{K:4d}{int(tr[msk].nunique()):9,}{mse:9.5f}"
                f"{p[0]:7.1f}{p[1]:7.1f}{p[2]:7.1f}{p[3]:7.1f}"
            )
        # agirliksiz kontrol
        mse, p, _ = ayrisim(e, np.ones(len(e)), tr, gn)
        print(
            f"  {b.ad:7}{'DUZ':>4}{int(tr.nunique()):9,}{mse:9.5f}"
            f"{p[0]:7.1f}{p[1]:7.1f}{p[2]:7.1f}{p[3]:7.1f}"
        )

    # ------------------------------------------------------------------ B
    print("=" * 78)
    print("B) GURULTU DUZELTMESI -- plug-in a_i^2 vs bolunmus-yari capraz carpimi")
    print("   rastgele: gunler rastgele ikiye; zaman: blogun ilk/ikinci yarisi")
    print(
        f"  {'blok':7}{'plugin%':>9}{'rastgele%':>11}{'zaman%':>9}{'guv_ras':>9}{'guv_zam':>9}{'kapsam':>8}"
    )
    GUV = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        w, e, tr, gn = v["w"], v["e"], v["trafo"], v["gun"]
        mu = wmean(e, w)
        e0 = e - mu
        mse = wmean(e**2, w)
        gunler = np.sort(gn.unique())
        orta = gunler[len(gunler) // 2]
        bolmeler = {
            "rastgele": rng.integers(0, 2, len(e)).astype(bool),
            "zaman": (gn.to_numpy() > orta),
        }
        sonuc = {}
        kapsam = None
        for ad, h2 in bolmeler.items():
            a1 = wgrup_ort(e0[~h2], w[~h2], tr[~h2].reset_index(drop=True))
            a2 = wgrup_ort(e0[h2], w[h2], tr[h2].reset_index(drop=True))
            c1 = tr.map(a1).to_numpy(dtype="float64")
            c2 = tr.map(a2).to_numpy(dtype="float64")
            ok = np.isfinite(c1) & np.isfinite(c2)
            # ayni satir altkumesinde plug-in
            a_tam = wgrup_ort(e0, w, tr)
            ct = tr.map(a_tam).to_numpy(dtype="float64")
            pi = wmean(ct[ok] ** 2, w[ok]) / mse * 100.0
            hn = wmean(c1[ok] * c2[ok], w[ok]) / mse * 100.0
            sonuc[ad] = (pi, hn)
            kapsam = float(w[ok].sum() / w.sum())
        pi_r, hn_r = sonuc["rastgele"]
        pi_z, hn_z = sonuc["zaman"]
        GUV[b.ad] = (hn_r / pi_r, hn_z / pi_z)
        print(
            f"  {b.ad:7}{pi_r:9.1f}{hn_r:11.1f}{hn_z:9.1f}"
            f"{hn_r / pi_r:9.3f}{hn_z / pi_z:9.3f}{kapsam * 100:8.1f}"
        )

    # ------------------------------------------------------------------ C
    print("=" * 78)
    print("C) BLOKLAR ARASI KOR -- ham ve SEYRELME DUZELTILMIS (zaman guvenilirligi)")
    A = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        A[b.ad] = wgrup_ort(v["e"] - wmean(v["e"], v["w"]), v["w"], v["trafo"])
    adlar = [b.ad for b in tm.BLOKLAR]
    for i in range(3):
        for j in range(i + 1, 3):
            x = pd.concat([A[adlar[i]], A[adlar[j]]], axis=1, join="inner").dropna()
            x.columns = ["a", "b"]
            r = float(x["a"].corr(x["b"]))
            gi, gj = GUV[adlar[i]][1], GUV[adlar[j]][1]
            duz = r / np.sqrt(max(gi, 1e-9) * max(gj, 1e-9)) if gi > 0 and gj > 0 else float("nan")
            print(
                f"  {adlar[i]:6} x {adlar[j]:6}  n={len(x):,}  ham kor {r:+.3f}"
                f"  guv {gi:+.3f}/{gj:+.3f}  duzeltilmis {duz:+.3f}"
                f"  OLS {np.polyfit(x['a'], x['b'], 1)[0]:+.3f}"
            )

    # ------------------------------------------------------------------ D
    print("=" * 78)
    print("D) ILERI YONLU TASIMA -- a_i blogun ILK yarisindan, IKINCI yarisina uygulanir")
    LAM = (0.10, 0.20, 0.30, 0.50, 1.00)
    kayit = {la: [] for la in LAM}
    kayit["EB"] = []
    bilgi = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        dg = v["cerceve"]
        gn = v["gun"]
        tr = v["trafo"]
        gunler = np.sort(gn.unique())
        orta = gunler[len(gunler) // 2]
        est = gn.to_numpy() <= orta
        uyg = ~est
        alt = dg[uyg].reset_index(drop=True)
        w, tani = olcut.test_agirliklari(alt, tsicak, gk)
        yy = v["y"][uyg]
        lg = v["lg"][uyg]
        satir = []
        for t_i, tohum in enumerate(TOHUMLAR):
            log_t = v["tohum_loglari"][t_i]
            r_all = log_t - v["lg"]
            e_all = np.asarray(v["g"] - r_all, dtype="float64")
            # kestirim: agirliksiz trafo ortalamasi + ampirik-Bayes buzme
            ee = pd.Series(e_all[est])
            grp = tr[est].reset_index(drop=True)
            n_i = ee.groupby(grp).size()
            m_i = ee.groupby(grp).mean()
            ici = float(ee.groupby(grp).transform("mean").sub(ee).pow(2).mean())
            arasi = max(float(m_i.var(ddof=1)) - ici / float(n_i.mean()), 1e-6)
            M = ici / arasi
            eb = (n_i / (n_i + M)) * m_i
            ham = m_i
            rr = r_all[uyg]
            taban = olcut.agirlikli_rmsle(yy, np.expm1(lg + rr), w)
            for etiket, kaynak in [(la, ham * la) for la in LAM] + [("EB", eb)]:
                dai = tr[uyg].reset_index(drop=True).map(kaynak).fillna(0.0).to_numpy()
                yeni = olcut.agirlikli_rmsle(yy, np.expm1(lg + rr + dai), w)
                kayit[etiket].append(taban - yeni)  # pozitif = KAZANC
            satir.append((tohum, taban, M))
        bilgi[b.ad] = (int(uyg.sum()), tani["ess_orani"], satir[0][2], satir)
        print(
            f"  {b.ad}: uygulama satiri {int(uyg.sum()):,}  ESS {tani['ess_orani']:.2f}"
            f"  M(EB) {satir[0][2]:.1f}  taban(ag) {satir[0][1]:.5f}"
        )
    print(f"\n  {'lambda':>8}{'ort kazanc':>12}{'SH':>10}{'t':>8}{'pozitif':>9}")
    for etiket in list(LAM) + ["EB"]:
        v_ = np.array(kayit[etiket])
        sh = v_.std(ddof=1) / np.sqrt(len(v_))
        print(
            f"  {str(etiket):>8}{v_.mean():+12.5f}{sh:10.5f}{v_.mean() / sh:+8.2f}"
            f"{int((v_ > 0).sum()):>6}/{len(v_)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
