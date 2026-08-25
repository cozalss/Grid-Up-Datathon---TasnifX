# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM L: iddiayi curutme denemesi.

Uc saldiri:
  A) AGIRLIKSIZ ayrisim -- 'yarisindan fazlasi TRAFO' dusuk ESS'li agirliga mi bagli?
  B) YARI-BOLME GUVENILIRLIGI -- a_i blok ICINDE ne kadar olculebiliyor?
     Yansiz TRAFO bileseni = sum_i W_i * a_i^A * a_i^B  (iki yari bagimsiz gurultu).
     Buradan sonumlemeye duzeltilmis bloklar-arasi korelasyon.
  C) TEK KAYNAKLI + BUZULMUS transfer -- iddia yalnizca 'diger IKI blogun ORTALAMASI'
     ile lambda>=0,25 denemis. Burada her (hedef,kaynak) cifti ayri, buzulmeli,
     kucuk lambda, TOHUM BAZINDA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import deney_sicak_artik as sa  # noqa: E402
import olcut  # noqa: E402
import tuketim_model as tm

TOHUMLAR = (1000, 1001, 1002)


def ayris(e: np.ndarray, w: np.ndarray, t: pd.Series, gn: pd.Series) -> dict:
    ws = pd.Series(w)
    es = pd.Series(e)
    mu = float(np.average(es, weights=w))
    e0 = es - mu
    a = (e0 * ws).groupby(t).transform("sum") / ws.groupby(t).transform("sum")
    r1 = e0 - a
    bd = (r1 * ws).groupby(gn).transform("sum") / ws.groupby(gn).transform("sum")
    eps = r1 - bd
    mse = float(np.average(es**2, weights=w))
    f = lambda x: float(np.average(x**2, weights=w)) / mse * 100  # noqa: E731
    return {
        "mse": mse,
        "sabit": mu**2 / mse * 100,
        "trafo": f(a),
        "gun": f(bd),
        "etk": f(eps),
        "e0": e0,
        "a": a,
    }


def grup_ort(e0: pd.Series, w: pd.Series, t: pd.Series, maske: np.ndarray) -> pd.Series:
    num = (e0 * w)[maske].groupby(t[maske]).sum()
    den = w[maske].groupby(t[maske]).sum()
    return num / den


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    rng = np.random.default_rng(7)

    V, W, T, G, E0, AW, AU, NI = {}, {}, {}, {}, {}, {}, {}, {}
    adlar = [b.ad for b in tm.BLOKLAR]

    print("A) AYRISIM: AGIRLIKLI vs AGIRLIKSIZ")
    print(f"  {'blok':8}{'kip':>11}{'MSE':>9}{'sabit':>8}{'TRAFO':>8}{'GUN':>8}{'ETK':>8}")
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        V[b.ad] = v
        w, tani = olcut.test_agirliklari(dg, tsicak, gk)
        t = pd.Series(dg["tanim"].to_numpy())
        gn = pd.Series(pd.to_datetime(dg["tarih"]).values.astype("datetime64[D]"))
        e = v["g"] - v["r"]
        W[b.ad], T[b.ad], G[b.ad] = w, t, gn
        for kip, ww in (("agirlikli", w), ("agirliksiz", np.ones(len(w)))):
            r = ayris(e, ww, t, gn)
            print(
                f"  {b.ad:8}{kip:>11}{r['mse']:9.5f}{r['sabit']:8.1f}{r['trafo']:8.1f}"
                f"{r['gun']:8.1f}{r['etk']:8.1f}"
            )
            if kip == "agirlikli":
                E0[b.ad] = r["e0"]
        AU[b.ad] = pd.Series(e).groupby(t).mean()
        NI[b.ad] = pd.Series(e).groupby(t).size()

    print("\nB) YARI-BOLME GUVENILIRLIGI  (R = yansiz TRAFO SS / gozlenen TRAFO SS)")
    print(
        f"  {'blok':8}{'bolme':>12}{'kip':>11}{'gozlenen':>10}{'yansiz':>10}{'R':>8}{'r_yari':>8}"
    )
    REL = {}
    for b in tm.BLOKLAR:
        t, gn, e0 = T[b.ad], G[b.ad], E0[b.ad]
        n = len(t)
        gunler = np.sort(gn.unique())
        gsira = pd.Series(gn).map({g: i for i, g in enumerate(gunler)}).to_numpy()
        bolmeler = {
            "rasgele": rng.random(n) < 0.5,
            "tek/cift gun": (gsira % 2) == 0,
            "on/arka yari": gsira < len(gunler) / 2,
        }
        for ad, mA in bolmeler.items():
            for kip, ww in (("agirlikli", W[b.ad]), ("agirliksiz", np.ones(n))):
                ws = pd.Series(ww)
                e0k = (
                    pd.Series(e0.to_numpy())
                    if kip == "agirlikli"
                    else pd.Series(e0.to_numpy() + 0.0)
                )
                aA = grup_ort(e0k, ws, t, mA)
                aB = grup_ort(e0k, ws, t, ~mA)
                Wi = ws.groupby(t).sum()
                ort = pd.concat([aA, aB, Wi], axis=1, join="inner").dropna()
                ort.columns = ["A", "B", "Wi"]
                aF = (e0k * ws).groupby(t).sum() / ws.groupby(t).sum()
                aF = aF.reindex(ort.index)
                toplam = float(ws.sum())
                goz = float((ort["Wi"] * aF**2).sum()) / toplam
                yansiz = float((ort["Wi"] * ort["A"] * ort["B"]).sum()) / toplam
                R = yansiz / goz if goz > 0 else np.nan
                mA_ = ort["A"] - np.average(ort["A"], weights=ort["Wi"])
                mB_ = ort["B"] - np.average(ort["B"], weights=ort["Wi"])
                ryari = float(
                    np.average(mA_ * mB_, weights=ort["Wi"])
                    / np.sqrt(
                        np.average(mA_**2, weights=ort["Wi"])
                        * np.average(mB_**2, weights=ort["Wi"])
                    )
                )
                print(f"  {b.ad:8}{ad:>12}{kip:>11}{goz:10.5f}{yansiz:10.5f}{R:8.3f}{ryari:8.3f}")
                if ad == "rasgele":
                    REL[(b.ad, kip)] = R
    print("  NOT: R = 1 -> a_i tam olculuyor; R < 1 -> TRAFO payi o oranda SISIRILMIS.")

    print("\n  SONUMLEMEYE DUZELTILMIS bloklar-arasi korelasyon (agirliksiz a_i, rasgele bolme R)")
    for i in range(3):
        for j in range(i + 1, 3):
            x = pd.concat([AU[adlar[i]], AU[adlar[j]]], axis=1, join="inner").dropna()
            x.columns = ["a", "b"]
            r = float(x["a"].corr(x["b"]))
            R1, R2 = REL[(adlar[i], "agirliksiz")], REL[(adlar[j], "agirliksiz")]
            print(
                f"    {adlar[i]:6} x {adlar[j]:6}  ham {r:+.3f}   R {R1:.3f}/{R2:.3f}"
                f"   duzeltilmis {r / np.sqrt(max(R1 * R2, 1e-9)):+.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
