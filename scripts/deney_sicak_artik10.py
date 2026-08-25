# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM I: TRAFO BAZLI KALICI YANLILIK bloklar arasi tasiniyor mu?

Hata ayrisimi sicak MSE'nin %47-64'unun trafo duzeyinde SABIT bir ofset
oldugunu gosterdi. Soru: o ofset MODELDEN mi (kalici trafo yanliligi) yoksa
BLOGA/MEVSIME mi ozgu? Blok-disi protokol: a_i diger iki bloktan kestirilir,
ucuncu blokta uygulanir.
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


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    V, A, N = {}, {}, {}
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        V[b.ad] = v
        e = pd.Series(v["g"] - v["r"])
        t = pd.Series(v["cerceve"]["tanim"].to_numpy())
        A[b.ad] = e.groupby(t).mean()
        N[b.ad] = e.groupby(t).size()

    print("  TRAFO YANLILIGI a_i BLOKLAR ARASI KORELASYON (ortak trafolar)")
    adlar = [b.ad for b in tm.BLOKLAR]
    for i in range(3):
        for j in range(i + 1, 3):
            x = pd.concat([A[adlar[i]], A[adlar[j]]], axis=1, join="inner").dropna()
            x.columns = ["a", "b"]
            print(
                f"    {adlar[i]:6} x {adlar[j]:6}  n={len(x):,}  kor={x['a'].corr(x['b']):+.3f}"
                f"  OLS={np.polyfit(x['a'], x['b'], 1)[0]:+.3f}"
                f"  std {x['a'].std():.3f}/{x['b'].std():.3f}"
            )

    print("\n  BLOK-DISI UYGULAMA:  tahmin' = tahmin + lambda * ort(a_i, diger iki blok)")
    print(f"  {'blok':8}{'lam':>6}{'duz':>10}{'agirlikli':>11}{'fark':>10}{'kapsam%':>9}")
    for b in tm.BLOKLAR:
        kaynak = [o for o in adlar if o != b.ad]
        ai = pd.concat([A[k] for k in kaynak], axis=1).mean(axis=1)
        v = V[b.ad]
        dg = v["cerceve"]
        d_ai = pd.Series(dg["tanim"].to_numpy()).map(ai).fillna(0.0).to_numpy()
        kapsam = float((pd.Series(dg["tanim"].to_numpy()).map(ai).notna()).mean())
        w, _ = olcut.test_agirliklari(dg, tsicak, gk)
        taban = olcut.agirlikli_rmsle(v["y"], np.expm1(v["lg"] + v["r"]), w)
        for lam in (0.0, 0.25, 0.5, 1.0):
            tah = np.expm1(v["lg"] + v["r"] + lam * d_ai)
            duz = olcut.agirlikli_rmsle(v["y"], tah)
            ag = olcut.agirlikli_rmsle(v["y"], tah, w)
            print(
                f"  {b.ad:8}{lam:6.2f}{duz:10.5f}{ag:11.5f}{ag - taban:+10.5f}{kapsam * 100:9.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
