# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""IDDIA CURUTME -- 3. tur: AYRISIM SIRASI.

Iddianin TRAFO payi SIRALI bir ayrisimdan geliyor: once trafo ortalamasi, sonra
gun ortalamasi. Dengesiz panelde ilk cikarilan eksen digerinin kompozisyonunu
YUTAR. docs/41 §6e'nin kalici kural 6'si tam bunun aynasi ("gun ekseni olcumu,
trafo etkisi cikarilmadan yapilmaz"). Kural kendi uzerine uygulaniyor:

  1) trafo -> gun   (iddianin sirasi)
  2) gun -> trafo   (ters sira)
  3) iki yonlu ITERATIF (dik) ayrisim -- sira bagimsiz
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


def wmean(x, w):
    return float(np.dot(w, x) / w.sum())


def wgt(x, w, kod):
    return pd.Series(np.asarray(x) * w).groupby(kod).transform("sum") / pd.Series(w).groupby(
        kod
    ).transform("sum")


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]

    print(f"  {'blok':7}{'sira':>16}{'sabit':>8}{'TRAFO':>8}{'GUN':>8}{'ETKIL':>8}{'toplam':>9}")
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        tr = pd.Series(dg["tanim"].to_numpy())
        gn = pd.Series(pd.to_datetime(dg["tarih"]).values.astype("datetime64[D]"))
        w, _ = olcut.test_agirliklari(dg, tsicak, gk)
        e = np.asarray(v["g"] - v["r"], dtype="float64")
        mse = wmean(e**2, w)
        mu = wmean(e, w)
        e0 = pd.Series(e - mu)
        pay = lambda x: wmean(np.asarray(x) ** 2, w) / mse * 100.0  # noqa: E731
        sb = mu**2 / mse * 100.0

        # 1) trafo -> gun
        a = wgt(e0, w, tr)
        r1 = e0 - a
        bd = wgt(r1, w, gn)
        eps = r1 - bd
        print(
            f"  {b.ad:7}{'trafo->gun':>16}{sb:8.1f}{pay(a):8.1f}{pay(bd):8.1f}"
            f"{pay(eps):8.1f}{sb + pay(a) + pay(bd) + pay(eps):9.1f}"
        )

        # 2) gun -> trafo
        bd2 = wgt(e0, w, gn)
        r2 = e0 - bd2
        a2 = wgt(r2, w, tr)
        eps2 = r2 - a2
        print(
            f"  {b.ad:7}{'gun->trafo':>16}{sb:8.1f}{pay(a2):8.1f}{pay(bd2):8.1f}"
            f"{pay(eps2):8.1f}{sb + pay(a2) + pay(bd2) + pay(eps2):9.1f}"
        )

        # 3) iteratif dik ayrisim (agirlikli, degisimli izdusum)
        A = pd.Series(np.zeros(len(e0)))
        B = pd.Series(np.zeros(len(e0)))
        cur = e0.copy()
        for _ in range(60):
            da = wgt(cur, w, tr)
            A = A + da
            cur = cur - da
            db = wgt(cur, w, gn)
            B = B + db
            cur = cur - db
        print(
            f"  {b.ad:7}{'ITERATIF':>16}{sb:8.1f}{pay(A):8.1f}{pay(B):8.1f}"
            f"{pay(cur):8.1f}{sb + pay(A) + pay(B) + pay(cur):9.1f}"
        )
        # capraz terim (dik olmayan artik)
        print(
            f"  {'':7}{'':>16}  -> trafo/gun capraz kovaryans payi "
            f"{2 * wmean(np.asarray(A) * np.asarray(B), w) / mse * 100:+.1f}%   "
            f"panel doluluk {len(e0) / (tr.nunique() * gn.nunique()) * 100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
