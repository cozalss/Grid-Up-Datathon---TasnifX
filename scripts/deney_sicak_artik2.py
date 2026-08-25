# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM A: zaman yapisi (variogram / nugget) ve trafo yogunlasmasi."""

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
import tuketim_model as tm  # noqa: E402


def panel(vals, trafo, gun):
    df = pd.DataFrame({"t": trafo, "g": gun, "v": vals})
    return df.pivot(index="t", columns="g", values="v").to_numpy(dtype="float64")


def otokov(M, lags):
    out = {}
    for h in lags:
        a, b = M[:, :-h] if h else M, M[:, h:] if h else M
        m = np.isfinite(a) & np.isfinite(b)
        out[h] = (float(np.nansum(np.where(m, a * b, 0.0)) / m.sum()), int(m.sum()))
    return out


def main() -> int:
    egitim, _ = d.cerceveleri_kur()
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        trafo = dg["tanim"].to_numpy()
        gun = pd.to_datetime(dg["tarih"]).values.astype("datetime64[D]")
        ra = sa.iki_yonlu_arindir(v["r"], trafo, gun)
        ga = sa.iki_yonlu_arindir(v["g"], trafo, gun)
        ea = ga - ra
        print(f"\n=== {b.ad}  n={len(dg):,}")
        for ad, x in (("GERCEK artik", ga), ("HATA (gercek-model)", ea)):
            M = panel(x, trafo, gun)
            var = float(np.nanvar(M))
            lags = [1, 2, 3, 5, 7, 14, 21, 30, 60]
            ok = otokov(M, lags)
            r0 = float(np.nanmean(M[np.isfinite(M)] ** 2))
            s = "  ".join(f"h{h}:{ok[h][0] / r0:+.3f}" for h in lags)
            print(f"  {ad:22} var={var:.4f}  ac -> {s}")
        # gun-ici tekrarlanabilirlik yok; trafo yogunlasmasi:
        s2 = pd.Series(ea**2).groupby(trafo).sum()
        pay = (s2.sort_values(ascending=False) / s2.sum()).to_numpy()
        print(
            f"  HATA^2 yogunlasmasi: en buyuk trafo %{pay[0] * 100:.2f}  "
            f"ilk5 %{pay[:5].sum() * 100:.2f}  ilk1% %{pay[: max(1, len(pay) // 100)].sum() * 100:.1f}  "
            f"ilk10% %{pay[: max(1, len(pay) // 10)].sum() * 100:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
