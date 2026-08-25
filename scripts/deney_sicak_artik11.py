# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM J: TEST KARISIMINA AGIRLIKLANDIRILMIS sicak hata ayrisimi."""

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
    print(
        f"  {'blok':8}{'ESS':>6}{'MSE(ag)':>10}{'sabit':>8}{'TRAFO':>8}{'GUN':>8}{'ETKILESIM':>11}"
    )
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        w, tani = olcut.test_agirliklari(dg, tsicak, gk)
        e = pd.Series(v["g"] - v["r"])
        ws = pd.Series(w)
        mu = float(np.average(e, weights=w))
        e0 = e - mu
        t = pd.Series(dg["tanim"].to_numpy())
        gn = pd.Series(pd.to_datetime(dg["tarih"]).values.astype("datetime64[D]"))
        num = (e0 * ws).groupby(t).transform("sum")
        den = ws.groupby(t).transform("sum")
        a = num / den
        r1 = e0 - a
        num = (r1 * ws).groupby(gn).transform("sum")
        den2 = ws.groupby(gn).transform("sum")
        bd = num / den2
        eps = r1 - bd
        mse = float(np.average(e**2, weights=w))
        f = lambda x: float(np.average(x**2, weights=w)) / mse * 100  # noqa: E731
        print(
            f"  {b.ad:8}{tani['ess_orani']:6.2f}{mse:10.5f}{mu**2 / mse * 100:8.1f}{f(a):8.1f}"
            f"{f(bd):8.1f}{f(eps):11.1f}"
        )
        # trafo bazinda hata payi (agirlikli)
        pay = ((e**2) * ws).groupby(t).sum().sort_values(ascending=False)
        p = (pay / pay.sum()).to_numpy()
        print(
            f"           trafo yogunlasmasi: en buyuk %{p[0] * 100:.2f}  ilk5 %{p[:5].sum() * 100:.2f}"
            f"  ilk1%% %{p[: max(1, len(p) // 100)].sum() * 100:.1f}  ilk10%% %{p[: max(1, len(p) // 10)].sum() * 100:.1f}"
            f"   (n_trafo={len(p):,})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
