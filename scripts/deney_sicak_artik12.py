# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM K: DURUST ileri kol, seyreltmesiz (yalniz duzeltilen satirlarda olcum)
+ trafo bazinda ayrisim."""

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
from deney_sicak_artik7 import z_kur  # noqa: E402

RNG = np.random.default_rng(11)


def main() -> int:
    import lightgbm as lgb

    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        z = z_kur(dg)
        e = (
            (v["g"] - v["r"]).to_numpy()
            if hasattr(v["g"] - v["r"], "to_numpy")
            else (v["g"] - v["r"])
        )
        trafo = dg["tanim"].to_numpy()
        gun = pd.to_datetime(dg["tarih"])
        orta = gun.min() + (gun.max() - gun.min()) / 2
        ikinci = (gun > orta).to_numpy()
        tk = pd.unique(trafo)
        h = pd.Series(trafo).map(pd.Series(RNG.integers(0, 2, len(tk)), index=tk)).to_numpy()
        egit = (~ikinci) & (h == 0)
        olc = ikinci & (h == 1)
        m = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=0,
        )
        m.fit(z[egit], e[egit])
        eh = m.predict(z[olc])
        alt = dg[olc]
        w, tani = olcut.test_agirliklari(alt, tsicak, gk)
        yy = v["y"][olc]
        lg = v["lg"][olc]
        rr = v["r"][olc]
        taban = olcut.agirlikli_rmsle(yy, np.expm1(lg + rr), w)
        print(
            f"\n=== {b.ad}  olcum satiri {int(olc.sum()):,}  trafo {pd.unique(trafo[olc]).size:,}"
            f"  ESS {tani['ess_orani']:.2f}  taban(ag) {taban:.5f}"
        )
        for a in (0.25, 0.5, 1.0):
            tah = np.expm1(lg + rr + a * eh)
            duz = olcut.agirlikli_rmsle(yy, tah)
            ag = olcut.agirlikli_rmsle(yy, tah, w)
            print(f"    a={a:4.2f}  duz {duz:.5f}  agirlikli {ag:.5f}  fark {ag - taban:+.5f}")
        # trafo bazinda d(MSE) ayrisimi, a=0,5
        a = 0.5
        e0 = (np.log1p(np.clip(yy, 0, None)) - (lg + rr)) ** 2
        e1 = (np.log1p(np.clip(yy, 0, None)) - (lg + rr + a * eh)) ** 2
        dd = (
            pd.Series((e0 - e1) * w)
            .groupby(pd.Series(trafo[olc]))
            .sum()
            .sort_values(ascending=False)
        )
        p = (dd / dd.sum()).to_numpy()
        print(
            f"    a=0,50 d(MSE) toplam {dd.sum():.1f}  EN BUYUK %{p[0] * 100:.1f}  ilk5 %{p[:5].sum() * 100:.1f}"
        )
        # KIRPILMIS: en buyuk K trafo atilarak
        srt = dd.index.to_numpy()
        for K in (0, 1, 5, 10, 25, 50):
            at = set(srt[:K])
            msk = ~pd.Series(trafo[olc]).isin(at).to_numpy()
            t0 = olcut.agirlikli_rmsle(yy[msk], np.expm1(lg[msk] + rr[msk]), w[msk])
            t1 = olcut.agirlikli_rmsle(yy[msk], np.expm1(lg[msk] + rr[msk] + a * eh[msk]), w[msk])
            print(
                f"      K={K:3d}  kalan {int(msk.sum()):7,}  taban {t0:.5f}  yeni {t1:.5f}  fark {t1 - t0:+.5f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
