# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM G: BLOK ICI (ayni mevsim) trafo-bolunmus capraz uydurma.

Adim F blok-DISI uydurmanin cokttugunu gosterdi. Soru: sinyal hic yok mu, yoksa
mevsimler arasi TASINMIYOR mu? Ayni blokta, trafolarin yarisinda uydurup diger
yarisinda olcersek mevsim farki YOK -- yalnizca 'gorulmemis trafo' farki kalir.
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
from deney_sicak_artik7 import z_kur  # noqa: E402

RNG = np.random.default_rng(11)


def main() -> int:
    import lightgbm as lgb

    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]
    print(f"  {'blok':8}{'alfa':>6}{'duz':>10}{'agirlikli':>11}{'fark':>10}")
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        z = z_kur(dg)
        e = v["g"] - v["r"]
        trafo = dg["tanim"].to_numpy()
        tk = pd.unique(trafo)
        yari = pd.Series(RNG.integers(0, 2, len(tk)), index=tk)
        h = pd.Series(trafo).map(yari).to_numpy()
        e_hat = np.zeros(len(dg))
        for kat in (0, 1):
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
            m.fit(z[h == kat], e[h == kat])
            e_hat[h != kat] = m.predict(z[h != kat])
        w, _ = olcut.test_agirliklari(dg, tsicak, gk)
        taban = None
        for a in (0.0, 0.25, 0.5, 1.0):
            tah = np.expm1(v["lg"] + v["r"] + a * e_hat)
            duz = olcut.agirlikli_rmsle(v["y"], tah)
            ag = olcut.agirlikli_rmsle(v["y"], tah, w)
            if a == 0.0:
                taban = ag
            print(f"  {b.ad:8}{a:6.2f}{duz:10.5f}{ag:11.5f}{ag - taban:+10.5f}")
        c = float(np.corrcoef(e_hat, e)[0, 1])
        print(
            f"    kor(e_hat, e) = {c:+.4f}   R2 = {c * c * 100:.2f}%   "
            f"std(e_hat)={e_hat.std():.4f} std(e)={float(np.std(e)):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
