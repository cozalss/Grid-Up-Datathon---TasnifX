# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SOGUK SABIT SEVIYE KAYMASI -- kuvvet ortalamasi sinamasinin yan bulgusu.

kis26 sogukta c=+0,15 log birimlik SABIT yukari kayma +0,00709 veriyor
(genele -0,00265). Kalici kural 1: trafo bazinda ayristir ve kirp.
Ayrica y=0 / y>0 kirilimi: kazanc nereden geliyor?

    python scripts/deney_soguk_kayma_sabit.py
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
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
from deney_kuvvet_ekseni import SOG, SOGUK_KATSAYI, TOHUMLAR, kuvvet_ort  # noqa: E402

BETA = 0.60
C = 0.15


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    dg = dogrulama[soguk]
    y = gercek[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    tanim = dg["tanim"].astype(str).to_numpy()
    te_c = test[test["soguk_mu"] == 1]
    w, _ = ol.test_agirliklari(dg, te_c, ol.guc_kenarlari(te_c), eksenler=("guc",))
    tah = [np.load(SOG / f"kis26_{t}_taban.npy").astype("float64") for t in TOHUMLAR]

    def buz(log_t):
        r = log_t - log_guc
        return np.clip(np.expm1(r.mean() + BETA * (r - r.mean()) + log_guc), 0.0, None)

    taban_l = kuvvet_ort(tah, [1.0] * 3, 0.0)
    b0, b1 = buz(taban_l), buz(taban_l + C)
    g = np.log1p(np.clip(y, 0, None))
    a0, a1 = np.log1p(b0), np.log1p(b1)
    dm = w * ((g - a0) ** 2 - (g - a1) ** 2)

    # TOHUM BAZINDA (her tohum ayri tahminci)
    tekil = np.array(
        [
            ol.agirlikli_rmsle(y, buz(tah[i]), w) - ol.agirlikli_rmsle(y, buz(tah[i] + C), w)
            for i in range(3)
        ]
    )
    print(f"  tohum bazinda fark: {tekil}  ort {tekil.mean():+.5f}  {int((tekil > 0).sum())}/3")

    print("\n  y=0 / y>0 kirilimi (agirlikli d(MSE) payi)")
    sifir = y <= 0
    for ad, m in (("y=0", sifir), ("y>0", ~sifir)):
        print(
            f"    {ad:5} n={int(m.sum()):>6,}  agir.pay %{100 * w[m].sum() / w.sum():5.2f}  "
            f"d(MSE) {dm[m].sum():+9.1f}"
        )

    s_tr = pd.Series(dm).groupby(tanim).sum().sort_values(ascending=False)
    top = float(s_tr.sum())
    print(
        f"\n  trafo {len(s_tr):,}  toplam d(MSE) {top:+.1f}  en buyuk %{100 * s_tr.iloc[0] / top:.1f}"
        f"  ilk5 %{100 * s_tr.iloc[:5].sum() / top:.1f}  pozitif trafo %"
        f"{100 * (s_tr > 0).mean():.1f}"
    )
    print(f"  {'K':>4}{'fark':>11}{'genele':>10}")
    for K in (0, 1, 5, 10, 25, 50, 100):
        at = set(s_tr.index[:K])
        m = ~pd.Series(tanim).isin(at).to_numpy()
        f = ol.agirlikli_rmsle(y[m], b0[m], w[m]) - ol.agirlikli_rmsle(y[m], b1[m], w[m])
        print(f"  {K:>4}{f:+11.5f}{-f * SOGUK_KATSAYI:+10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
