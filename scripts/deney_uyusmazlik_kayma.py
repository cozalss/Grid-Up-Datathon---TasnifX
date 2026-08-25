# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""UYUSMAZLIK KAYMASI -- kuvvet ortalamasinin 2. mertebe ACILIMI.

log M_p ~ ort + (p/2) Var_w(uyeler).  Yani p taramasi aslinda
"aileler ayrisinca tahmini YUKARI it" eksenidir. Tek parametreli
(lambda) hali daha kararli; sicak tarafta lambda taramasi + blok
kirilimi + KIRPILMIS tablo.

    python scripts/deney_uyusmazlik_kayma.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402
from deney_kuvvet_ekseni import (  # noqa: E402
    AGIRLIK,
    SICAK_KATSAYI,
    TOHUMLAR,
    _skor,
    _t_tablo,
    sicak_veri,
)


def main() -> int:
    veri = sicak_veri()
    ciftler = [(b.ad, t) for b in tm.BLOKLAR for t in TOHUMLAR]
    aileler = list(AGIRLIK)
    agir = [AGIRLIK[a] for a in aileler]
    top = sum(agir)

    def parcala(bad, t):
        L = [veri[bad][(t, a)] for a in aileler]
        m = sum(w * x for w, x in zip(agir, L)) / top
        var = sum(w * (x - m) ** 2 for w, x in zip(agir, L)) / top
        return m, var

    onbellek = {c: parcala(*c) for c in ciftler}
    taban = {c: _skor(veri[c[0]], onbellek[c][0]) for c in ciftler}
    print(f"\n  {'lambda':>8}{'fark':>11}{'SH':>9}{'t':>7}{'tohum':>7}   bloklar (yaz/guz/kis)")
    for lam in (-0.20, -0.10, 0.0, 0.05, 0.10, 0.20, 0.40, 0.80):
        s = {c: _skor(veri[c[0]], onbellek[c][0] + lam * onbellek[c][1]) for c in ciftler}
        f = np.array([taban[c] - s[c] for c in ciftler])
        ort, sh, td, kaz = _t_tablo(f)
        blok = "  ".join(
            f"{np.mean([taban[(b.ad, t)] - s[(b.ad, t)] for t in TOHUMLAR]):+.5f}"
            for b in tm.BLOKLAR
        )
        print(f"  {lam:+8.2f}{ort:+11.5f}{sh:9.5f}{td:+7.2f}{kaz:5d}/9   {blok}")

    lam = 0.20
    print(f"\n  --- KIRPILMIS (lambda={lam:+.2f}) ---")
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        g = np.log1p(np.clip(v["y"], 0, None))
        dm = np.zeros(len(g))
        for t in TOHUMLAR:
            m, var = onbellek[(b.ad, t)]
            dm += v["w"] * ((g - m) ** 2 - (g - m - lam * var) ** 2)
        s_tr = pd.Series(dm).groupby(v["tanim"]).sum().sort_values(ascending=False)
        tp = float(s_tr.sum())
        print(
            f"    {b.ad}  toplam d(MSE) {tp:+.1f}  en buyuk %{100 * s_tr.iloc[0] / tp:.1f}  "
            f"ilk5 %{100 * s_tr.iloc[:5].sum() / tp:.1f}"
        )
        v["sira"] = s_tr
    print(f"    {'K':>4}{'fark':>11}{'SH':>9}{'t':>7}{'tohum':>7}{'genele':>10}")
    for K in (0, 1, 5, 10, 25, 50):
        f = []
        for bad, t in ciftler:
            v = veri[bad]
            at = set(v["sira"].index[:K])
            msk = ~pd.Series(v["tanim"]).isin(at).to_numpy()
            m, var = onbellek[(bad, t)]
            yy, ww = v["y"][msk], v["w"][msk]
            a0 = ol.agirlikli_rmsle(yy, np.clip(np.expm1(m[msk]), 0, None), ww)
            a1 = ol.agirlikli_rmsle(yy, np.clip(np.expm1((m + lam * var)[msk]), 0, None), ww)
            f.append(a0 - a1)
        f = np.array(f)
        ort, sh, td, kaz = _t_tablo(f)
        print(f"    {K:>4}{ort:+11.5f}{sh:9.5f}{td:+7.2f}{kaz:5d}/9{-ort * SICAK_KATSAYI:+10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
