"""SICAK seviye desili -- desenin bloklar arasi ISARETI gercekten ne?

desen.py kappa'yi NEGATIF buldu (-1.06 / -0.80 / -2.56), tavan.py ise
korelasyonu POZITIF (+0.57). Ikisi ayni sey olmali. Hangisi dogru?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))

from tavan import BLOKLAR, SIC, desil  # noqa: E402


def main() -> None:
    bl = SIC.bloklari_kur()
    tab = {k: SIC.taban_r(bl[k]) for k in BLOKLAR}
    desen, boyut = {}, {}
    for k in BLOKLAR:
        b = bl[k]
        r0 = tab[k]
        lgp = np.maximum(r0 + b.lgc, 0.0)
        g = desil(lgp, 10)
        e = b.lgy - lgp
        e = e - e.mean()
        s = pd.Series(e).groupby(pd.Series(g))
        desen[k] = s.mean()
        boyut[k] = s.size()

    df = pd.DataFrame(desen)
    print("SICAK seviye desili -- blok basina ortalama artik (merkezlenmis)\n")
    print(df.round(4).to_string())
    print("\nblok basina desil buyuklukleri")
    print(pd.DataFrame(boyut).to_string())

    print("\nham (agirliksiz) Pearson:")
    print(df.corr().round(3).to_string())

    print("\nsatir-agirlikli kovaryans / kappa (hedef blok agirliklariyla):")
    for hedef in BLOKLAR:
        digerleri = [k for k in BLOKLAR if k != hedef]
        d = df[digerleri].mean(axis=1)
        w = boyut[hedef] / boyut[hedef].sum()
        dc = d - float((w * d).sum())
        ec = df[hedef] - float((w * df[hedef]).sum())
        L = float((w * dc * ec).sum())
        Q = float((w * dc * dc).sum())
        print(f"  hedef {hedef}: L={L:+.6f}  Q={Q:.6f}  kappa={L / Q:+.3f}")

    print("\nDESIL KENARLARI blok basina (log1p tahmin):")
    for k in BLOKLAR:
        lgp = np.maximum(tab[k] + bl[k].lgc, 0.0)
        print(f"  {k}: {np.round(np.quantile(lgp, np.linspace(0, 1, 11)), 3)}")


if __name__ == "__main__":
    main()
