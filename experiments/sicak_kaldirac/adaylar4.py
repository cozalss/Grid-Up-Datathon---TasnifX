"""SICAK ADAYLAR -- 4. tur: trafo-ICI buzme ve sifir kutlesi hedge'i.

``adaylar.py`` A9 trafo-ARASI bileseni buzuyordu. Bu tur trafo-ICI
bileseni (satirin kendi trafo ortalamasindan sapmasi) buzer, ve ayrica
"olu trafo" kuralinin YUMUSAK surumlerini olcer -- sert sifirlamanin
sicak tarafta karsiligi var mi?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, bloklari_kur, mse, rapor, taban_r, tablo_yaz  # noqa: E402


def ici_buzme(beta: float):
    def aday(b, r0):
        t = pd.Series(b.cerceve["tanim"].to_numpy())
        s = pd.Series(r0)
        tort = s.groupby(t).transform("mean").to_numpy()
        return tort + beta * (r0 - tort)

    return aday


def sifir_hedge(kappa: float, sifir_esik: float, kuyruk_esik: int):
    def aday(b, r0):
        c = b.cerceve
        m = (c["t_sifir_orani"].to_numpy() >= sifir_esik) & (
            c["t_kuyruk_sifir"].to_numpy() >= kuyruk_esik
        )
        r = r0.copy()
        r[m] = r0[m] - kappa
        return r

    return aday


def dusuk_hedge(kappa: float, esik_kwh: float):
    """Tahmini dusuk olan satirlarda asagi kaydirma (sifir kutlesi hedge'i)."""

    def aday(b, r0):
        tah = np.expm1(np.maximum(r0 + b.lgc, 0.0))
        m = tah <= esik_kwh
        r = r0.copy()
        r[m] = r0[m] - kappa
        return r

    return aday


def main() -> int:
    bl = bloklari_kur()
    taban = {k: taban_r(b) for k, b in bl.items()}
    print("TABAN sicak MSE:", {k: round(mse(bl[k], taban[k]), 5) for k in BLOKLAR})

    print("\nSIFIR HEDGE MASKESININ BUYUKLUGU ve GERCEK SIFIR ORANI")
    for se, ke in ((0.5, 7), (0.7, 7), (0.9, 14)):
        for k in BLOKLAR:
            b = bl[k]
            c = b.cerceve
            m = (c["t_sifir_orani"].to_numpy() >= se) & (c["t_kuyruk_sifir"].to_numpy() >= ke)
            if m.sum():
                print(
                    f"  sifir>={se} kuyruk>={ke}  {k:6} n={int(m.sum()):>7,} "
                    f"gercek sifir orani %{100 * (b.y[m] == 0).mean():5.1f}  "
                    f"medyan y {np.median(b.y[m]):8.1f}"
                )
            else:
                print(f"  sifir>={se} kuyruk>={ke}  {k:6} n=0")

    satirlar = []
    for beta in (0.90, 0.95, 1.05, 1.10):
        satirlar.append(rapor(bl, ici_buzme(beta), f"D1 trafo-ICI buzme b={beta}", taban))
    for kappa in (0.25, 0.5, 1.0):
        for se, ke in ((0.5, 7), (0.9, 14)):
            satirlar.append(
                rapor(
                    bl,
                    sifir_hedge(kappa, se, ke),
                    f"D2 sifir hedge k={kappa} s>={se} q>={ke}",
                    taban,
                )
            )
    for kappa in (0.15, 0.3):
        for esik in (5.0, 20.0):
            satirlar.append(
                rapor(
                    bl, dusuk_hedge(kappa, esik), f"D3 dusuk hedge k={kappa} <={esik:.0f}kWh", taban
                )
            )

    tablo_yaz(satirlar)
    yol = Path(__file__).resolve().parent / "adaylar4.jsonl"
    with yol.open("w", encoding="utf-8") as f:
        for s in satirlar:
            f.write(pd.Series(s).to_json() + "\n")
    print(f"\nyazildi: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
