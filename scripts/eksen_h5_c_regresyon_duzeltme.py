# ruff: noqa
"""H5 -- adim 3 regresyonunun SIZINTISIZ yeniden okunmasi.

eksen_h5_a'daki 'kapasite payi x lok toplam' referansi RHS'de ``yeni_yuk``
tasiyordu (sizinti). Burada referans YALNIZCA kesme oncesi (as-of) buyuklukten
kurulur ve ``devir``in EK aciklayiciligi olculur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ARA = KOK / "data" / "interim" / "eksen_h5"
CIK = KOK / "reports" / "eksen_h5"


def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype="float64") for c in X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - float((e**2).sum() / ss), beta


def main() -> int:
    satir = []

    def yaz(s=""):
        print(s)
        satir.append(s)

    for seviye in ("ilce", "bolge"):
        E = pd.read_parquet(ARA / f"olaylar_{seviye}.parquet")
        E = E[np.isfinite(E["devir"]) & np.isfinite(E["yeni_yuk"]) & (E["n_yeni_gecerli"] > 0)]
        yaz(f"\n=== {seviye} ===  olay {len(E):,}")
        for ad, a in (
            ("TUMU", E),
            ("TEKIL", E[~E["toplu"]]),
            ("yaz25", E[E["mevsim"] == "yaz25"]),
            ("yaz25 TEKIL", E[(E["mevsim"] == "yaz25") & ~E["toplu"]]),
            ("guz25", E[E["mevsim"] == "guz25"]),
        ):
            if len(a) < 20:
                yaz(f"  {ad:<14} n={len(a)} yetersiz")
                continue
            y = a["yeni_yuk"].to_numpy("float64")
            pay = (a["guc_yeni"] / (a["guc_yeni"] + a["guc_yerlesik"])).to_numpy("float64")
            asof = pay * a["on_top"].to_numpy("float64")  # SIZINTISIZ referans
            dev = a["devir"].to_numpy("float64")
            r_dev, _ = r2(y, [dev])
            r_asof, _ = r2(y, [asof])
            r_ikisi, b = r2(y, [asof, dev])
            yaz(
                f"  {ad:<14} n={len(a):>5}  R2(devir)={r_dev:.4f}  "
                f"R2(as-of pay*on_top)={r_asof:.4f}  R2(ikisi)={r_ikisi:.4f}  "
                f"EK={r_ikisi - r_asof:+.4f}  b_devir={b[2]:+.4f}"
            )
    (CIK / "adim3_duzeltme.txt").write_text("\n".join(satir), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
