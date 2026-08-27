"""v87: v83 + YALNIZ olu trafo sifirlamasi. Tek degiskenli izole prob.

v83 (LB 1.01318) tabanina SOTA'nin isaretledigi 14.484 satiri sifirlar.
Baska hicbir sey degismez -- gelen skor farki TAM OLARAK sifirlamanin degeridir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
GONDERIM = KOK / "submissions"


def main() -> int:
    taban = pd.read_csv(GONDERIM / "tuketim_v83_sicak_optimum.csv")
    sota = pd.read_csv(GONDERIM / "tuketim_sota_v1.csv")
    if not taban["id"].equals(sota["id"]):
        raise RuntimeError("id sirasi eslesmiyor")

    maske = sota["tuketim"].to_numpy() == 0.0
    tahmin = taban["tuketim"].to_numpy().astype("float64").copy()
    onceki = np.log1p(tahmin[maske])
    tahmin[maske] = 0.0

    yol = GONDERIM / "tuketim_v87_olu_izole.csv"
    pd.DataFrame({"id": taban["id"], "tuketim": tahmin}).to_csv(yol, index=False)
    print(f"YAZILDI {yol.name}")
    print(f"  sifirlanan satir      : {int(maske.sum()):,}")
    print(f"  v83 log1p ort/medyan  : {onceki.mean():.3f} / {np.median(onceki):.3f}")
    print(f"  hepsi 0 ise MSE payi  : {float(np.sum(onceki**2) / len(tahmin)):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
