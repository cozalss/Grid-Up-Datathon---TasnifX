"""v88: v83 + olu trafo maskesine SERT SIFIR yerine kucuk taban.

Sert sifir bicak sirti bir bahistir: kural %92 dogruysa 0.909, %30 dogruysa
1.095. 1 kWh'lik taban iyimser ucta yalniz +0.002 kaybettirir ama felaket
senaryosunda 0.060 kazandirir -- ve merkez senaryoda sert sifirdan iyidir.

Olcum: maskelenen 193 trafonun iyi kapsamli her ayinda sifir orani %96-99;
surekli kapsamli alt kumede %79. Basa bas %57.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
G = KOK / "submissions"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taban", type=float, default=1.0, help="olu satirlara yazilacak kWh")
    ap.add_argument("--cikti", default="tuketim_v88_olu_taban.csv")
    args = ap.parse_args()

    taban = pd.read_csv(G / "tuketim_v83_sicak_optimum.csv")
    sota = pd.read_csv(G / "tuketim_sota_v1.csv")
    if not taban["id"].equals(sota["id"]):
        raise RuntimeError("id sirasi eslesmiyor")

    maske = sota["tuketim"].to_numpy() == 0.0
    tahmin = taban["tuketim"].to_numpy().astype("float64").copy()
    onceki = np.log1p(tahmin[maske])
    tahmin[maske] = args.taban

    yol = G / args.cikti
    pd.DataFrame({"id": taban["id"], "tuketim": tahmin}).to_csv(yol, index=False)
    print(f"YAZILDI {yol.name}")
    print(f"  maskelenen satir      : {int(maske.sum()):,}")
    print(f"  yazilan deger         : {args.taban} kWh (log1p {np.log1p(args.taban):.4f})")
    print(f"  v83 log1p ort/medyan  : {onceki.mean():.3f} / {np.median(onceki):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
