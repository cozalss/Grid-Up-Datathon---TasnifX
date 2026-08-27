"""v87 Tabanli: 14.484 olu trafo satirina sert 0 yerine 1.0 kWh taban atar.

Log1p uzayinda asimetrik kayip fonksiyonu korumasi:
- Gercek 0 ise: (log(2) - 0)^2 = 0.48 MSE (onemsiz mikro kayip)
- Gercek 4000 ise: (log(4001) - log(2))^2 = 57.77 MSE (68.79 yerine -> 11.02 MSE kurtarma!)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[1]
GONDERIM = KOK / "submissions"


def main():
    taban = pd.read_csv(GONDERIM / "tuketim_v83_sicak_optimum.csv")
    sota = pd.read_csv(GONDERIM / "tuketim_sota_v1.csv")

    maske = sota["tuketim"].to_numpy() == 0.0
    tahmin = taban["tuketim"].to_numpy().astype("float64").copy()

    # 1.0 kWh guvenlik tabani
    tahmin[maske] = 1.0

    yol = GONDERIM / "tuketim_v87_tabanli.csv"
    pd.DataFrame({"id": taban["id"], "tuketim": tahmin}).to_csv(yol, index=False)
    print(f"YAZILDI: {yol.name}")
    print(f"  1.0 kWh taban atanan satir: {int(maske.sum()):,}")
    return 0


if __name__ == "__main__":
    main()
