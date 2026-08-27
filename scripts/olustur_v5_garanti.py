"""Zirve garanti modelini olustur: %85 SOTA + %15 v83 (Sicak), %80 SOTA + %20 v85 (Soguk)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
GONDERIM = KOK / "submissions"
DATA = KOK / "data" / "raw"


def main():
    sota_df = pd.read_csv(GONDERIM / "tuketim_sota_v1.csv")
    v83_df = pd.read_csv(GONDERIM / "tuketim_v83_sicak_optimum.csv")
    v85_df = pd.read_csv(GONDERIM / "tuketim_v85_gram_rank2.csv")
    tr = pd.read_csv(DATA / "train.csv", usecols=["tanim"], dtype={"tanim": str})

    sicak_set = set(tr["tanim"].unique())
    tanimlar = sota_df["id"].str.rsplit("_", n=1).str[0]
    sicak_mask = tanimlar.isin(sicak_set).to_numpy()

    l_sota = np.log1p(sota_df["tuketim"].to_numpy())
    l_v83 = np.log1p(v83_df["tuketim"].to_numpy())
    l_v85 = np.log1p(v85_df["tuketim"].to_numpy())

    l_final = np.zeros_like(l_sota)
    # Sicak trafolarda %85 SOTA + %15 v83 (Maksimum SOTA gucu + kucuk varyans korumasi)
    l_final[sicak_mask] = 0.85 * l_sota[sicak_mask] + 0.15 * l_v83[sicak_mask]
    # Soguk trafolarda %80 SOTA + %20 v85
    l_final[~sicak_mask] = 0.80 * l_sota[~sicak_mask] + 0.20 * l_v85[~sicak_mask]

    tuketim_final = np.clip(np.expm1(l_final), 0, None)
    # Olu trafo sifirlamasini koru
    tuketim_final[sota_df["tuketim"] == 0] = 0.0

    yol = GONDERIM / "tuketim_sota_v5_zirve_garanti.csv"
    pd.DataFrame({"id": sota_df["id"], "tuketim": tuketim_final}).to_csv(yol, index=False)
    print(f"YAZILDI: {yol.name} (Sicak: %85 SOTA + %15 v83, Soguk: %80 SOTA + %20 v85)")
    return 0


if __name__ == "__main__":
    main()
