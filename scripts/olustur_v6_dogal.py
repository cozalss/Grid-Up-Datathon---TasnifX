"""Sifir kirpmasiz, dogal ve tam guvenli harman: v6_dogal_garanti."""

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

    # v83 ve v85 gibi kanitlanmis tabanlari esas alarak, yeni SOTA modelin
    # ogrendigi takvim ve kohort bilgilerini guvenli oranda aktar:
    l_sota = np.log1p(sota_df["tuketim"].to_numpy())
    l_v83 = np.log1p(v83_df["tuketim"].to_numpy())
    l_v85 = np.log1p(v85_df["tuketim"].to_numpy())

    # Eger sota_df sifira cekilmisse, v83'un dogal degerlerini koru (asimetrik risk yok)
    l_sota_dogal = l_sota.copy()
    sifir_idx = sota_df["tuketim"] == 0
    l_sota_dogal[sifir_idx] = l_v83[sifir_idx]

    l_final = np.zeros_like(l_sota)
    # Sicak trafolarda %70 SOTA (dogal) + %30 v83 (kanitlanmis taban)
    l_final[sicak_mask] = 0.70 * l_sota_dogal[sicak_mask] + 0.30 * l_v83[sicak_mask]
    # Soguk trafolarda %65 SOTA + %35 v85 (Gram)
    l_final[~sicak_mask] = 0.65 * l_sota_dogal[~sicak_mask] + 0.35 * l_v85[~sicak_mask]

    tuketim_final = np.clip(np.expm1(l_final), 0, None)

    yol = GONDERIM / "tuketim_sota_v6_dogal_garanti.csv"
    pd.DataFrame({"id": sota_df["id"], "tuketim": tuketim_final}).to_csv(yol, index=False)
    print(f"YAZILDI: {yol.name}")
    return 0


if __name__ == "__main__":
    main()
