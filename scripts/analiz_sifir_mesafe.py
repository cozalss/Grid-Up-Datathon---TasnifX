"""Sifir kirpmanin ve seviye farklarinin ayristirilmasi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
GONDERIM = KOK / "submissions"

v83 = pd.read_csv(GONDERIM / "tuketim_v83_sicak_optimum.csv")
sota1 = pd.read_csv(GONDERIM / "tuketim_sota_v1.csv")

lv83 = np.log1p(v83["tuketim"].to_numpy())
lsota = np.log1p(sota1["tuketim"].to_numpy())

# Sifir olmayan satirlardaki mesafe
sifir_mask = sota1["tuketim"] == 0
print(f"Toplam Satir: {len(v83):,}")
print(f"Sifira Cekilen Satir Sayisi: {sifir_mask.sum():,}")
print(
    f"Sifir Haric Satirlardaki RMSLE Mesafesi: {np.sqrt(np.mean((lsota[~sifir_mask] - lv83[~sifir_mask]) ** 2)):.5f}"  # noqa: E501
)
print(
    f"Yalniz Sifir Satirlardaki RMSLE Mesafesi: {np.sqrt(np.mean((lsota[sifir_mask] - lv83[sifir_mask]) ** 2)):.5f}"  # noqa: E501
)
