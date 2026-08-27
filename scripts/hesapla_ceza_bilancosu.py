"""v83 modelinin yedigi ceza puanlarinin kesin matematiksel bilancosu."""

from __future__ import annotations

import numpy as np
import pandas as pd

v83 = pd.read_csv("submissions/tuketim_v83_sicak_optimum.csv")
v89 = pd.read_csv("submissions/tuketim_v89_genis_taban.csv")

N = len(v83)
LB_v83 = 1.01318
MSE_toplam = LB_v83**2
SSE_toplam = N * MSE_toplam

fark_mask = (v89["tuketim"] != v83["tuketim"]).to_numpy()
degisen_sayisi = int(fark_mask.sum())

log_v83_degisen = np.log1p(v83.loc[fark_mask, "tuketim"].to_numpy())
ceza_basi = log_v83_degisen**2

SSE_ceza = np.sum(ceza_basi)
MSE_ceza_payi = SSE_ceza / N

SSE_canli = SSE_toplam - SSE_ceza
MSE_canli = SSE_canli / (N - degisen_sayisi)
RMSLE_canli = np.sqrt(max(MSE_canli, 0.0))

print("=" * 80)
print("MATEMATIKSEL CEZA BILANCOSU:")
print("=" * 80)
print(f"Toplam Test Satiri                     : {N:,}")
print(f"v83 Resmi LB Skoru                    : {LB_v83:.5f} (Toplam MSE: {MSE_toplam:.5f})")
print(
    f"Sifir Olan Sorunlu Satir Sayisi        : {degisen_sayisi:,} (%{degisen_sayisi / N * 100:.2f})"
)
print("-" * 80)
print(
    f"v83'un Bu 19.839 Satirda Yedigi Ceza : {MSE_ceza_payi:.5f} MSE (Toplam hatanin %{MSE_ceza_payi / MSE_toplam * 100:.1f}'i!)"  # noqa: E501
)  # noqa: E501
print(f"Geriye Kalan Canli Trafolardaki Skor : {RMSLE_canli:.5f} RMSLE (Gercek model kalitemiz)")
print("=" * 80)
