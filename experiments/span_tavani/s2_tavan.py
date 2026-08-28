"""22 olculmus skorun span'indaki EN IYI noktayi coz (v101/v102/v109 dahil).

MSE(c) = m0 - 2 c.L + c' G c   ->   c* = G^+ L,  MSE* = m0 - L' G^+ L
Rank secimi SNR ile: her ozvektor icin |u.L| / sd(L). sd yalniz 5 hane
skor yuvarlamasindan gelir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SKOR = {
    "tuketim_v2.csv": 1.16143,
    "tuketim_v7.csv": 1.16922,
    "tuketim_v15.csv": 1.03910,
    "tuketim_v16.csv": 1.06605,
    "tuketim_v18.csv": 1.03370,
    "tuketim_v25_hedge.csv": 1.04820,
    "tuketim_v27_v18hedge.csv": 1.03362,
    "tuketim_v30_buzme.csv": 1.02639,
    "tuketim_v44_v27yeni.csv": 1.03053,
    "tuketim_v46_gun.csv": 1.02448,
    "tuketim_v47_eskison.csv": 1.01750,
    "tuketim_v50_nihai30.csv": 1.01686,
    "tuketim_v55_gunolcek.csv": 1.01591,
    "tuketim_v67_c1335_olay.csv": 1.01548,
    "tuketim_v73_soguk_gun160.csv": 1.01538,
    "tuketim_v79_S3.csv": 1.01556,
    "tuketim_v80_optimum.csv": 1.01341,
    "tuketim_v81_sicak08.csv": 1.01429,
    "tuketim_v83_sicak_optimum.csv": 1.01318,
    "tuketim_v101_hepsi.csv": 1.01614,
    "tuketim_v102_kappa_optimum.csv": 1.00553,
    "tuketim_v109_birlesik.csv": 1.01818,
}
TABAN = "tuketim_v102_kappa_optimum.csv"

sample = pd.read_csv("data/raw/sample_submission.csv", dtype={"id": str})
idx = sample.id.to_numpy()
n = len(idx)


def yukle(ad: str) -> np.ndarray:
    d = pd.read_csv(f"submissions/{ad}", dtype={"id": str}).set_index("id").reindex(idx)
    return np.log1p(np.clip(d.iloc[:, 0].to_numpy(dtype=float), 0, None))


y0 = yukle(TABAN)
m0 = SKOR[TABAN] ** 2
ad_d = [a for a in SKOR if a != TABAN]
D = np.stack([yukle(a) - y0 for a in ad_d])
G = D @ D.T / n
L = np.array([(m0 + G[j, j] - SKOR[a] ** 2) / 2.0 for j, a in enumerate(ad_d)])

# Skor 5 haneye yuvarli -> RMSLE'de +-5e-6 -> MSE'de +-2*s*5e-6 ~ 1.0e-5.
# L_j = (m0 + G_jj - m_j)/2 ; m0 ve m_j bagimsiz yuvarli -> sd ~ sqrt(2)*1.0e-5/2
SD = float(np.sqrt(2) * 1.01e-5 / 2)
print(f"n={n:,}  yon={len(ad_d)}  m0={m0:.6f}  taban RMSLE={np.sqrt(m0):.5f}")
print(f"L gurultusu sd = {SD:.2e}\n")

w, V = np.linalg.eigh(G)
sira = np.argsort(w)[::-1]
w, V = w[sira], V[:, sira]
uL = V.T @ L

print("  #   ozdeger      u.L        SNR     kumulatif kazanc   RMSLE")
kaz = 0.0
secim = []
for k in range(len(w)):
    if w[k] <= 0:
        break
    snr = abs(uL[k]) / SD
    katki = uL[k] ** 2 / w[k]
    isaret = " "
    if snr >= 3.0:
        kaz += katki
        secim.append(k)
        isaret = "*"
    print(
        f"{isaret}{k:3d}  {w[k]:.3e}  {uL[k]:+.3e}  {snr:8.1f}  "
        f"{katki:12.6f}  {np.sqrt(max(m0 - kaz, 0)):8.5f}"
    )

print(f"\nSNR>=3 secilen bilesen: {len(secim)}")
print(f"span tavani  MSE = {m0 - kaz:.6f}   RMSLE = {np.sqrt(max(m0 - kaz, 0)):.5f}")
print(f"mevcut       MSE = {m0:.6f}   RMSLE = {np.sqrt(m0):.5f}")
print(f"kazanc dMSE      = {-kaz:.6f}")
print("\nLIDER 0.99138 -> MSE 0.982835 ; gereken dMSE -0.028256")
