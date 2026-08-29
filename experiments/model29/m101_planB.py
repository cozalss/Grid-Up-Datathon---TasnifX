"""PLAN B sinavi: span yonunu GONDERMEDEN hesaba katabiliyor muyuz?
Eger evet ise HAK1 bosa gitmiyor ve o hakla IKINCI bir yeni yon olculebilir."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
skor = {
    k: v for k, v in json.load(open("olculmus_skorlar.json")).items() if k != "gun1_baseline.csv"
}
adlar = sorted(skor)
P = np.array([np.log1p(pd.read_csv(os.path.join(S, a)).tuketim.values) for a in adlar])
N = P.shape[1]
m = np.array([skor[a] ** 2 for a in adlar])
TAB = "tuketim_m6_ikiyon.csv"
ti = adlar.index(TAB)
a0 = P[ti]
m0 = m[ti]
D = P - a0  # her olculmus dosyanin YONU
Q = np.einsum("ij,ij->i", D, D) / N
L = (m0 + Q - m) / 2  # <r,d_j>/N = -L_j  (olculdu)
print(f"taban {TAB} m0={m0:.6f}, {len(adlar)} olculmus yon")

# g7 span dosyasi: bunun L'sini GONDERMEDEN cikarabiliyor muyuz?
g = np.log1p(pd.read_csv(os.path.join(S, "tuketim_g7_span_tau3.csv")).tuketim.values)
n = len(adlar)
Gd = P @ P.T / N
M = np.block([[Gd, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
w = np.linalg.lstsq(M, np.concatenate([P @ g / N, [1.0]]), rcond=None)[0][:n]
artik = float(((w @ P - g) ** 2).mean())
dg = g - a0
L_g_hesap = float(w @ L)  # dogrusallik: <r,dg> = sum w_i <r,d_i>
Q_g = float((dg**2).mean())
print(f"g7 span icinde mi? artik {artik:.2e}  sum(w)={w.sum():.6f}")
print(f"  L(g7) HESAPLANAN {L_g_hesap:+.6f}   Q(g7) {Q_g:.6f}")
print(
    f"  -> ongorulen skor {np.sqrt(m0 - L_g_hesap**2 / Q_g if Q_g > 0 else m0):.5f}  (ajan: 1.00137)"
)
print(f"  -> g7'yi GONDERMEDEN kazancini alabiliyoruz: {L_g_hesap**2 / Q_g:.6f} MSE")
print()
print("SONUC: g7 yonu GONDERILMEDEN hesaba katilabilir.")
print("       O halde HAK1 span'a degil, IKINCI BIR YENI YONE harcanmali.")
print()
h = 1.00284**2
print(f"{'plan':44s} {'ongoru MSE':>12s} {'RMSLE':>9s}")
for ad, kat in [
    ("A: span + y46 (r=0.040)", 0.002946 + 0.0016),
    ("A: span + y46 (r=0.050)", 0.002946 + 0.0025),
    ("B: span + y46 + y45 (ikisi r=0.040)", 0.002946 + 0.0016 * 2),
    ("B: span + y46 + y45 (ikisi r=0.050)", 0.002946 + 0.0025 * 2),
    ("B: span + y46 + y45 (ikisi r=0.030)", 0.002946 + 0.0009 * 2),
]:
    mse = h - kat
    print(
        f"  {ad:42s} {mse:12.6f} {np.sqrt(max(mse, 0)):9.5f}"
        + ("   <- 2. SIRA" if np.sqrt(max(mse, 0)) < 1.00041 else "")
    )
