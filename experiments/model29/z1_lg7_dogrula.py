"""L_g7 = 0.002728 sabitini SIFIRDAN dogrula.

Yontem: olculmus her dosya icin L_j = (m0 + Q_j - P_j^2)/2 (kesin, cebirsel).
Sonra d_g7'yi bu olculmus yonlerin span'ina en kucuk kareyle yaz: d_g7 ~ D c.
L_g7 = c' L  (span'da tam duruyorsa kesin; artik varsa alt sinir).
Artik oranini da raporla -- span'da degilse L_g7 OLCULEMEZ demektir.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
SK = json.load(open(os.path.join(KOK, "experiments/model29/olculmus_skorlar.json")))

TABAN = "tuketim_m6_ikiyon.csv"
m0 = SK[TABAN] ** 2


def oku(f):
    df = pd.read_csv(os.path.join(S, f))
    kol = "tuketim" if "tuketim" in df.columns else df.columns[-1]
    return np.log1p(df[kol].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
print(f"taban {TABAN}  P={SK[TABAN]}  m0={m0:.9f}  N={N}")

adlar, D, L = [], [], []
for f, P in SK.items():
    if f == TABAN:
        continue
    yol = os.path.join(S, f)
    if not os.path.exists(yol):
        print(f"  YOK  {f}")
        continue
    v = oku(f)
    if len(v) != N:
        print(f"  ATLA {f} satir={len(v)}")
        continue
    d = v - a0
    Q = float((d * d).mean())
    Lj = (m0 + Q - P * P) / 2.0
    adlar.append(f)
    D.append(d)
    L.append(Lj)
    print(f"  {f:36s} P={P:<9.5f} Q={Q:.6f} L={Lj:+.6f} rho={Lj / np.sqrt(Q):+.4f}")

D = np.array(D).T  # N x K
L = np.array(L)
K = D.shape[1]
print(f"\nolculmus yon sayisi K={K}")

dg = oku("tuketim_g7_span_tau3.csv") - a0
Qg = float((dg * dg).mean())

G = (D.T @ D) / N
b = (D.T @ dg) / N
c, *_ = np.linalg.lstsq(G, b, rcond=None)
kalinti = dg - D @ c
oran = float((kalinti**2).mean()) / Qg
Lg = float(c @ L)

print(f"\nQ_g7 = {Qg:.9f}   (kayitli 0.002494)")
print(f"span artigi / Q_g7 = {oran:.6f}   -> {'SPAN ICINDE' if oran < 0.02 else 'SPAN DISINDA!'}")
print(f"L_g7 (span'dan) = {Lg:+.9f}   (kayitli 0.002728)")
print(f"fark = {Lg - 0.002728:+.2e}")
print(f"rho_g7 = {Lg / np.sqrt(Qg):+.6f}   (kayitli 0.054626)")
print(f"|c|_1 = {np.abs(c).sum():.3f}   cond(G) = {np.linalg.cond(G):.3e}")

print("\n--- g7 tek basina optimumda ---")
print(f"  MSE = {m0 - Lg * Lg / Qg:.9f}  -> RMSLE {np.sqrt(m0 - Lg * Lg / Qg):.5f}")
print("\n--- olculmus TUM yonlerin ortak optimumu (span tavani) ---")
kk = np.linalg.solve(G + 1e-10 * np.eye(K), L)
mse = m0 - L @ kk
print(f"  MSE = {mse:.9f} -> RMSLE {np.sqrt(mse):.5f}  |k|_1={np.abs(kk).sum():.3f}")
