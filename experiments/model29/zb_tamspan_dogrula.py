"""IDDIA: olculmus span'in TAM optimumu ||r_hat||^2 = 0.003872 aciklar,
g7 ise yalnizca 0.003036. Yani g7 buzulmus ve masada 0.000247 duruyor.

BU IDDIAYI KIR. Uc test:
  A) rcond kararliligi -- pinv esigi degisince ||r_hat||^2 ne oluyor?
  B) LEAVE-ONE-OUT -- bir yonu disarida birak, kalanlardan r_hat kur,
     disarida birakilanin L'sini TAHMIN et, olculenle karsilastir.
     Bu, span icinde kalibrasyonun gercek testidir.
  C) YUVARLAMA -- her L'ye +-5e-6/2 gurultu ekleyip 200 kez coz, ||r_hat||^2 sacilimi.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
SK = json.load(open(os.path.join(KOK, "experiments/model29/olculmus_skorlar.json")))
TABAN = "tuketim_m6_ikiyon.csv"
M0 = 1.005846366
L_Y40 = -0.002229  # bugun olculdu


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    return np.log1p(d[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
V, L, ad = [], [], []
for f, P in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if len(v) != N:
        continue
    d = v - a0
    Q = float((d * d).mean())
    V.append(d)
    L.append((M0 + Q - P * P) / 2)
    ad.append(f)
# y40 dogrudan olculdu
V.append(oku("tuketim_y40_sota_temiz.csv") - a0)
L.append(L_Y40)
ad.append("tuketim_y40_sota_temiz.csv")
V = np.array(V).T
L = np.array(L)
K = V.shape[1]
G = (V.T @ V) / N
sv = np.linalg.svdvals(G)
print(
    f"olculmus yon {K}  cond(G)={sv[0] / sv[-1]:.2e}  tekil deger araligi "
    f"{sv[0]:.2e} .. {sv[-1]:.2e}"
)

# --- A) rcond kararliligi ---
print("\n[A] rcond kararliligi   (g7'nin acikladigi: 0.003036)")
print(f"{'rcond':>10s} {'rank':>5s} {'||r_hat||^2':>13s} {'tahmini LB':>12s} {'|c|_1':>10s}")
for rc in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]:
    Gi = np.linalg.pinv(G, rcond=rc)
    c = Gi @ L
    nrm = float(c @ G @ c)
    rank = int((sv > sv[0] * rc).sum())
    print(f"{rc:10.0e} {rank:5d} {nrm:13.6f} {np.sqrt(M0 - nrm):12.6f} {np.abs(c).sum():10.2f}")

# --- B) LEAVE-ONE-OUT ---
print(f"\n[B] LEAVE-ONE-OUT: {K - 1} yonden r_hat kur, disaridakinin L'sini tahmin et")
print(f"{'disarida':>34s} {'span-ici pay':>13s} {'olculen L':>11s} {'tahmin':>11s} {'hata':>11s}")
Gi_tam = np.linalg.pinv(G, rcond=1e-6)
sat = []
for i in range(K):
    idx = [j for j in range(K) if j != i]
    Gs = G[np.ix_(idx, idx)]
    c = np.linalg.pinv(Gs, rcond=1e-6) @ L[idx]
    # r_hat = V[:,idx] c ;  tahmin L_i = <r_hat, d_i>/N = (G[i,idx] @ c)
    tah = float(G[i, idx] @ c)
    # d_i'nin digerlerinin span'indaki payi
    b = G[idx, i]
    ci, *_ = np.linalg.lstsq(Gs, b, rcond=1e-6)
    ici = float(ci @ Gs @ ci) / G[i, i]
    sat.append((ad[i], ici, L[i], tah, tah - L[i]))
for a_, ici, l_, t_, h_ in sorted(sat, key=lambda x: -abs(x[4]))[:8]:
    print(
        f"{a_.replace('tuketim_', '').replace('.csv', ''):>34s} {ici:13.3f} "
        f"{l_:+11.6f} {t_:+11.6f} {h_:+11.6f}"
    )
ici_yuksek = [x for x in sat if x[1] > 0.95]
print(
    f"\n  span-ici payi >%95 olan {len(ici_yuksek)} yonde ortalama |hata| = "
    f"{np.mean([abs(x[4]) for x in ici_yuksek]):.2e}"
)
print("  (bu yonlerde tahmin ETMELI; hata buyukse cebir kalibre degil demektir)")

# --- C) yuvarlama gurultusu ---
rng = np.random.default_rng(5)
Gi = np.linalg.pinv(G, rcond=1e-6)
nrm0 = float((Gi @ L) @ G @ (Gi @ L))
oyn = []
for _ in range(300):
    Lg = L + rng.uniform(-5e-6, 5e-6, K)  # skor yuvarlamasi -> L'de ~P*dP/1
    c = Gi @ Lg
    oyn.append(float(c @ G @ c))
oyn = np.array(oyn)
print(
    f"\n[C] LB yuvarlamasi: ||r_hat||^2 = {nrm0:.6f}  sacilim sd {oyn.std():.2e}  "
    f"aralik [{oyn.min():.6f}, {oyn.max():.6f}]"
)
print(f"    skor cinsinden sd {oyn.std() / 2:.2e}")
print(f"\nSONUC: g7 0.003036 -> tam span {nrm0:.6f}   kazanc {nrm0 - 0.003036:.6f}")
print(f"       skor {np.sqrt(M0 - 0.003036):.5f} -> {np.sqrt(M0 - nrm0):.5f}")
