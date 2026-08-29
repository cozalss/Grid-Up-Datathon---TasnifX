"""DURUST OLASILIK.

Onceki iki modelim de eksikti:
  z3: rho'yu yonun TAMAMINA cekti -> span ici parcanin L'si de rastgele saniliyor (fazla iyimser)
  z5: span ici parcayi tamamen yok saydi (fazla kotumser)
DOGRUSU: her adayin span ICI parcasinin L'si BILINIYOR (olculmus L'lerden cikar),
yalnizca span DISI parcasi bilinmiyor. Onseli sadece o parcaya uygula.
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
HEDEF, UCUNCU = 0.99940, 1.00267


def oku(f):
    df = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in df.columns else df.columns[-1]
    return np.log1p(df[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
V, L = [], []
for f, P in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if len(v) != N:
        continue
    d = v - a0
    V.append(d)
    L.append((M0 + Q if (Q := float((d * d).mean())) else 0) - P * P)
L = np.array(L) / 2
V = np.array(V).T
Gm = (V.T @ V) / N
# olculmus span icinde bir x yonunun L'si: x = V c  ->  L_x = c'L
Gi = np.linalg.pinv(Gm, rcond=1e-10)


def L_span(x):
    """x'in span ICI bileseninin bilinen L'si."""
    return float(Gi @ ((V.T @ x) / N) @ L)


ADAY = {
    "y40": "tuketim_y40_sota_temiz.csv",
    "z2": "tuketim_z2_analog.csv",
    "sul": "tuketim_t1_sulama.csv",
    "y46": "tuketim_y46_amnezik_kirpik.csv",
    "y45": "tuketim_y45_mevsimsel_kirpik.csv",
    "q1c": "tuketim_q1c_kapasite_siki.csv",
    "t3": "tuketim_t3_turizm.csv",
    "p42": "tuketim_p42_seviye_egrilik.csv",
    "h1": "tuketim_h1_isil.csv",
}
dg = oku("tuketim_g7_span_tau3.csv") - a0
U, sv, _ = np.linalg.svd(V, full_matrices=False)
BS = np.ascontiguousarray(U[:, : int((sv > sv[0] * 1e-7).sum())])
del U

isim = ["g7", *ADAY]
D = [dg] + [oku(f) - a0 for f in ADAY.values()]
Lbil = np.array([L_span(x) for x in D])  # span ici BILINEN kisim
Qdik = np.array([float(((x - BS @ (BS.T @ x)) ** 2).mean()) for x in D])
D = np.array(D).T
G = (D.T @ D) / N
print(f"m0={M0:.9f}  taban {np.sqrt(M0):.5f}")
print(f"{'yon':>5s} {'Q':>9s} {'L_span(bilinen)':>16s} {'Q_dik':>9s} {'dik oran':>9s}")
for i, ad in enumerate(isim):
    print(f"{ad:>5s} {G[i, i]:9.5f} {Lbil[i]:+16.6f} {Qdik[i]:9.5f} {Qdik[i] / G[i, i]:9.3f}")

RHO_DURUST = np.array([0.0023, 0.0072, 0.0114, 0.0119, 0.0146, 0.0160, 0.0161, 0.0214, 0.0253])
RHO_HAM_ORT = 0.0272
rng = np.random.default_rng(17)
NS = 200_000
GUR = 2.0e-4


def mc(k_yon, rho_havuz, olcek=1.0, etiket=""):
    idx = list(range(k_yon + 1))  # 0 = g7
    Gs = G[np.ix_(idx, idx)]
    Gis = np.linalg.inv(Gs)
    Lb = Lbil[idx]
    qd = Qdik[idx]
    K = len(idx)
    r = rng.choice(rho_havuz, size=(NS, K)) * olcek * rng.choice([-1.0, 1.0], size=(NS, K))
    r[:, 0] = 0.0  # g7 span ICINDE, dik parcasi yok
    Ltam = Lb + r * np.sqrt(qd)
    Lh = Ltam + rng.normal(0, GUR, (NS, K))
    Lh[:, 0] = Ltam[:, 0]
    kk = Lh @ Gis.T
    mse = M0 - 2 * (kk * Ltam).sum(1) + ((kk @ Gs) * kk).sum(1)
    s = np.sqrt(np.maximum(mse, 1e-12))
    taban = np.sqrt(max(M0 - Lb @ Gis @ Lb, 1e-12))
    print(
        f"{etiket:34s} P(2.)={(s < HEDEF).mean() * 100:5.1f}%  medyan {np.median(s):.5f}"
        f"  %5 {np.percentile(s, 5):.5f}  %95 {np.percentile(s, 95):.5f}"
        f"  dik-bilgi-yok {taban:.5f}  P(kotu) {(s > np.sqrt(M0)).mean() * 100:.1f}%"
    )
    return s


print(f"\nDURUST ONSEL (artimli rho, ortanca {np.median(RHO_DURUST):.4f}) -- ANA TAHMIN")
for k in [0, 3, 5, 6, 7, 8]:
    mc(k, RHO_DURUST, 1.0, f"  {k} yeni eksen")
print(f"\nIYIMSER ONSEL (ham rho, ortanca {RHO_HAM_ORT:.4f}) -- UST SINIR")
mc(8, np.array([RHO_HAM_ORT]), 1.0, "  8 eksen, ham rho")
print("\nKOTUMSER (durust onselin yarisi)")
mc(8, RHO_DURUST, 0.5, "  8 eksen")
