"""KALIBRE RISK: 2. sirayi gecme olasiligi.

Onceki belgelerdeki "olculmus kaliteler 0.049-0.124" ESKI TABANA gore.
m6 tabanina gore olculmus GERCEK rho dagilimini onsel olarak kullan.
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
HEDEF = 0.99940
UCUNCU = 1.00267
LG = 0.002728


def oku(f):
    df = pd.read_csv(os.path.join(S, f))
    kol = "tuketim" if "tuketim" in df.columns else df.columns[-1]
    return np.log1p(df[kol].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)

# --- 1. m6 tabanina gore olculmus rho dagilimi ---
ABSORBE = {  # m6'nin optimize edildigi yonler -- rho ~ 0 olmasi ZORUNLU
    "tuketim_p51_sicak05.csv",
    "tuketim_m4_hava_capali.csv",
    "tuketim_v102_kappa_optimum.csv",
    "tuketim_v109_birlesik.csv",
}
rho_gecmis = []
for f, P in SK.items():
    if f == TABAN or f in ABSORBE:
        continue
    yol = os.path.join(S, f)
    if not os.path.exists(yol):
        continue
    v = oku(f)
    if len(v) != N:
        continue
    d = v - a0
    Q = float((d * d).mean())
    Lj = (m0 + Q - P * P) / 2.0
    rho_gecmis.append(abs(Lj / np.sqrt(Q)))
rho_gecmis = np.array(sorted(rho_gecmis))
print("GECMIS |rho| (m6 tabanina gore, absorbe edilmemis 20 yon)")
print("  " + "  ".join(f"{x:.4f}" for x in rho_gecmis))
print(
    f"  ortanca {np.median(rho_gecmis):.4f}  ort {rho_gecmis.mean():.4f}"
    f"  %10 {np.percentile(rho_gecmis, 10):.4f}  %90 {np.percentile(rho_gecmis, 90):.4f}"
)
print(f"  esigin (0.01373) UZERINDE olan: {(rho_gecmis >= 0.01373).sum()}/{len(rho_gecmis)}")

# --- 2. dort yonun G matrisi (kendi hesabim) ---
YON = [
    "tuketim_g7_span_tau3.csv",
    "tuketim_y40_sota_temiz.csv",
    "tuketim_z2_analog.csv",
    "tuketim_t1_sulama.csv",
]
D = np.array([oku(f) - a0 for f in YON]).T
G = (D.T @ D) / N
sq = np.sqrt(np.diag(G))
print(f"\nQ = {np.diag(G)}")
print(f"cond(G) = {np.linalg.cond(G):.2f}")
Gi = np.linalg.inv(G)


def skor(rho3, olcum_gurultusu=0.0, rng=None):
    L = np.concatenate([[LG], rho3 * sq[1:]])
    if olcum_gurultusu:
        Lh = L + np.concatenate([[0.0], rng.normal(0, olcum_gurultusu, 3)])
        k = Gi @ Lh
        mse = m0 - 2 * k @ L + k @ G @ k  # GERCEK L ile degerlendir
    else:
        mse = m0 - L @ Gi @ L
    return np.sqrt(max(mse, 1e-12))


# --- 3. Monte Carlo ---
rng = np.random.default_rng(7)
NS = 200_000
GUR = 1.6e-4  # sonda olcum hatasi -> L hatasi

for ad, olcek in [
    ("gecmis dagilimi", 1.0),
    ("KOTUMSER (yarisi)", 0.5),
    ("COK KOTUMSER (dortte biri)", 0.25),
]:
    # gecmis |rho|'lardan orneklem + rastgele isaret (isaret onemsiz, L^2/Q)
    r = rng.choice(rho_gecmis, size=(NS, 3)) * olcek
    r = r * rng.choice([-1.0, 1.0], size=(NS, 3))
    L = np.concatenate([np.full((NS, 1), LG), r * sq[1:]], axis=1)
    Lh = L + np.concatenate([np.zeros((NS, 1)), rng.normal(0, GUR, (NS, 3))], axis=1)
    K = Lh @ Gi.T
    mse = m0 - 2 * (K * L).sum(1) + ((K @ G) * K).sum(1)
    s = np.sqrt(np.maximum(mse, 1e-12))
    print(
        f"\n[{ad}]  medyan {np.median(s):.5f}  %5 {np.percentile(s, 5):.5f}"
        f"  %95 {np.percentile(s, 95):.5f}"
    )
    print(
        f"   P(2. sira, <{HEDEF}) = {(s < HEDEF).mean() * 100:5.1f}%"
        f"   P(3. sira, <{UCUNCU}) = {(s < UCUNCU).mean() * 100:5.1f}%"
        f"   P(taban 1.00284'ten kotu) = {(s > 1.00284).mean() * 100:4.1f}%"
    )

print(f"\nHIC BILGI YOKSA (r=0): {skor(np.zeros(3)):.5f}")
