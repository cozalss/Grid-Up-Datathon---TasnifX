"""AYRISTIRMA KARAR BETIGI -- Gram, ayristirma kazanc araligi, prob sirasi, butce."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gram2"))
sys.stdout.reconfigure(encoding="utf-8")
from g01_havuz import HAVUZ, yukle  # noqa

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
CIK = Path(__file__).resolve().parent
adlar, X, skorlar, ids = yukle()
n = X.shape[1]
i83 = adlar.index("v83")
m0 = float(skorlar[i83] ** 2)
yon = [a for a in adlar if a != "v83"]
idx = [adlar.index(a) for a in yon]
D = X[idx] - X[i83]
Gm = (D @ D.T) / n
m = skorlar[idx] ** 2
b = (m0 + np.diag(Gm) - m) / 2.0


def span_L(u, D, Gm, b, tol=1e-10):
    """u'nun olculmus span'a izdusumunun L'si + dik artik Q'su."""
    c = (D @ u) / n
    lam, V = np.linalg.eigh(Gm)
    kes = lam > lam.max() * tol
    alpha = V[:, kes] @ ((V[:, kes].T @ c) / lam[kes])
    L_bilinen = float(alpha @ b)
    upar = alpha @ D
    Qperp = float((u - upar) @ (u - upar)) / n
    return L_bilinen, Qperp, float(u @ u) / n


# ---------- 0. Span tukenmisligi ----------
lam, V = np.linalg.eigh(Gm)
sira = np.argsort(lam)[::-1]
lam, V = lam[sira], V[:, sira]
beta = V.T @ b
kes = lam > lam.max() * 1e-10
span_kazanc = float((beta[kes] ** 2 / lam[kes]).sum())
print("=" * 86)
print("0. OLCULMUS SPAN'IN TUKENMISLIGI")
print("=" * 86)
print(f"m0(v83)            = {m0:.7f}   RMSLE {np.sqrt(m0):.5f}")
print(
    f"span toplam kazanc = {span_kazanc:.7f}  -> MSE {m0 - span_kazanc:.7f} RMSLE {np.sqrt(m0 - span_kazanc):.5f}"
)
print("v102 gerceklesen   = 0.0154434       -> MSE 1.0110910 RMSLE 1.00553")
print(
    f"SPAN'DA KALAN      = {span_kazanc - 0.0154434:+.7f}   (RMSLE kazanci {np.sqrt(m0 - span_kazanc) - 1.00553:+.6f})"
)


# ---------- 1. Bes bilesen ----------
def lg(a):
    d = pd.read_csv(GON / a)
    assert (d["id"].values == ids.values).all(), a
    return np.log1p(d["tuketim"].to_numpy("float64"))


FL = {
    k: lg(v)
    for k, v in {
        "v83": "tuketim_v83_sicak_optimum.csv",
        "v93": "tuketim_v93_gram_optimum.csv",
        "P1": "tuketim_p1_sicak_ilce.csv",
        "P3": "tuketim_p3_soguk_seviye.csv",
        "B": "tuketim_v96_grupb_optimum.csv",
        "bos": "tuketim_v94_bosluk_oncesi.csv",
        "v101": "tuketim_v101_hepsi.csv",
        "v102": "tuketim_v102_kappa_optimum.csv",
        "P2": "tuketim_p2_sicak_seviye.csv",
        "P4": "tuketim_p4_sicak_ay.csv",
        "P5": "tuketim_p5_soguk_kva.csv",
        "yas": "tuketim_prob_yas790.csv",
        "v82": "tuketim_v82_ayirici.csv",
        "v99": "tuketim_v99_mimari_sekil.csv",
        "v90": "tuketim_v90_temiz_sota.csv",
        "v98": "tuketim_v98_mimari_prob.csv",
        "v91": "tuketim_v91_grupb_kaldirma.csv",
        "v95": "tuketim_v95_gram_grupb.csv",
        "sota1": "tuketim_sota_v1.csv",
    }.items()
}
C = {
    "c1_v93x2": 2 * (FL["v93"] - FL["v83"]),
    "c2_P1": FL["P1"] - FL["v93"],
    "c3_P3": FL["P3"] - FL["v93"],
    "c4_grupB": FL["B"] - FL["v93"],
    "c5_bosluk": FL["bos"] - FL["v93"],
}
cad = list(C)
U5 = np.array([C[k] for k in cad])
G5 = (U5 @ U5.T) / n
Q5 = np.diag(G5).copy()
d101 = FL["v101"] - FL["v83"]
Qd = float(d101 @ d101) / n
kal = d101 - U5.sum(0)
S = 0.033643
print("\n" + "=" * 86)
print("1. DIKLIK -- 5 BILESENIN GRAM MATRISI (v83 tabanli)")
print("=" * 86)
print(f"kimlik kalintisi ||d101 - sum c_i||^2/n = {float(kal @ kal) / n:.3e}  (TAM)")
print(f"\n{'':11s}" + "".join(f"{k[:9]:>12s}" for k in cad))
for i, k in enumerate(cad):
    print(f"{k:11s}" + "".join(f"{G5[i, j]:>12.7f}" for j in range(5)))
nr = np.sqrt(Q5)
Cs = G5 / np.outer(nr, nr)
print(f"\nKOSINUS\n{'':11s}" + "".join(f"{k[:9]:>10s}" for k in cad))
for i, k in enumerate(cad):
    print(f"{k:11s}" + "".join(f"{Cs[i, j]:>10.4f}" for j in range(5)))
print(
    f"\nsum Q_i = {Q5.sum():.7f}   1'G1 = Q(d101) = {Qd:.7f}   capraz = {Qd - Q5.sum():+.7f} ({100 * (Qd - Q5.sum()) / Q5.sum():+.1f}%)"
)
print(
    f"maks |kosegen disi kosinus| = {np.abs(Cs - np.eye(5)).max():.4f}   cond(G5) = {np.linalg.cond(G5):.1f}"
)

# ---------- 2. Ayristirma kazanc araligi ----------
L1, Qp1, _ = span_L(C["c1_v93x2"], D, Gm, b)
print("\n" + "=" * 86)
print("2. AYRISTIRMA KAZANC ARALIGI")
print("=" * 86)
print(f"c1 = 2(v93-v83) OLCULMUS SPAN ICINDE: Qperp={Qp1:.3e}  ->  L1 = {L1:.7f} (TAM, LB cebiri)")
print(f"   kappa*_1 = {L1 / Q5[0]:.4f}   tek basina kazanc = {L1**2 / Q5[0]:.7f}")
S_kalan = S - L1
Qk = Q5[1:].sum()
print(f"sum_i L_i = L(d101) = {S:.7f}  ->  L2+L3+L4+L5 = {S_kalan:.7f}")
print(f"   Q2..Q5 toplami = {Qk:.7f}  ->  ORTALAMA GERCEKLESME f = {S_kalan / Qk:.4f}")
Gi = np.linalg.inv(G5)
one = np.ones(5)
print(f"\nMEVCUT (tek kappa)  = S^2/(1'G1) = {S**2 / Qd:.7f}   [OLCULDU 0.0154434]")


# kisitli min: L1 sabit, sum L_2..5 = S_kalan
def kazanc(L):
    return float(L @ Gi @ L)


from scipy.optimize import minimize

A = np.zeros((2, 5))
A[0, 0] = 1
A[1, 1:] = 1
kis = [{"type": "eq", "fun": lambda L, A=A: A @ L - np.array([L1, S_kalan])}]
x0 = np.array([L1, *(S_kalan * Q5[1:] / Qk)])
r = minimize(kazanc, x0, constraints=kis, method="SLSQP", options={"maxiter": 2000, "ftol": 1e-16})
print(f"ALT SINIR (L1 sabit, kalan esit-kappa'ya en yakin) = {kazanc(r.x):.7f}")
print(f"   L = {np.round(r.x, 7)}")
# ust sinirlar: tum kalan tek bilesende
print("\nUST SINIR SENARYOLARI (kalan L tek bilesende):")
for j in range(1, 5):
    L = np.zeros(5)
    L[0] = L1
    L[j] = S_kalan
    print(f"   hepsi {cad[j]:10s}: kazanc {kazanc(L):.7f}  kappa={S_kalan / Q5[j]:+.3f}")
# gercekci: kappa dagilimi senaryolari
print("\nGERCEKCI SENARYOLAR (kalan kappa'lar orana gore):")
for etiket, w in [
    ("esit kappa (f=0.373 hepsi)", np.array([1, 1, 1, 1.0])),
    ("P3 agir (CV -0.0207 en buyuk)", np.array([0.6, 1.8, 0.6, 0.6])),
    ("P3+B agir", np.array([0.4, 1.7, 1.5, 0.4])),
    ("P1 ters (-0.3), P3 agir", np.array([-0.3, 2.0, 1.0, 0.6])),
]:
    f = w * S_kalan / float(w @ Q5[1:])
    L = np.array([L1, *(f * Q5[1:])])
    print(
        f"   {etiket:32s} kazanc {kazanc(L):.7f}  (+{kazanc(L) - S**2 / Qd:.7f} ek)  kappa={np.round(f, 2)}"
    )
json.dump(
    {
        "L1": L1,
        "S": S,
        "S_kalan": S_kalan,
        "Q5": Q5.tolist(),
        "G5": G5.tolist(),
        "Qd": Qd,
        "span_kazanc": span_kazanc,
        "m0": m0,
        "f_bundle": S_kalan / Qk,
    },
    open(CIK / "g04.json", "w"),
    indent=2,
)
