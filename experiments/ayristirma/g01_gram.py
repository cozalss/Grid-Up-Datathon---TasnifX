"""Bes bilesenin ve aday yonlerin Gram matrisi (v83 ve v102 tabanli)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
G = KOK / "submissions"
CIK = Path(__file__).resolve().parent


def lg(ad: str):
    d = pd.read_csv(G / ad)
    assert list(d.columns) == ["id", "tuketim"], ad
    assert len(d) == 714_688, (ad, len(d))
    return np.log1p(d["tuketim"].to_numpy(dtype="float64")), d["id"]


v83, kid = lg("tuketim_v83_sicak_optimum.csv")
n = v83.size

DOSYA = {
    "v93": "tuketim_v93_gram_optimum.csv",
    "P1": "tuketim_p1_sicak_ilce.csv",
    "P3": "tuketim_p3_soguk_seviye.csv",
    "grupB": "tuketim_v96_grupb_optimum.csv",
    "bosluk": "tuketim_v94_bosluk_oncesi.csv",
    "v101": "tuketim_v101_hepsi.csv",
    "v102": "tuketim_v102_kappa_optimum.csv",
    "P2": "tuketim_p2_sicak_seviye.csv",
    "P4": "tuketim_p4_sicak_ay.csv",
    "P5": "tuketim_p5_soguk_kva.csv",
    "yas790": "tuketim_prob_yas790.csv",
    "v82": "tuketim_v82_ayirici.csv",
    "v99": "tuketim_v99_mimari_sekil.csv",
    "v90": "tuketim_v90_temiz_sota.csv",
    "v95": "tuketim_v95_gram_grupb.csv",
    "v85": "tuketim_v85_gram_rank2.csv",
}
L = {}
for k, f in DOSYA.items():
    a, i2 = lg(f)
    assert i2.equals(kid), k
    L[k] = a

v93, v101, v102 = L["v93"], L["v101"], L["v102"]

# --- 1. v101 kimligi: v101 - v93 =? (P1-v93)+(P3-v93)+(B-v93)+(bosluk-v93)
u = {k: L[k] - v93 for k in ("P1", "P3", "grupB", "bosluk")}
top = sum(u.values())
kalan = (v101 - v93) - top
print(f"KIMLIK DENETIMI  ||v101-v93 - sum(u_i)||^2/n = {float(kalan @ kalan) / n:.3e}")
print(f"                 ||v101-v93||^2/n           = {float((v101 - v93) @ (v101 - v93)) / n:.7f}")
for k, val in u.items():
    print(f"   Q(u_{k:7s} | v93 tabanli) = {float(val @ val) / n:.7f}")

# --- 2. v102 kimligi
d_demet = v101 - v83
kstar = 0.459022
kalan2 = (v102 - v83) - kstar * d_demet
print(f"\nv102 DENETIMI  ||v102-v83 - k*d||^2/n = {float(kalan2 @ kalan2) / n:.3e}")
Qd = float(d_demet @ d_demet) / n
print(f"Q(d_demet) = {Qd:.7f}   (belge 0.073292)")

# --- 3. Bes bilesen, v83 TABANLI yonler
BILESEN = {
    "v93": v93 - v83,
    "P1": u["P1"],
    "P3": u["P3"],
    "grupB": u["grupB"],
    "bosluk": u["bosluk"],
}
ad = list(BILESEN)
U = np.array([BILESEN[k] for k in ad])
Gm = U @ U.T / n
Q = np.diag(Gm).copy()
nrm = np.sqrt(Q)
C = Gm / np.outer(nrm, nrm)
print("\n=== GRAM (v83 tabanli 5 bilesen) ===")
print(f"{'':9s}" + "".join(f"{k:>12s}" for k in ad))
for i, k in enumerate(ad):
    print(f"{k:9s}" + "".join(f"{Gm[i, j]:>12.7f}" for j in range(len(ad))))
print("\n=== KOSINUS ===")
print(f"{'':9s}" + "".join(f"{k:>10s}" for k in ad))
for i, k in enumerate(ad):
    print(f"{k:9s}" + "".join(f"{C[i, j]:>10.4f}" for j in range(len(ad))))
print(f"\nsum Q_i = {Q.sum():.7f}   Q(d_demet) = {Qd:.7f}   fark = {Qd - Q.sum():+.7f}")
print(f"capraz terim toplami 2*sum_{{i<j}} G_ij = {Qd - Q.sum():+.7f}")
print(f"cond(G) = {np.linalg.cond(Gm):.3f}")

# --- 4. TUM aday yonler, v102 TABANLI
ADAY = {
    "v93": v93 - v83,
    "P1": u["P1"],
    "P3": u["P3"],
    "grupB": u["grupB"],
    "bosluk": u["bosluk"],
    "P2": L["P2"] - v93,
    "P4": L["P4"] - v93,
    "P5": L["P5"] - v93,
    "yas790": L["yas790"] - v93,
    "v82": L["v82"] - v83,
    "v99": L["v99"] - v83,
    "v90": L["v90"] - v83,
    "v85": L["v85"] - v83,
}
ad2 = list(ADAY)
U2 = np.array([ADAY[k] for k in ad2])
G2 = U2 @ U2.T / n
Q2 = np.diag(G2).copy()
n2 = np.sqrt(Q2)
C2 = G2 / np.outer(n2, n2)
print("\n=== TUM ADAYLAR: Q ve d_demet ile kosinus ===")
dd = d_demet / np.sqrt(Qd)
print(f"{'yon':10s}{'Q':>12s}{'cos(d_demet)':>14s}{'<u,d_demet>/n':>16s}")
for i, k in enumerate(ad2):
    c = float(ADAY[k] @ dd) / n / n2[i]
    print(f"{k:10s}{Q2[i]:>12.7f}{c:>14.4f}{float(ADAY[k] @ d_demet) / n:>16.7f}")
print("\n=== KOSINUS (tum adaylar) ===")
print(f"{'':9s}" + "".join(f"{k[:6]:>8s}" for k in ad2))
for i, k in enumerate(ad2):
    print(f"{k:9s}" + "".join(f"{C2[i, j]:>8.3f}" for j in range(len(ad2))))

json.dump(
    {
        "ad": ad,
        "G": Gm.tolist(),
        "Q": Q.tolist(),
        "Q_demet": Qd,
        "ad2": ad2,
        "G2": G2.tolist(),
        "Q2": Q2.tolist(),
        "kimlik_kalan": float(kalan @ kalan) / n,
    },
    open(CIK / "g01_gram.json", "w"),
    indent=2,
)
print(f"\nYAZILDI {CIK / 'g01_gram.json'}")
