"""G6 -- v93 gercekten ESKI span icinde miydi? v101'in span-disi kazanci ne kadardi?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from g03_sinav import kur  # noqa: E402

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"


def proj(Gs, gj, Qj):
    lam, U = np.linalg.eigh(Gs)
    s = np.argsort(lam)[::-1]
    lam, U = lam[s], U[:, s]
    tut = lam > lam[0] * 1e-10
    inv = np.zeros_like(lam)
    inv[tut] = 1.0 / lam[tut]
    c = U @ (inv * (U.T @ gj))
    artik = Qj - float(gj @ c)
    return c, artik, int(tut.sum())


H = kur("v83")
yon, G, b, m0, D, n = H["yon"], H["G"], H["b"], H["m0"], H["D"], H["n"]
x0 = H["X"][H["i0"]]
ESKI = [
    "v2",
    "v7",
    "v15",
    "v16",
    "v18",
    "v25",
    "v27",
    "v30",
    "v46",
    "v44",
    "v47",
    "v50",
    "v55",
    "v67",
    "v73",
    "v79",
    "v80",
    "v81",
]
iE = [yon.index(a) for a in ESKI]

d93 = np.log1p(pd.read_csv(GON / "tuketim_v93_gram_optimum.csv")["tuketim"].to_numpy("f8")) - x0
Q93 = float(d93 @ d93 / n)
g93 = (D @ d93) / n
c, art, rk = proj(G[np.ix_(iE, iE)], g93[iE], Q93)
print("=" * 88)
print("v93 -- ESKI 18 yonun span'inda mi?")
print(
    f"  Q93={Q93:.8f}  artik={art:.3e}  artik/Q={art / Q93:.3e}  rank={rk}  |c|_1={np.abs(c).sum():.3f}"
)
b93 = float(g93[iE] @ np.linalg.lstsq(G[np.ix_(iE, iE)], b[iE], rcond=None)[0])
print(f"  -> v93 SPAN ICI: b93 = sum c_i b_i = {float(c @ b[iE]):+.8f}")
print(
    f"     MSE = m0 + Q93 - 2b93 = {m0 + Q93 - 2 * float(c @ b[iE]):.8f}  RMSLE = {np.sqrt(m0 + Q93 - 2 * float(c @ b[iE])):.6f}"
)
sdc = float(np.sqrt(c @ H["C"][np.ix_(iE, iE)] @ c))
print(
    f"     yuvarlamadan sd(b93)={sdc:.2e} -> RMSLE belirsizligi +-{sdc / np.sqrt(m0 + Q93 - 2 * float(c @ b[iE])):.6f}"
)

print("\n" + "=" * 88)
print("v101'in ESKI span'a gore span-DISI bileseni ve gercek kazanci")
i101 = yon.index("v101")
d101 = D[i101]
Q101 = float(G[i101, i101])
c1, art1, _ = proj(G[np.ix_(iE, iE)], G[i101, iE], Q101)
b101 = float(b[i101])
b101_ici = float(c1 @ b[iE])
print(f"  Q101={Q101:.6f}  span-disi artik={art1:.6f} ({art1 / Q101 * 100:.1f}%)")
print(f"  b101 (OLCULEN)      = {b101:+.6f}")
print(f"  b101 span-ici kismi = {b101_ici:+.6f}")
print(
    f"  b_perp (YENI SINYAL)= {b101 - b101_ici:+.6f}   -> tek basina kazanc L^2/Q = "
    f"{(b101 - b101_ici) ** 2 / art1:.6f} MSE"
)
print("  eski span icinde ulasilabilir kazanc  = 0.009797 MSE")
print(f"  v101 sonrasi ulasilabilir kazanc      = 0.015838 MSE  (+{0.015838 - 0.009797:.6f})")
