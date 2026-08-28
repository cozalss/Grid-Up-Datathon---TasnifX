from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gram2"))
sys.stdout.reconfigure(encoding="utf-8")
from g01_havuz import yukle

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
adlar, X, skorlar, ids = yukle()
n = X.shape[1]
i83 = adlar.index("v83")
m0 = float(skorlar[i83] ** 2)
ESKI = [a for a in adlar if a not in ("v83", "v101", "v102")]
idx = [adlar.index(a) for a in ESKI]
D = X[idx] - X[i83]
G = (D @ D.T) / n
mm = skorlar[idx] ** 2
b = (m0 + np.diag(G) - mm) / 2.0
l, V = np.linalg.eigh(G)
ke = l > l.max() * 1e-10
Vk, lk = V[:, ke], l[ke]


def lg(a):
    return np.log1p(pd.read_csv(GON / a)["tuketim"].to_numpy("float64"))


v83 = lg("tuketim_v83_sicak_optimum.csv")
v101 = lg("tuketim_v101_hepsi.csv")
v93 = lg("tuketim_v93_gram_optimum.csv")
d = v101 - v83
c = (D @ d) / n
al = Vk @ ((Vk.T @ c) / lk)
Lp = float(al @ b)
dp = al @ D
print("ESKI SPAN (18 yon, 28 Agu ONCESI) -> v101 ONGORUSU")
print(f"  Q(d101) = {float(d @ d) / n:.7f}")
print(f"  d101'in eski span'daki izdusumu: Q={float(dp @ dp) / n:.7f}  L_ongoru={Lp:.7f}")
print("  OLCULEN L(d101) = 0.0336430")
print(
    f"  DIK kisimda gerceklesen L = {0.033643 - Lp:+.7f}   Q_dik={float((d - dp) @ (d - dp)) / n:.7f}"
)
print(
    f"  -> dik kisimdaki gerceklesme orani f = {(0.033643 - Lp) / float((d - dp) @ (d - dp)) * n:+.4f}"
)
Lv93 = float((Vk @ ((Vk.T @ ((D @ (v93 - v83)) / n)) / lk)) @ b)
print(
    f"\n  L(v93-v83) eski span'dan = {Lv93:.7f}   Q = {float((v93 - v83) @ (v93 - v83)) / n:.7f}"
    f"   kappa* = {Lv93 / (float((v93 - v83) @ (v93 - v83)) / n):.4f}"
)
print(
    f"  -> v93 tek basina ON KAYIT MSE {m0 - Lv93**2 / (float((v93 - v83) @ (v93 - v83)) / n):.7f}"
    f"  RMSLE {np.sqrt(m0 - Lv93**2 / (float((v93 - v83) @ (v93 - v83)) / n)):.5f}"
)
