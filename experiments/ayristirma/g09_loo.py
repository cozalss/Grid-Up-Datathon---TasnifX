"""Span cozumunun BIRINI-DISARIDA-BIRAK sinavi: skoru tahmin et."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gram2"))
sys.stdout.reconfigure(encoding="utf-8")
from g01_havuz import yukle

adlar, X, skorlar, ids = yukle()
n = X.shape[1]
i83 = adlar.index("v83")
m0 = float(skorlar[i83] ** 2)
yon = [a for a in adlar if a != "v83"]
idx = [adlar.index(a) for a in yon]
D = X[idx] - X[i83]
Gm = (D @ D.T) / n
mm = skorlar[idx] ** 2
b = (m0 + np.diag(Gm) - mm) / 2.0
K = len(yon)
hata = []
print(f"{'disarida':8s}{'gercek':>10s}{'tahmin':>10s}{'hata':>10s}")
for j in range(K):
    tut = [i for i in range(K) if i != j]
    Gt = Gm[np.ix_(tut, tut)]
    bt = b[tut]
    l, Vv = np.linalg.eigh(Gt)
    ke = l > l.max() * 1e-10
    Vk, lk = Vv[:, ke], l[ke]
    cj = Gm[tut, j]
    al = Vk @ ((Vk.T @ cj) / lk)
    b_tah = float(al @ bt)  # b_j tahmini (izdusum)
    m_tah = m0 + Gm[j, j] - 2 * b_tah
    s_tah = np.sqrt(max(m_tah, 0))
    s = skorlar[idx][j]
    hata.append(s_tah - s)
    print(f"{yon[j]:8s}{s:>10.5f}{s_tah:>10.5f}{s_tah - s:>+10.5f}")
h = np.array(hata)
print(
    f"\nort |hata| = {np.abs(h).mean():.5f}   medyan |hata| = {np.median(np.abs(h)):.5f}   maks = {np.abs(h).max():.5f}"
)
print(f"yanlilik (ort hata) = {h.mean():+.5f}")
