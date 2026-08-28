"""D: yeni yonun OLCULMUS LB span'ina ve mevcut prob envanterine DIK payi."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "gram2"))
from g01_havuz import yukle  # noqa

GON = KOK / "submissions"
adlar, X, skorlar, ids = yukle()
n = X.shape[1]
i83 = adlar.index("v83")
idx = [adlar.index(a) for a in adlar if a != "v83"]
Dm = X[idx] - X[i83]
Gm = (Dm @ Dm.T) / n
lam, V = np.linalg.eigh(Gm)
kes = lam > lam.max() * 1e-10
Vk, lk = V[:, kes], lam[kes]
print(f"havuz {len(adlar)} dosya | span rank {int(kes.sum())}")


def lg(f):
    d = pd.read_csv(GON / f)
    return np.log1p(d["tuketim"].to_numpy("float64"))


F = {
    k: lg(v)
    for k, v in {
        "v102": "tuketim_v102_kappa_optimum.csv",
        "v93": "tuketim_v93_gram_optimum.csv",
        "v83": "tuketim_v83_sicak_optimum.csv",
        "v90": "tuketim_v90_temiz_sota.csv",
        "P1": "tuketim_p1_sicak_ilce.csv",
        "P2": "tuketim_p2_sicak_seviye.csv",
        "P3": "tuketim_p3_soguk_seviye.csv",
        "P4": "tuketim_p4_sicak_ay.csv",
        "P5": "tuketim_p5_soguk_kva.csv",
        "yas": "tuketim_prob_yas790.csv",
        "v82": "tuketim_v82_ayirici.csv",
        "v99": "tuketim_v99_mimari_sekil.csv",
        "B": "tuketim_v96_grupb_optimum.csv",
        "bos": "tuketim_v94_bosluk_oncesi.csv",
        "v107": "tuketim_v107_soguk_onarim.csv",
    }.items()
}


def perp(u):
    c = (Dm @ u) / n
    al = Vk @ ((Vk.T @ c) / lk)
    return u - al @ Dm


ENV = [
    ("P1c", F["P1"] - F["v93"]),
    ("P3c", F["P3"] - F["v93"]),
    ("Bc", F["B"] - F["v93"]),
    ("bosc", F["bos"] - F["v93"]),
    ("v90", F["v90"] - F["v83"]),
    ("P2", F["P2"] - F["v93"]),
    ("yas", F["yas"] - F["v93"]),
    ("v99", F["v99"] - F["v90"]),
    ("P4", F["P4"] - F["v93"]),
    ("v82", F["v82"] - F["v83"]),
    ("P5", F["P5"] - F["v93"]),
]
E, top = [], 0.0
for ad, u in ENV:
    v = perp(u)
    for e in E:
        v -= float(v @ e) / n * e
    Qy = float(v @ v) / n
    if Qy > 1e-8:
        E.append(v / np.sqrt(Qy))
    top += Qy
print(f"mevcut envanter YENI Q toplami = {top:.7f}  ({len(E)} boyut)")

u = F["v107"] - F["v102"]
Q = float(u @ u) / n
v = perp(u)
Qd = float(v @ v) / n
for e in E:
    v -= float(v @ e) / n * e
Qy = float(v @ v) / n
print("\nv107 - v102:")
print(f"  Q_dosya                    = {Q:.7f}")
print(f"  Q (LB span'ina dik)        = {Qd:.7f}   (%{100 * Qd / Q:.1f})")
print(f"  Q (span + envantere DIK)   = {Qy:.7f}   (%{100 * Qy / Q:.1f})")
print(f"  envantere katkisi          = {100 * Qy / top:.1f}%  -> yeni toplam {top + Qy:.7f}")
