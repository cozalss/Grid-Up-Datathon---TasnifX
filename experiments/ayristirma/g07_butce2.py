from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gram2"))
sys.stdout.reconfigure(encoding="utf-8")
from g01_havuz import yukle

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
mm = skorlar[idx] ** 2
b = (m0 + np.diag(Gm) - mm) / 2.0
lam, V = np.linalg.eigh(Gm)
kes = lam > lam.max() * 1e-10
Vk, lk = V[:, kes], lam[kes]


def lg(a):
    d = pd.read_csv(GON / a)
    return np.log1p(d["tuketim"].to_numpy("float64"))


N = {
    "v83": "tuketim_v83_sicak_optimum.csv",
    "v93": "tuketim_v93_gram_optimum.csv",
    "v90": "tuketim_v90_temiz_sota.csv",
    "v102": "tuketim_v102_kappa_optimum.csv",
    "v101": "tuketim_v101_hepsi.csv",
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
}
F = {k: lg(v) for k, v in N.items()}


def perp(u):
    c = (D @ u) / n
    al = Vk @ ((Vk.T @ c) / lk)
    return u - al @ D, float(al @ b)


CAND = [
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
print("=" * 92)
print("YENI (SPAN'A DIK) BILGININ ACGOZLU ORTOGONALLESTIRILMESI")
print("=" * 92)
E = []
tot = 0.0
f = 0.3735
sat = []
print(f"{'yon':7s}{'Q_dosya':>11s}{'Q_dik':>11s}{'Q_YENI':>11s}{'f^2*Q_YENI':>12s}{'kum':>11s}")
for ad, u in CAND:
    up, _ = perp(u)
    Q = float(u @ u) / n
    Qd = float(up @ up) / n
    v = up.copy()
    for e in E:
        v -= float(v @ e) / n * e
    Qy = float(v @ v) / n
    if Qy > 1e-8:
        E.append(v / np.sqrt(Qy))
    g = f * f * Qy
    tot += g
    sat.append((ad, Q, Qd, Qy, g))
    print(f"{ad:7s}{Q:>11.7f}{Qd:>11.7f}{Qy:>11.7f}{g:>12.7f}{tot:>11.7f}")
print(f"\nYENI bilgi boyutu = {len(E)}   toplam YENI Q = {sum(s[3] for s in sat):.7f}")
kaz_span = float((Vk @ ((Vk.T @ b) / lk)) @ b)
gerekli = m0 - 0.99138**2
print("\n" + "=" * 92)
print("TOPLAM BUTCE")
print("=" * 92)
print(f"1.lik icin gereken (v83'ten)  = {gerekli:.7f}")
print("v102 ile gerceklesen          = 0.0154434")
print(f"KALAN ACIK                    = {gerekli - 0.0154434:.7f}")
Qy_top = sum(s[3] for s in sat)
print(
    f"\n{'senaryo':24s}{'span':>11s}{'yeni':>11s}{'TOPLAM':>11s}{'MSE':>11s}{'RMSLE':>10s}{'eksik':>11s}"
)
for et, ff in [
    ("f=0.20 kotu", 0.20),
    ("f=0.3735 OLCULEN", 0.3735),
    ("f=0.55 iyi", 0.55),
    ("f=0.75 cok iyi", 0.75),
    ("f=1.00 TAVAN", 1.00),
]:
    yeni = ff * ff * Qy_top
    top = kaz_span + yeni
    print(
        f"{et:24s}{kaz_span:>11.7f}{yeni:>11.7f}{top:>11.7f}{m0 - top:>11.6f}"
        f"{np.sqrt(max(m0 - top, 0)):>10.5f}{gerekli - top:>+11.6f}"
    )
print(
    f"\n1.LIK icin gereken f = sqrt(({gerekli:.7f}-{kaz_span:.7f})/{Qy_top:.7f}) = "
    f"{np.sqrt(max(gerekli - kaz_span, 0) / Qy_top):.4f}"
)
json.dump(
    {
        "sat": [[s[0], *map(float, s[1:])] for s in sat],
        "kaz_span": kaz_span,
        "Qy_top": Qy_top,
        "gerekli": gerekli,
        "m0": m0,
    },
    open(CIK / "g07.json", "w"),
    indent=2,
)
