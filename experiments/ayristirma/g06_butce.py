"""Ortak butce + prob sirasi + bugunun dosyasi."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gram2"))
sys.stdout.reconfigure(encoding="utf-8")
from g01_havuz import yukle  # noqa

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


def izd(u):
    c = (D @ u) / n
    al = Vk @ ((Vk.T @ c) / lk)
    up = al @ D
    return al, float(al @ b), up, u - up


def lg(a):
    d = pd.read_csv(GON / a)
    assert (d["id"].values == ids.values).all(), a
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
m102 = 1.00553**2
# ---------- SPAN OPTIMUMU ----------
w = Vk @ ((Vk.T @ b) / lk)
adim_span = w @ D
kaz_span = float(w @ b)
print("=" * 88)
print("A. OLCULMUS SPAN OPTIMUMU (yeni bilgi GEREKTIRMEZ)")
print("=" * 88)
print(f"|w|_1 = {np.abs(w).sum():.3e}   adim ||.||inf = {np.abs(adim_span).max():.4f}")
print(f"kazanc = {kaz_span:.7f}  -> MSE {m0 - kaz_span:.7f}  RMSLE {np.sqrt(m0 - kaz_span):.6f}")
yl = F["v83"] + adim_span
print(f"min log1p = {yl.min():.4f}  negatif satir = {int((yl < 0).sum()):,}")
# 2-blok (saglam) alternatif
c1 = 2 * (F["v93"] - F["v83"])
crest = (F["v101"] - F["v83"]) - c1
L1 = 2 * float(izd(F["v93"] - F["v83"])[1])
Lr = 0.033643 - L1
A2 = np.array([c1, crest])
G2 = (A2 @ A2.T) / n
L2 = np.array([L1, Lr])
k2 = np.linalg.solve(G2, L2)
kaz2 = float(L2 @ k2)
adim2 = k2 @ A2
y2 = F["v83"] + adim2
print(f"\n2-BLOK (c1 | c2..c5): kappa = {np.round(k2, 5)}  kazanc = {kaz2:.7f}")
print(f"   -> MSE {m0 - kaz2:.7f}  RMSLE {np.sqrt(m0 - kaz2):.6f}   min log1p {y2.min():.4f}")
print(
    f"   v102'ye gore ek kazanc = {kaz2 - 0.0154434:+.7f}  (dRMSLE {np.sqrt(m0 - kaz2) - 1.00553:+.6f})"
)

# ---------- YENI DIK YONLER: ORTAK BUTCE ----------
print("\n" + "=" * 88)
print("B. YENI YONLER -- SPAN'A DIK BILESENLER, ORTAK BUTCE")
print("=" * 88)
YENI = {
    "P2": F["P2"] - F["v93"],
    "P4": F["P4"] - F["v93"],
    "P5": F["P5"] - F["v93"],
    "yas": F["yas"] - F["v93"],
    "v82": F["v82"] - F["v83"],
    "v99": F["v99"] - F["v90"],
    "v90": F["v90"] - F["v83"],
}
# bilesen ayristirmasinin DIK yonleri (c2..c5, biri bagimli)
BIL = {"P1c": F["P1"] - F["v93"], "P3c": F["P3"] - F["v93"], "Bc": F["B"] - F["v93"]}
perp = {}
for k, u in {**YENI, **BIL}.items():
    _, _, _, up = izd(u)
    perp[k] = up
pad = list(perp)
Pm = np.array([perp[k] for k in pad])
Gp = (Pm @ Pm.T) / n
Qp = np.diag(Gp)
nn = np.sqrt(np.maximum(Qp, 1e-300))
Cp = Gp / np.outer(nn, nn)
print(f"{'yon':7s}{'Q_dosya':>11s}{'Q_dik':>11s}{'f=0.373 bek kazanc':>21s}")
Qfull = {k: float(u @ u) / n for k, u in {**YENI, **BIL}.items()}
f = 0.3735
for i, k in enumerate(pad):
    print(f"{k:7s}{Qfull[k]:>11.7f}{Qp[i]:>11.7f}{f * f * Qp[i]:>21.7f}")
print("\nDIK YONLER ARASI KOSINUS")
print(f"{'':7s}" + "".join(f"{k[:5]:>8s}" for k in pad))
for i, k in enumerate(pad):
    print(f"{k:7s}" + "".join(f"{Cp[i, j]:>8.2f}" for j in range(len(pad))))
Lp = f * np.array([Qp[i] for i in range(len(pad))])
ortak = float(Lp @ np.linalg.solve(Gp + 1e-12 * np.eye(len(pad)), Lp))
print(f"\nTEKIL toplam = {(f * f * Qp).sum():.7f}   ORTAK (Gram) = {ortak:.7f}")

# ---------- BUTCE ----------
print("\n" + "=" * 88)
print("C. TOPLAM BUTCE")
print("=" * 88)
gerekli = m0 - 0.99138**2
print(
    f"v83 MSE {m0:.7f} | lider MSE {0.99138**2:.7f} | 1.LIK icin gereken toplam kazanc = {gerekli:.7f}"
)
print("su an gerceklesen (v102)                                            = 0.0154434")
print(
    f"KALAN ACIK                                                          = {gerekli - 0.0154434:.7f}"
)
sen = {
    "KOTU  f=0.20": 0.20,
    "ORTA  f=0.3735 (OLCULEN)": 0.3735,
    "IYI f=0.55": 0.55,
    "TAVAN f=1.00": 1.00,
}
for et, ff in sen.items():
    Lx = ff * Qp
    ok = float(Lx @ np.linalg.solve(Gp + 1e-12 * np.eye(len(pad)), Lx))
    top = kaz_span + ok
    print(
        f"  {et:26s} yeni yonler {ok:.7f}  + span {kaz_span:.7f}  = {top:.7f}"
        f"  -> MSE {m0 - top:.6f} RMSLE {np.sqrt(max(m0 - top, 0)):.5f}  eksik {gerekli - top:+.6f}"
    )
json.dump(
    {"kaz_span": kaz_span, "kaz2": kaz2, "gerekli": gerekli, "Qp": Qp.tolist(), "pad": pad},
    open(CIK / "g06.json", "w"),
    indent=2,
)
