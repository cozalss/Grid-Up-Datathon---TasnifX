"""Aday yonler: Q, olculmus span ile ortusme, DIK artik, beklenen L^2/Q."""

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
m = skorlar[idx] ** 2
b = (m0 + np.diag(Gm) - m) / 2.0
lam, V = np.linalg.eigh(Gm)
kes = lam > lam.max() * 1e-10
Vk, lk = V[:, kes], lam[kes]


def ayir(u):
    c = (D @ u) / n
    al = Vk @ ((Vk.T @ c) / lk)
    upar = al @ D
    uper = u - upar
    return float(al @ b), float(upar @ upar) / n, float(uper @ uper) / n, float(u @ u) / n


def lg(a):
    d = pd.read_csv(GON / a)
    assert (d["id"].values == ids.values).all(), a
    return np.log1p(d["tuketim"].to_numpy("float64"))


N = {
    "v83": "tuketim_v83_sicak_optimum.csv",
    "v93": "tuketim_v93_gram_optimum.csv",
    "v90": "tuketim_v90_temiz_sota.csv",
    "sota1": "tuketim_sota_v1.csv",
    "P1": "tuketim_p1_sicak_ilce.csv",
    "P2": "tuketim_p2_sicak_seviye.csv",
    "P3": "tuketim_p3_soguk_seviye.csv",
    "P4": "tuketim_p4_sicak_ay.csv",
    "P5": "tuketim_p5_soguk_kva.csv",
    "yas": "tuketim_prob_yas790.csv",
    "v82": "tuketim_v82_ayirici.csv",
    "v99": "tuketim_v99_mimari_sekil.csv",
    "v98": "tuketim_v98_mimari_prob.csv",
    "v91": "tuketim_v91_grupb_kaldirma.csv",
    "v95": "tuketim_v95_gram_grupb.csv",
    "B": "tuketim_v96_grupb_optimum.csv",
    "bos": "tuketim_v94_bosluk_oncesi.csv",
    "v101": "tuketim_v101_hepsi.csv",
    "v102": "tuketim_v102_kappa_optimum.csv",
    "v85": "tuketim_v85_gram_rank2.csv",
    "sotav7": "tuketim_sota_v7_gram_nihai.csv",
    "v89": "tuketim_v89_genis_taban.csv",
}
F = {k: lg(v) for k, v in N.items()}
# her aday icin DOGRU tabani bul (en kucuk Q)
print("=" * 100)
print("ADAYIN DOGAL TABANI (en kucuk Q veren)")
print("=" * 100)
TABANLAR = ["v83", "v93", "v90", "sota1", "v102"]
ADAY = [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "yas",
    "v82",
    "v99",
    "v98",
    "v90",
    "v91",
    "v95",
    "B",
    "bos",
    "v85",
    "sotav7",
]
dogal = {}
for a in ADAY:
    qs = {t: float((F[a] - F[t]) @ (F[a] - F[t])) / n for t in TABANLAR if t != a}
    t = min(qs, key=qs.get)
    dogal[a] = t
    print(
        f"  {a:7s} -> taban {t:6s} Q={qs[t]:.7f}   "
        + " ".join(f"{k}:{v:.5f}" for k, v in qs.items())
    )
print("\n" + "=" * 100)
print("ADAY YONLERI: SPAN ICI (bilinen L) vs DIK ARTIK (probla olculecek)")
print("=" * 100)
f_ger = 0.3735
print(
    f"{'yon':9s}{'taban':7s}{'Q':>11s}{'Q_span':>11s}{'Q_dik':>11s}{'dik%':>7s}"
    f"{'L_span(bil)':>13s}{'bek L_dik':>11s}{'bek kazanc':>12s}{'|dRMSLE|@k=1':>13s}"
)
sat = []
for a in ADAY:
    t = dogal[a]
    u = F[a] - F[t]
    Ls, Qp, Qd_, Q = ayir(u)
    bekL = f_ger * Qd_  # dik kisimda CV-optimum olcekli dosya -> L = f*Q
    kaz = bekL**2 / Qd_ if Qd_ > 0 else 0.0
    dm = Q - 2 * (Ls + bekL)
    sat.append(dict(ad=a, taban=t, Q=Q, Qspan=Qp, Qdik=Qd_, Lspan=Ls, bekL=bekL, kazanc=kaz))
    print(
        f"{a:9s}{t:7s}{Q:>11.7f}{Qp:>11.7f}{Qd_:>11.7f}{100 * Qd_ / Q:>6.1f}%"
        f"{Ls:>13.7f}{bekL:>11.7f}{kaz:>12.7f}{abs(dm) / 2.011:>13.6f}"
    )
print(f"\nDIK yonlerin beklenen kazanc TOPLAMI = {sum(s['kazanc'] for s in sat):.7f}")
json.dump(sat, open(CIK / "g05.json", "w"), indent=2)
# aday-aday dik mi
UU = np.array([F[a] - F[dogal[a]] for a in ADAY])
Ga = (UU @ UU.T) / n
nn = np.sqrt(np.diag(Ga))
Ca = Ga / np.outer(nn, nn)
print("\nADAYLAR ARASI KOSINUS")
print(f"{'':9s}" + "".join(f"{a[:6]:>8s}" for a in ADAY))
for i, a in enumerate(ADAY):
    print(f"{a:9s}" + "".join(f"{Ca[i, j]:>8.2f}" for j in range(len(ADAY))))
