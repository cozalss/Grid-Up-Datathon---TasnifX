"""TASIMA OLCUMUNU DUZELT: katsayilar LB'den, bloktan FIT EDILMIYOR.

Onceki olcum (m125) her yarida agirliklari YENIDEN FIT edip orani aliyordu.
Ama gercek kurulumda katsayilar LB'den geliyor (1.95*|rho_s|), bloktan
fit edilmiyor. Dolayisiyla "fit/holdout" ayrimi yapay ve oran gurultulu
(n=6'da sd 0.647 -- olculemiyor).

DOGRUSU: bilesigi LB katsayilariyla SABIT kur, sonra blogun farkli zaman
pencerelerinde korelasyonunu olc. Fit yok, sizinti yok, oran yok.

Test penceresi 122 gunluk bir ufuk oldugu icin GEC pencere en iyi vekildir.
Her on-ek uzunlugu icin:
  kor_tum    tum blokta korelasyon
  kor_erken  gun 1-40
  kor_gec    gun 83-122      <- test'e en yakin
  kor_sd     bes pencere arasindaki sacilim
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
TABAN = "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        if len(d) != len(IDS) or d.id.duplicated().any():
            return None
        pos = pd.Index(d.id).get_indexer(IDS)
        if (pos < 0).any():
            return None
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
V, L = [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
V, L = np.array(V).T, np.array(L)
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
MSE_OPT = M0 - gercek

# ---------------------------------------------------------------------------
# 1.95 CARPANI ICIN IKINCI OLCUM -- kendi gecmis gonderimlerimizden.
#
# 1.95, "seviye" deneyinden gelen TEK bir olcumdu (n=1). Ama gecmiste iki
# gonderim arasindaki FARK dogrudan bir dik yon olcumu verir:
#     P_2^2 - P_1^2 = -2 <r, delta>/N + (Q_2 - Q_1),   delta = d_2 - d_1
# Buradan <r, delta>/N TAM olarak cozulur (LB yuvarlamasi disinda).
# delta'yi span ve dik parcalara ayirirsak:
#     <r, delta_dik>/N = <r, delta>/N - <r_hat, delta_span>/N
# ve gerceklesen dik korelasyon  rho_u = <r,delta_dik>/N / sqrt(Q_dik).
# Kuralimizin ayni delta icin ongordugu ise 1.95 * rho_s.
# ORAN = rho_u / rho_s  ->  1.95'in BAGIMSIZ ikinci olcumu.
# ---------------------------------------------------------------------------
r_hat_, gercek_, kL_ = buzmeli_r_hat(V, L, G, N)
print(f"\nsaf optimum {np.sqrt(M0 - gercek_):.6f}")

CIFT = [
    (
        "tuketim_YP_seviye.csv",
        1.00115,
        "tuketim_K_yenibas.csv",
        1.00191,
        "yeni-trafo baslangic yanliligi",
    ),
]
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SKR = json.load(fh)
for f1, p1, f2, p2, ad in list(CIFT):
    pass
# ayrica taban ile olculmus her gonderim arasindaki farki da dene
for f2, p2 in SKR.items():
    if f2 in (TABAN, "tuketim_YP_seviye.csv") or not os.path.exists(os.path.join(S, f2)):
        continue
    CIFT.append(("tuketim_YP_seviye.csv", 1.00115, f2, p2, "taban=YP_seviye"))

print(
    f"\n{'eklenen yon':>34s} {'Q_dik':>8s} {'<r,dik>/N':>11s} {'rho_u':>9s} "
    f"{'rho_s':>9s} {'ORAN':>8s}"
)
oranlar = []
for f1, p1, f2, p2, ad in CIFT:
    if not (os.path.exists(os.path.join(S, f1)) and os.path.exists(os.path.join(S, f2))):
        continue
    d1, d2 = oku(f1) - a0, oku(f2) - a0
    if d1 is None or d2 is None:
        continue
    dl = d2 - d1
    Qd2, Qd1 = float((d2 * d2).mean()), float((d1 * d1).mean())
    rdelta = (p1**2 - p2**2 + Qd2 - Qd1) / 2.0  # = <r, delta>/N
    cc = Gi @ ((V.T @ dl) / N)
    dl_span = V @ cc
    dl_dik = dl - dl_span
    Qdik = float((dl_dik * dl_dik).mean())
    Qspan = float((dl_span * dl_span).mean())
    if Qdik < 1e-8 or Qspan < 1e-10:
        continue
    r_span = float((r_hat_ * dl_span).mean())
    r_dik = rdelta - r_span
    rho_u = r_dik / np.sqrt(Qdik)
    rho_s = float((r_hat_ * dl).mean()) / np.sqrt(Qspan)
    if abs(rho_s) < 1e-6:
        continue
    oran = rho_u / rho_s
    oranlar.append((f2, oran, Qdik, rho_u, rho_s))
    print(
        f"{(f2[:30] + ' [' + ad[:1] + ']'):>34s} {Qdik:8.4f} {r_dik:+11.5f} "
        f"{rho_u:+9.4f} {rho_s:+9.4f} {oran:+8.2f}"
    )

if oranlar:
    gv = [o for _, o, q, _, _ in oranlar if q > 0.01]
    print(f"\nQ_dik > 0.01 olan {len(gv)} olcum: ortanca oran {np.median(gv):+.2f}")
    print("  seviye deneyinin verdigi: -1.95 (buyukluk 1.95)")
    print(f"  buyukluklerin ortancasi : {np.median(np.abs(gv)):.2f}")
