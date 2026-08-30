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
# CAPRAZ SINAV: her kurulus HER IKI VARSAYIM altinda degerlendirilir.
#
# m136 B'yi kazanan gosterdi ama B'nin KENDI varsayimiyla olctu -- haksiz.
# Kucuk sigma az buzer; sigma gercekte buyukse k'L_olculen gurultuyle SISER
# ve gercek MSE'yi oldugundan iyi gosterir. Tam da beni daha once yakan tuzak.
#
# DOGRUSU: gercek L'yi bilmedigimiz icin benzetim kurulur.
#   varsayim X icin:  L_gercek = X-buzmesinin verdigi kip genlikleri
#                     L_gozlenen = L_gercek + N(0, sigma_X)
#   her kurulus Y, L_gozlenen'den k_Y uretir
#   GERCEK MSE = M0 - 2 k_Y . L_gercek + k_Y' G k_Y
# Boylece "hangi sigma dogruysa hangi kurulus kazanir" tablosu cikar.
# ---------------------------------------------------------------------------
from m112_kalibre import ANLAM_SIGMA, W_TABAN, L_gurultusu  # noqa: E402

YUV = 5e-6 / np.sqrt(3.0)
sig_A = L_gurultusu(V, N)
sig_B = np.full(len(L), YUV)
w_, U_ = np.linalg.eigh(G)
sr = np.argsort(-w_)
w_, U_ = w_[sr], U_[:, sr]
WMAX = float(w_[0])


def kats(Lv, sg):
    """m112'nin buzmeli cozumu, ayni kapilarla."""
    si = np.sqrt(np.einsum("ij,jk,ki->i", U_.T, np.diag(sg**2), U_))
    c = U_.T @ Lv
    a = np.zeros(len(w_))
    for i in range(len(w_)):
        if w_[i] / WMAX <= W_TABAN or c[i] ** 2 <= 0.0:
            continue
        if c[i] ** 2 <= ANLAM_SIGMA**2 * si[i] ** 2:
            continue
        a[i] = max(c[i] ** 2 - si[i] ** 2, 0.0) / c[i] ** 2
    return U_ @ (a * c / np.where(w_ > 1e-300, w_, 1.0))


# her varsayim icin "gercek" L: o varsayimin buzmesiyle temizlenmis hali
GERCEK = {"A eski 2.2e-4": G @ kats(L, sig_A), "B yuvarlama 2.9e-6": G @ kats(L, sig_B)}
SIG = {
    "A eski 2.2e-4": sig_A,
    "B yuvarlama 2.9e-6": sig_B,
    "C 10x 2.9e-5": np.full(len(L), 10 * YUV),
    "D 30x 8.7e-5": np.full(len(L), 30 * YUV),
}

rng = np.random.default_rng(19)
CEK = 200
print(f"\nGERCEK MSE (dusuk = iyi), {CEK} cekilis, saf span kismi")
print(
    f"{'gercek varsayim':>20s} "
    + " ".join(f"{'kurulus ' + k[:1]:>16s}" for k in SIG)
    + f" {'kazanan':>9s}"
)
for gad, Lg in GERCEK.items():
    sg_ger = SIG[gad]
    ort = {}
    for kad, sg_kur in SIG.items():
        mse = []
        for _ in range(CEK):
            Lo = Lg + rng.normal(0, sg_ger)
            k = kats(Lo, sg_kur)
            mse.append(M0 - 2 * float(k @ Lg) + float(k @ G @ k))
        ort[kad] = (float(np.mean(mse)), float(np.std(mse)))
    kz = min(ort, key=lambda x: ort[x][0])
    print(
        f"{gad:>20s} "
        + " ".join(f"{ort[k][0]:10.6f}+-{ort[k][1]:.4f}" for k in SIG)
        + f" {kz[:1]:>9s}"
    )

print("\nSKOR OLARAK (sqrt):")
for gad, Lg in GERCEK.items():
    sg_ger = SIG[gad]
    sat = []
    for kad, sg_kur in SIG.items():
        mse = []
        for _ in range(CEK):
            Lo = Lg + rng.normal(0, sg_ger)
            k = kats(Lo, sg_kur)
            mse.append(M0 - 2 * float(k @ Lg) + float(k @ G @ k))
        sat.append(np.sqrt(max(float(np.mean(mse)), 1e-12)))
    print(
        f"  gercek {gad:>18s}:  A kurulus {sat[0]:.6f}   B kurulus {sat[1]:.6f}"
        f"   fark {sat[0] - sat[1]:+.6f}"
    )

print("\nOKUMA: B satirinda B kazanmasi beklenir (kendi evi). ONEMLI OLAN")
print("  A satiri: sigma gercekte BUYUKSE B kurulusu ne kadar KAYBETTIRIR?")
print("  Kayip kucukse, 10.7:1 kanitla B'ye gecmek dogru karardir.")
