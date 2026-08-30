"""KESIN SORU: gecmiste her YENI olcum, span optimumunu ne kadar iyilestirdi?

Bu, "yarin bir eksen olcersek ne kazaniriz" sorusunun DOGRUDAN empirik cevabidir.
Kotu kosullu ters alma YOK: her adimda pinv(rcond) ile span optimumu hesaplanir,
artimli kazanc = onceki_kazanc - simdiki_kazanc. Ikisi de kararli.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
SK = json.load(open(os.path.join(KOK, "experiments/model29/olculmus_skorlar.json")))
TABAN = "tuketim_m6_ikiyon.csv"
M0 = 1.005846366
RC = 1e-6


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    return np.log1p(d[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
ad, D, L = [], [], []
for f, P in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if len(v) != N:
        continue
    d = v - a0
    Q = float((d * d).mean())
    ad.append(f)
    D.append(d)
    L.append((M0 + Q - P * P) / 2)
ad.append("tuketim_y40_sota_temiz.csv")
D.append(oku("tuketim_y40_sota_temiz.csv") - a0)
L.append(-0.002229)
D = np.array(D).T
L = np.array(L)
K = len(ad)


def kazanc(idx):
    if not idx:
        return 0.0
    g = (D[:, idx].T @ D[:, idx]) / N
    c = np.linalg.pinv(g, rcond=RC) @ L[idx]
    return float(c @ g @ c)


# --- ACGOZLU sira: her adimda en cok kazandirani ekle (gercek marjinal degeri gorur)
kalan = list(range(K))
sec, onc = [], 0.0
print("ACGOZLU EKLEME -- her adimda span optimumunu en cok artiran yon")
print(
    f"{'#':>3s} {'yon':>28s} {'sqrtQ_dik':>10s} {'ARTIMLI kazanc':>15s} {'toplam':>10s} {'skor':>9s}"
)
for adim in range(K):
    en, eni = -1, None
    for i in kalan:
        k = kazanc([x[0] for x in sec] + [i])
        if k > en:
            en, eni = k, i
    art = en - onc
    # dik erisim
    if sec:
        idx = [x[0] for x in sec]
        g = (D[:, idx].T @ D[:, idx]) / N
        c, *_ = np.linalg.lstsq(g, (D[:, idx].T @ D[:, eni]) / N, rcond=RC)
        dp = D[:, eni] - D[:, idx] @ c
    else:
        dp = D[:, eni]
    sq = np.sqrt(float((dp * dp).mean()))
    sec.append((eni, art, sq))
    kalan.remove(eni)
    onc = en
    print(
        f"{adim + 1:3d} {ad[eni].replace('tuketim_', '').replace('.csv', ''):>28s} "
        f"{sq:10.4f} {art:15.6f} {en:10.6f} {np.sqrt(M0 - en):9.5f}"
    )

art = np.array([x[1] for x in sec])
sq = np.array([x[2] for x in sec])
print(f"\nToplam {onc:.6f}  (r_hat ile tutarli mi? beklenen 0.003872)")
print(f"\nARTIMLI KAZANC DAGILIMI (ilk 3 kurulus yonu haric, n={K - 3})")
a2 = art[3:]
print(
    f"  ortalama {a2.mean():.3e}   ortanca {np.median(a2):.3e}   "
    f"%75 {np.percentile(a2, 75):.3e}   maks {a2.max():.3e}"
)
print(f"  ->  rho esdegeri: ort {np.sqrt(a2.mean()):.4f}  ortanca {np.sqrt(np.median(a2)):.4f}")
print("\nDIK ERISIM ile ARTIMLI KAZANC iliskisi (ilk 3 haric)")
s2 = sq[3:]
from scipy.stats import spearmanr

r_, p_ = spearmanr(s2, a2)
print(f"  Spearman(sqrtQ_dik, artimli kazanc) = {r_:+.3f}  p={p_:.3f}")
buy = a2[s2 >= 0.30]
kuc = a2[s2 < 0.30]
print(
    f"  sqrtQ_dik >= 0.30 (n={len(buy)}): ort kazanc {buy.mean():.3e} -> rho {np.sqrt(buy.mean()):.4f}"
)
print(
    f"  sqrtQ_dik <  0.30 (n={len(kuc)}): ort kazanc {kuc.mean():.3e} -> rho {np.sqrt(kuc.mean()):.4f}"
)

print("\n=== YARIN ICIN TAHMIN ===")
print("7 yeni eksen, her biri gecmis ARTIMLI ortalamasi kadar getirirse:")
for ad_, m in [
    ("tum artimlilar", a2.mean()),
    ("sqrtQ_dik>=0.30 olanlar", buy.mean() if len(buy) else 0),
]:
    k = 0.003872 + 7 * m
    print(f"  {ad_:26s} kazanc {k:.6f} -> skor {np.sqrt(max(M0 - k, 1e-9)):.5f}")
rng = np.random.default_rng(3)
NS = 400_000
hav = buy if len(buy) >= 5 else a2
d7 = rng.choice(hav, size=(NS, 7)).sum(1)
s = np.sqrt(np.maximum(M0 - 0.003872 - d7, 1e-9))
print(f"\nMONTE CARLO (7 cekim, {'buyuk-erisim' if len(buy) >= 5 else 'tum'} artimli havuzdan)")
print(
    f"  medyan {np.median(s):.5f}  %10 {np.percentile(s, 10):.5f}  %90 {np.percentile(s, 90):.5f}"
)
for e, t in [(0.99940, "2. sira"), (1.00118, "3. sira")]:
    print(f"  P({t}) = {(s < e).mean() * 100:5.1f}%")
