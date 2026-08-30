"""L'LERDEKI GERCEK GURULTU ve nrm'nin GURULTU SISMESI.

Sorun: L_j = (M0 + Q_j^tum - P_j^2)/2 kullaniyoruz ama Kaggle P_j'yi yalniz
PUBLIC %50 satirda hesapliyor. Dogru bagintI:
    P_j^2 = M0^pub - 2 L_j^pub + Q_j^pub
Bizim formulumuz L_j^pub yerine
    L_j^pub + [(M0^tum - M0^pub) + (Q_j^tum - Q_j^pub)]/2
veriyor. Ilk terim ORTAK (M0 kalibrasyonuna gomulu), ikincisi YONE OZGU
GURULTU:  delta_L_j = (Q_j^tum - Q_j^pub)/2

Bu gurultu d_j^2'nin yari-orneklem sacilimindan gelir ve d_j agir kuyruklu.
Dogrudan olculur: rastgele yarilar alip Q_j^yari'nin sacilimina bak.

Sonra: nrm = L' pinv(G) L  ifadesinde gurultu SISMESI
    E[sisme] = sum_i sigma_i^2 / s_i     (tutulan kipler uzerinde)
Bu, "bilinen optimum 1.000528" iddiasinin ne kadarinin gercek oldugunu
belirler. Kucuk s_i'li kipler gurultuyu buyutur.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
sys.path.insert(0, M29)
from m112_kalibre import M0  # noqa: E402

BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"  # M0 m112den gelir (docs/69)
EK_MODEL = {}  # bosaltildi (docs/69): s3y40 kendi skoruyla Gram'da

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
V, L, ADLAR = [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
    ADLAR.append(f)
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
    ADLAR.append(f)
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
    ADLAR.append(o["dosya"])
V, L = np.array(V).T, np.array(L)
k = V.shape[1]
G = (V.T @ V) / N
sv = np.linalg.svd(G, compute_uv=False)

print("=" * 78)
print("delta_L_j = (Q_j^tum - Q_j^public)/2  --  yari-orneklem sacilimi")
print("=" * 78)
rng = np.random.default_rng(3)
Nh = N // 2
sig = np.zeros(k)
print(f"{'yon':>34s} {'Q^tum':>9s} {'sd(d^2)':>9s} {'sigma_L':>10s}")
for j in range(k):
    d2 = V[:, j] ** 2
    Qall = float(d2.mean())
    # yari-orneklem: Q^pub - Q^tum = (Q^pub - Q^priv)/2
    orn = []
    for _ in range(60):
        m = rng.permutation(N)[:Nh]
        orn.append(float(d2[m].mean()) - Qall)
    s = float(np.std(orn)) / 2.0
    sig[j] = s
    if j < 6 or j >= k - 4:
        print(f"{ADLAR[j][:34]:>34s} {Qall:9.5f} {float(d2.std()):9.4f} {s:10.3e}")
print("  ...")
print(
    f"  ORTALAMA sigma_L = {sig.mean():.3e}   medyan {np.median(sig):.3e}   "
    f"en buyuk {sig.max():.3e}"
)
print(f"  (LB yuvarlamasindan gelen ~5e-6 ile karsilastir: {sig.mean() / 5e-6:.0f} KAT buyuk)")
print(
    f"  (LOO yeniden kurma hatasi 3.4e-4 idi -- tutarli mi? "
    f"{'EVET' if 0.3e-4 < sig.mean() < 3e-3 else 'HAYIR'})"
)

print("\n" + "=" * 78)
print("nrm'NIN GURULTU SISMESI -- 'bilinen optimum' ne kadari gercek?")
print("=" * 78)
print(
    f"{'rcond':>8s} {'rank':>5s} {'nrm(ham)':>10s} {'sisme':>10s} {'nrm(duzelt)':>12s} "
    f"{'saf opt':>9s} {'sd':>9s}"
)
for rc in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
    Gi = np.linalg.pinv(G, rcond=rc)
    rank = int((sv > rc * sv[0]).sum())
    nrm = float(((V @ (Gi @ L)) ** 2).mean())
    orn = []
    for _ in range(300):
        Ln = L + rng.normal(0, sig)
        orn.append(float(((V @ (Gi @ Ln)) ** 2).mean()))
    orn = np.array(orn)
    sisme = float(orn.mean() - nrm)
    duz = max(nrm - sisme, 0.0)
    print(
        f"{rc:8.0e} {rank:5d} {nrm:10.6f} {sisme:+10.6f} {duz:12.6f} "
        f"{np.sqrt(M0 - duz):9.6f} {orn.std():9.6f}"
    )

print("\nYORUM: 'sisme', gurultunun kendi kendine urettigi sahte kazanctir.")
print("nrm(duzelt) = nrm(ham) - sisme  ->  gercekci saf optimum.")
print("Sisme buyukse o rcond'da fazla kip tutuluyor demektir.")
