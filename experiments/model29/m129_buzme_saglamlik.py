"""DOKUZUNCU HATA -- buzmeli_r_hat gurultu altinda patliyor. Duzeltmeyi SEC.

TEShIS. Kip tablosunda w = 1.86e-12, 2.5e-17, 1.25e-18, -4.2e-17 var.
Koddaki koruma "w <= 1e-12: continue"; 1.856e-12 o esigin HEMEN USTUNDE.
Gercek veride c/sigma = 0.09 oldugu icin a=0 ve zarar yok. Ama c gurultuyle
sigma'yi gecerse (sansin %50'si) a>0 olur ve a*c/w patlar:
    nrm  gercek 0.003772  ->  bozulmus L ile medyan 0.0126, ort 678, maks 8144
    40 cekilisin 13'u 0.05'i asiyor.

UC ADAY DUZELTME olculur:
  A  mevcut                      w > 1e-12
  B  goreli taban                w/w_max > 1e-8
  C  goreli taban + anlamlilik    w/w_max > 1e-8  VE  c^2 > 4*sigma^2  (2 sigma)

Olcut: (1) gurultu altinda nrm kararliligi, (2) gercek veride nrm'nin ne
kadar degistigi. Kararli olan ve gercek degeri az bozan kazanir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
TABAN = "tuketim_m6_ikiyon.csv"
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, L_gurultusu  # noqa: E402

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
sig = L_gurultusu(V, N)
w, U = np.linalg.eigh(G)
sira = np.argsort(-w)
w, U = w[sira], U[:, sira]
sig_i = np.sqrt(np.einsum("ij,jk,ki->i", U.T, np.diag(sig**2), U))
WMAX = float(w[0])


KIPLER = {
    "A": (None, 0.0),  # mevcut: mutlak 1e-12 tabani, anlamlilik kapisi yok
    "B": (1e-8, 0.0),  # goreli w tabani
    "C": (1e-8, 4.0),  # + 2 sigma
    "D": (1e-8, 9.0),  # + 3 sigma
    "E": (1e-6, 4.0),  # daha sert w tabani + 2 sigma
    "F": (1e-6, 9.0),  # daha sert w tabani + 3 sigma
}


def coz(Lv, kip):
    wtab, ksig = KIPLER[kip]
    c = U.T @ Lv
    a = np.zeros(len(w))
    kaz = 0.0
    for i in range(len(w)):
        if wtab is None:
            if w[i] <= 1e-12:
                continue
        elif w[i] / WMAX <= wtab:
            continue
        if c[i] ** 2 <= 0:
            continue
        if ksig > 0 and c[i] ** 2 <= ksig * sig_i[i] ** 2:
            continue
        lam2 = max(c[i] ** 2 - sig_i[i] ** 2, 0.0)
        a[i] = lam2 / c[i] ** 2
        kaz += lam2**2 / (c[i] ** 2 * w[i])
    k = U @ (a * c / np.where(w > 1e-300, w, 1.0))
    return V @ k, float(kaz), float(k @ Lv), int((a > 0.05).sum())


print(
    f"{'kip':>4s} {'gercek nrm':>12s} {'gercek kL':>11s} {'tutulan kip':>12s} "
    f"{'bozuk: medyan':>14s} {'ort':>12s} {'maks':>12s} {'patlak':>7s}"
)
rng = np.random.default_rng(7)
BOZUK = [L + rng.normal(0, sig) for _ in range(60)]
sonuc = {}
for kip in KIPLER:
    rh, kaz, kL, nk = coz(L, kip)
    nrm = float((rh * rh).mean())
    ns = []
    for Ln in BOZUK:
        r2, _, _, _ = coz(Ln, kip)
        ns.append(float((r2 * r2).mean()))
    ns = np.array(ns)
    patlak = int((ns > 0.05).sum())
    sonuc[kip] = dict(
        nrm=nrm,
        kL=kL,
        kaz=kaz,
        nk=nk,
        medyan=float(np.median(ns)),
        ort=float(ns.mean()),
        maks=float(ns.max()),
        patlak=patlak,
    )
    print(
        f"{kip:>4s} {nrm:12.6f} {kL:11.6f} {nk:12d} {np.median(ns):14.6f} "
        f"{ns.mean():12.4g} {ns.max():12.4g} {patlak:5d}/60"
    )

print("\nYORUM:")
print("  A = mevcut kod. B = goreli w tabani. C = B + 2-sigma anlamlilik kapisi.")
print("  Gercek nrm'yi az bozup gurultu altinda KARARLI kalan kazanir.")
en = min(KIPLER, key=lambda k: (sonuc[k]["patlak"], sonuc[k]["maks"]))
print(f"  -> en kararli: {en}")
for k in KIPLER:
    d = sonuc[k]
    print(
        f"     {k}: gercek nrm {d['nrm']:.6f} (A'ya gore "
        f"{d['nrm'] / sonuc['A']['nrm'] - 1:+.1%}), tutulan kip {d['nk']}, "
        f"patlak {d['patlak']}/60"
    )
