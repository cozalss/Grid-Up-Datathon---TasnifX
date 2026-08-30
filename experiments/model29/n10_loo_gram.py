"""n10 ADIM 1 -- Gram onbellegi kur (yalniz OKUR, hicbir gonderim yapmaz).

29 olculmus/turetilmis yonun G matrisi, L vektoru, Q kosegeni ve LB skorlari
bir .npz'ye yazilir. Bundan sonraki butun n10 analizleri Gram duzeyindedir;
714.688 satirlik CSV'ler bir daha okunmaz.

DIKKAT: L_OBS yalniz LB'de GERCEKTEN olculmus dosyalar icin "olculmus"tur.
EK_MODEL (y40) TURETILMIS bir L'dir; span tabani olarak kullanilir ama
LOO'da HEDEF olarak KULLANILMAZ. olculmus_mu bayragi bunu tasir.
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
CIKTI = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad/n10_gram.npz"
)
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0  # noqa: E402

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
            raise ValueError(f"{f}: id eslesmedi")
        d = d.iloc[pos].reset_index(drop=True)
    x = np.log1p(d[k].values.astype(np.float64))
    if not np.isfinite(x).all():
        raise ValueError(f"{f}: sonlu olmayan deger var -- SESSIZCE DUZELTILMEZ")
    return x


a0 = oku(TABAN)
N = len(a0)
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)

AD, D, L, P, OLC = [], [], [], [], []
for f, Pj in SK.items():
    if f == TABAN:
        continue
    if not os.path.exists(os.path.join(S, f)):
        print(f"ATLANDI (dosya yok): {f}")
        continue
    v = oku(f)
    if v is None or len(v) != N:
        print(f"ATLANDI (test id kumesiyle uyusmuyor): {f}")
        continue
    d = v - a0
    AD.append(f)
    D.append(d)
    L.append((M0 + float((d * d).mean()) - Pj * Pj) / 2)
    P.append(Pj)
    OLC.append(True)
for o in DUR.get("olcumler", []):
    d = oku(o["dosya"]) - a0
    AD.append(o["dosya"])
    D.append(d)
    L.append((M0 + float((d * d).mean()) - o["skor"] ** 2) / 2)
    P.append(o["skor"])
    OLC.append(True)
for f, Lj in EK_MODEL.items():
    d = oku(f) - a0
    AD.append(f)
    D.append(d)
    L.append(float(Lj))
    P.append(np.nan)  # LB'de olculmedi -- TURETILMIS
    OLC.append(False)

D = np.array(D)
K = len(AD)
G = (D @ D.T) / N
del D
np.savez(
    CIKTI,
    ad=np.array(AD),
    G=G,
    L=np.array(L),
    P=np.array(P),
    olculmus=np.array(OLC),
    N=N,
    M0=M0,
)
print(f"N={N}  K={K}  olculmus={sum(OLC)}  turetilmis={K - sum(OLC)}")
print(f"YAZILDI {CIKTI}")
w = np.linalg.eigvalsh(G)
print(
    f"G ozdegerleri: en buyuk {w[-1]:.4e}  en kucuk {w[0]:.4e}  kosul {w[-1] / max(w[0], 1e-300):.2e}"
)
