"""p33-a: SPAN cebrini BAGIMSIZ yeniden kur.

m148'in kodunu CAGIRMAZ; yalnizca m112_kalibre'den M0 ve buzmeli_r_hat alir
(gorev bunu acikca soyluyor). Cikan r_hat / kL / TABAN_MSE, m148'in DOKUM
ciktisiyla karsilastirilir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
S = os.path.join(KOK, "submissions")
GEC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

TABAN = "tuketim_m6_ikiyon.csv"
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
with open(os.path.join(M29, "olculmus_skorlar.json"), encoding="utf-8") as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json"), encoding="utf-8") as fh:
    DUR = json.load(fh)

V, L, AD, TUR = [], [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
    AD.append(f)
    TUR.append("LB_olculmus")
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
    AD.append(f)
    TUR.append("TURETILMIS_L")
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
    AD.append(o["dosya"])
    TUR.append("LB_olculmus")

V, L = np.array(V).T, np.array(L)
G = (V.T @ V) / N
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
nrm = float((r_hat * r_hat).mean())
TABAN_MSE = float(M0 - 2 * kL + nrm)

print(f"N = {N}, yon sayisi = {V.shape[1]}")
print(f"  LB'de olculmus : {TUR.count('LB_olculmus')}")
print(f"  turetilmis L   : {TUR.count('TURETILMIS_L')}")
print(f"  taban (M0)     : {TABAN}  M0={M0}")
print(f"kL          = {kL:.9f}")
print(f"||r_hat||^2 = {nrm:.9f}")
print(f"beklenen kazanc (gercek) = {gercek:.9f}")
print(f"saf optimum sqrt(M0-gercek) = {np.sqrt(M0 - gercek):.7f}")
print(f"TABAN_MSE = M0-2kL+||r_hat||^2 = {TABAN_MSE:.9f}")
print(f"SAF SPAN SKORU = sqrt(TABAN_MSE) = {np.sqrt(TABAN_MSE):.7f}")

CIK = {
    "N": N,
    "M0": M0,
    "yon_sayisi": int(V.shape[1]),
    "lb_olculmus": TUR.count("LB_olculmus"),
    "turetilmis": TUR.count("TURETILMIS_L"),
    "adlar": AD,
    "kL": float(kL),
    "r_hat_norm2": nrm,
    "beklenen_kazanc": float(gercek),
    "saf_optimum_M0_gercek": float(np.sqrt(M0 - gercek)),
    "TABAN_MSE": TABAN_MSE,
    "saf_span_skoru": float(np.sqrt(TABAN_MSE)),
}

# --- m148 DOKUM ile karsilastir --------------------------------------------
sb = os.path.join(GEC, "sabitler.json")
if os.path.exists(sb):
    with open(sb, encoding="utf-8") as fh:
        SB = json.load(fh)
    rh = np.load(os.path.join(GEC, "r_hat.npy"))
    a0d = np.load(os.path.join(GEC, "a0.npy"))
    d = {
        "kL_fark": float(kL - SB["kL"]),
        "TABAN_MSE_fark": float(TABAN_MSE - SB["TABAN_MSE"]),
        "N_ayni": int(N) == int(SB["N"]),
        "r_hat_maxabs_fark": float(np.abs(r_hat - rh).max()),
        "a0_maxabs_fark": float(np.abs(a0 - a0d).max()),
    }
    print("\n--- m148 DOKUM ile karsilastirma ---")
    for k, v in d.items():
        print(f"  {k}: {v}")
    d["HUKUM"] = (
        "OZDES"
        if abs(d["kL_fark"]) < 1e-12 and d["r_hat_maxabs_fark"] < 1e-12
        else "AYRISMA VAR"
    )
    print(f"  HUKUM: {d['HUKUM']}")
    CIK["m148_karsilastirma"] = d
else:
    print("\n(DOKUM yok, karsilastirma atlandi)")

np.save(os.path.join(GEC, "p33_r_hat.npy"), r_hat)
np.save(os.path.join(GEC, "p33_a0.npy"), a0)
np.save(os.path.join(GEC, "p33_V.npy"), V)
with open(os.path.join(GEC, "p33_a_cebir.json"), "w", encoding="utf-8") as fh:
    json.dump(CIK, fh, indent=1)
print("\n-> p33_a_cebir.json")
