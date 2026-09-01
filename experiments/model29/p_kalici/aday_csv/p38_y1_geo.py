"""y1_geo: aday deltalarin TEST tarafi geometrisi (span dikligi, birbirine benzerlik)."""
import json, os
import numpy as np
SCR = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
PK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX/experiments/model29/p_kalici"
V = np.load(os.path.join(PK, "p34_V30.npy"))
BAZ = np.load(os.path.join(PK, "p34_dik_baz.npy"))
N = V.shape[0]
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)

def dik(d):
    y = d - V @ (Gi @ ((V.T @ d) / N))
    for b in BAZ:
        y = y - float((y * b).mean()) * b
    return y

ADLAR = {"RIDGE": os.path.join(SCR, "yon_ARTIK_RIDGE.npy"),
         "LGBM_merk_takvimsiz": os.path.join(SCR, "artik_modeli_g_test.npy")}
D, Dd = {}, {}
for k, v in ADLAR.items():
    if not os.path.exists(v):
        print("YOK", v); continue
    d = np.load(v).astype(np.float64)
    D[k] = d; Dd[k] = dik(d)
    nn = np.sqrt((d*d).mean()); nd = np.sqrt((Dd[k]*Dd[k]).mean())
    print(f"{k:22s} ||d||={nn:.5f} ||dik||={nd:.5f} dik_pay={nd/nn:.4f} "
          f"span_ici_var={1-(nd/nn)**2:.4f}")
ks = list(D)
for i in range(len(ks)):
    for j in range(i+1, len(ks)):
        a, b = D[ks[i]], D[ks[j]]
        ca = float((a*b).mean()/np.sqrt((a*a).mean()*(b*b).mean()))
        ad_, bd = Dd[ks[i]], Dd[ks[j]]
        cd = float((ad_*bd).mean()/np.sqrt((ad_*ad_).mean()*(bd*bd).mean()))
        print(f"kosinus ham({ks[i]},{ks[j]})={ca:+.4f}  dik={cd:+.4f}")
