# -*- coding: utf-8 -*-
"""k10'un CV kismi: kayitli rhat ile dik-yapma + cep ayrisimi (test kismi beklemeden)."""
import os, sys, json
import numpy as np
import pandas as pd
KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"

d = P.blok("yaz25", soguk_harman="cat", son_islem=True).reset_index(drop=True)
d["w"] = P.agirlik(d)
rhat = np.load(os.path.join(CIK, "k10_rhat_yaz25.npy"))
assert len(rhat) == len(d)
w = d["w"].values; r = d["r"].values; W = w.sum()
rms = float(np.sqrt(np.sum(w * r * r) / W))


def rho_olc(u):
    u = np.asarray(u, float)
    nrm = np.sqrt(np.sum(w * u * u) / W)
    if nrm <= 0:
        return None
    un = u / nrm
    rho = float(np.sum(w * r * un) / W)
    v = w * (r * un - rho)
    gg = pd.DataFrame({"g": d["tanim"].values, "v": v}).groupby("g", observed=True)["v"].sum().values
    se = float(np.sqrt(np.sum(gg * gg)) / W)
    return dict(rho=rho, oran=rho / rms, se=se, t=rho / se if se > 0 else 0.0)


def dik(u, Z):
    G = Z.T @ (w[:, None] * Z)
    b = Z.T @ (w * u)
    c = np.linalg.lstsq(G, b, rcond=None)[0]
    return u - Z @ c


n = len(d)
one = np.ones((n, 1))
ay = pd.get_dummies(d["ay"]).values.astype(float)
hg = pd.get_dummies(d["hg"]).values.astype(float)
uf = pd.get_dummies(np.minimum(d["ufuk_gun"].values.astype(int) // 15, 8)).values.astype(float)
rj = d["soguk_mu"].values.astype(float).reshape(-1, 1)
gun = pd.get_dummies(d["tarih"].astype(str)).values.astype(float)
S = {"ham": rho_olc(rhat)}
kat = [("sabit", one), ("sabit+ay", np.hstack([one, ay])),
       ("sabit+ay+hg", np.hstack([one, ay, hg])),
       ("sabit+ay+hg+ufuk", np.hstack([one, ay, hg, uf])),
       ("sabit+ay+hg+ufuk+rejim", np.hstack([one, ay, hg, uf, rj])),
       ("YAPAYLIK TAM (rejim x ay,ufuk)", np.hstack([one, ay, hg, uf, rj, rj * ay, rj * uf])),
       ("GUN(122) x rejim", np.hstack([gun, rj, rj * gun]))]
for ad, Z in kat:
    up = dik(rhat, Z)
    s = rho_olc(up)
    s["kalan_norm_payi"] = float(np.sum(w * up * up) / np.sum(w * rhat * rhat))
    S["dik_" + ad] = s
nrm = np.sqrt(np.sum(w * rhat * rhat) / W)
u = rhat / nrm
katki = w * r * u / W
tuk = d.tuketim.values.astype(float); sg = d.soguk_mu.values == 1
CEP = {"soguk&sifir": sg & (tuk == 0), "soguk&pozitif": sg & (tuk > 0),
       "sicak&sifir": (~sg) & (tuk == 0), "sicak&pozitif": (~sg) & (tuk > 0)}
S["cep_katkisi"] = {k: float(katki[v].sum()) for k, v in CEP.items()}
S["cep_katkisi"]["TOPLAM_rho"] = float(katki.sum())
for a in (4, 5, 6, 7):
    S["cep_katkisi"]["ay=%d" % a] = float(katki[d["ay"].values == a].sum())
json.dump(S, open(os.path.join(CIK, "k10b.json"), "w", encoding="utf-8"), indent=1)
print("%-36s %9s %8s %7s %9s" % ("A YONU (yaz25) dik-yapma sinavi", "oran", "SE", "t", "kalan||"))
for k, v in S.items():
    if isinstance(v, dict) and "oran" in v:
        print("%-36s %+9.4f %8.4f %7.1f %9s" % (
            k, v["oran"], v["se"], v["t"],
            ("%.3f" % v["kalan_norm_payi"]) if "kalan_norm_payi" in v else "-"))
print("\nrho'nun CEP AYRISIMI (toplami = rho):")
for k, x in S["cep_katkisi"].items():
    print("   %-16s %+.5f" % (k, x))
