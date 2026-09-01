# -*- coding: utf-8 -*-
"""SIFIR CEBININ ANATOMISI: kahin rho'su nerede? YAKALANAN sifirlarda mi,
   KACIRILAN sifirlarda mi? (uyelik vekili: gozlenebilir t_sifir_orani>0.9)"""
import os, sys, json
import numpy as np
import pandas as pd
KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
out = {}
for b in ("yaz25", "guz25", "kis26"):
    d = P.blok(b, soguk_harman="cat", son_islem=True).reset_index(drop=True)
    w = P.agirlik(d); r = d["r"].values; W = w.sum(); TOP = float(np.sum(w * r * r))
    tuk = d.tuketim.values.astype(float)
    sf = d.t_sifir_orani.fillna(0).values.astype(float)
    bay = sf > 0.9
    z = tuk == 0
    S = {}
    for ad, m in [("sifir & YAKALANAN(bayrakli)", z & bay),
                  ("sifir & KACIRILAN", z & ~bay),
                  ("YANLIS POZITIF (bayrakli ama pozitif)", ~z & bay),
                  ("bayrakli TUM", bay)]:
        A = float(w[m].sum())
        S[ad] = dict(pay=float(m.mean()), mse_payi=float(np.sum(w[m] * r[m] ** 2) / TOP),
                     kahin=float(np.sqrt(np.sum(w[m] * r[m] ** 2) / TOP)),
                     bayrak_rho=float(np.sum(w[m] * r[m]) / np.sqrt(A * W) / np.sqrt(TOP / W)) if A > 0 else 0.0,
                     ort_p=float(np.average(d["p"].values[m], weights=w[m])) if A > 0 else 0.0,
                     ort_y=float(np.average(d["y"].values[m], weights=w[m])) if A > 0 else 0.0)
    out[b] = S
    print("### " + b)
    for k, v in S.items():
        print("   %-38s sat=%5.2f%% MSE=%6.2f%% kahin=%.4f bayrak_rho=%+.4f ort_p=%.2f ort_y=%.2f" % (
            k, 100 * v["pay"], 100 * v["mse_payi"], v["kahin"], v["bayrak_rho"], v["ort_p"], v["ort_y"]))
json.dump(out, open(os.path.join(CIK, "k12_sifir.json"), "w", encoding="utf-8"), indent=1)
