# -*- coding: utf-8 -*-
"""YON 4 s2: aday soguk yonlerin TEST tarafinda 30-gonderim span'ina dik payi."""
import json
import os
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
SP = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
PK = os.path.join(KOK, "experiments/model29/p_kalici")

taban = np.load(os.path.join(SP, "taban_log.npy"))
ids = np.load(os.path.join(SP, "test_ids.npy"))
T = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
assert len(T) == len(taban)
# sira denetimi
assert np.array_equal(T.id.values.astype(str), ids.astype(str)), "SIRA UYUSMUYOR"

V = np.load(os.path.join(PK, "p34_V30.npy"))
B = np.load(os.path.join(PK, "p34_dik_baz.npy"))
M = np.c_[V, B.T]           # (n, 33)
Q, _ = np.linalg.qr(M)      # ortonormal span tabani
print("span boyutu:", Q.shape)

n = len(T)
sg = (T.soguk_mu.values == 1)
guc = T.guc.values.astype(np.float64)
lg = np.log1p(guc)
p = taban


def mrk(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(n)
    out[sg] = x[sg] - x[sg].mean()
    return out


A = {}
q = p - lg
A["T_kapasite_BUZME"] = -mrk(q)
A["T_soguk_SABIT"] = np.where(sg, 1.0, 0.0)
A["T_guc_egim"] = mrk(lg)
A["T_guc_egim_kare"] = mrk((lg - lg[sg].mean()) ** 2)
A["T_p_seviye_egim"] = mrk(p)
A["T_ilce_ort_BUZME"] = -mrk(p - T.g_ilce_log_ort.values.astype(np.float64))
A["T_kova_ort_BUZME"] = -mrk(p - T.g_kova_log_ort.values.astype(np.float64))
A["T_cdd18"] = mrk(T.cdd18.values.astype(np.float64))
A["T_ufuk"] = mrk(T.ufuk_gun.values.astype(np.float64))
A["T_yas"] = mrk(np.log1p(np.maximum(T.yas.values.astype(np.float64), 0)))
A["T_nufus_yog"] = mrk(np.log1p(T.ilce_nufus_yogunlugu.values.astype(np.float64)))
A["T_guc_yuzdelik"] = mrk(T.guc_yuzdelik.values.astype(np.float64))
# referans: sicak sabit ve kuresel sabit
A["ref_KURESEL_SABIT"] = np.ones(n)
A["ref_SICAK_SABIT"] = np.where(sg, 0.0, 1.0)

R = {}
print("%-24s | %8s | %8s | %8s" % ("YON", "||d||rms", "dik pay", "dik rms"))
print("-" * 60)
for ad, d in A.items():
    nd = float(np.sqrt(np.mean(d * d)))
    c = Q.T @ d
    dik = d - Q @ c
    ndik = float(np.sqrt(np.mean(dik * dik)))
    pay = ndik / nd if nd > 0 else 0.0
    R[ad] = dict(norm_rms=nd, dik_pay=pay, dik_rms=ndik)
    print("%-24s | %8.4f | %8.4f | %8.4f" % (ad, nd, pay, ndik))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_span.json"), "w") as f:
    json.dump({k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in R.items()}, f, indent=1)
print("\nsoguk satir sayisi TEST:", int(sg.sum()), "oran", round(float(sg.mean()), 4))
