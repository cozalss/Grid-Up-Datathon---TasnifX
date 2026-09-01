# -*- coding: utf-8 -*-
"""SIZINTILI UST SINIR (yalnizca TESHIS): yaz25 icinde trafo-gruplu capraz kestirim.
   Mevsim BILINIYOR varsayimi -> gercekte ulasilamaz; ama 'artik gozlenebilirlerden
   ne kadar kestirilebilir' sorusunun MUTLAK tavani. Blok-disi degerle farki
   = mevsimsel TASINAMAZLIK payi."""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
ATIL = {"tuketim", "id", "_blok", "tanim", "tarih", "lokasyon", "p", "y", "r", "w",
        "sog_cat", "sog_xgb", "sog_lgbm", "ay", "hg", "tanim_num"}
PAR = dict(objective="regression", learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7,
           bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4, seed=11)


def rho_olc(d, u_ham):
    w = d["w"].values; r = d["r"].values; W = w.sum()
    u = np.asarray(u_ham, dtype=np.float64)
    nrm = np.sqrt(np.sum(w * u * u) / W)
    if nrm <= 0:
        return None
    u = u / nrm
    rho = float(np.sum(w * r * u) / W)
    v = w * (r * u - rho)
    gg = pd.DataFrame({"g": d["tanim"].values, "v": v}).groupby("g", observed=True)["v"].sum().values
    se = float(np.sqrt(np.sum(gg * gg)) / W)
    rms = float(np.sqrt(np.sum(w * r * r) / W))
    return dict(rho=rho, se=se, oran=rho / rms, t=rho / se if se > 0 else 0.0)


t0 = time.time()
d = P.blok("yaz25", soguk_harman="cat", son_islem=True).reset_index(drop=True)
d["w"] = P.agirlik(d)
kol = [c for c in d.columns if c not in ATIL and
       (pd.api.types.is_numeric_dtype(d[c]) or pd.api.types.is_bool_dtype(d[c]))]
X = d[kol].astype("float32")
g = d["tanim"].values
oof = np.zeros(len(d))
oofz = np.zeros(len(d))
z = (d["tuketim"].values == 0).astype(np.int8)
gkf = GroupKFold(n_splits=4)
for k, (tr, te) in enumerate(gkf.split(X, groups=g)):
    ds = lgb.Dataset(X.iloc[tr], label=d["r"].values[tr], weight=d["w"].values[tr])
    m = lgb.train(PAR, ds, num_boost_round=500)
    oof[te] = m.predict(X.iloc[te])
    ds2 = lgb.Dataset(X.iloc[tr], label=z[tr], weight=d["w"].values[tr])
    m2 = lgb.train(dict(PAR, objective="binary"), ds2, num_boost_round=400)
    oofz[te] = m2.predict(X.iloc[te])
    print("fold %d bitti t=%.0fs" % (k, time.time() - t0)); sys.stdout.flush()

S = {}
S["SIZINTILI artik regresyonu"] = rho_olc(d, oof)
S["SIZINTILI sifir olasiligi (yumusak)"] = rho_olc(d, oofz)
for e in (0.3, 0.5, 0.7, 0.9):
    S["SIZINTILI sifir bayrak e=%.1f" % e] = rho_olc(d, (oofz > e).astype(float))
from sklearn.metrics import roc_auc_score
S["auc_sifir"] = float(roc_auc_score(z, oofz, sample_weight=d["w"].values))
S["KAHIN sifir bayrak"] = rho_olc(d, z.astype(float))
S["KAHIN tam artik"] = rho_olc(d, d["r"].values)
with open(os.path.join(CIK, "k07_sizintili.json"), "w", encoding="utf-8") as f:
    json.dump(S, f, indent=1)
for k, v in S.items():
    if isinstance(v, dict):
        print("%-38s oran=%+.4f  SE=%.4f  t=%.1f" % (k, v["oran"], v["se"], v["t"]))
    else:
        print("%-38s %.4f" % (k, v))
