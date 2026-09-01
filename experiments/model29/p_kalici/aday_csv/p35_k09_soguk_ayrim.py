# -*- coding: utf-8 -*-
"""SOGUK CEP: soguk-sifir (rho -0.54) ile soguk-pozitif (rho +0.14) ZIT isaretli ve
   ikisi birlikte MSE'nin %54'u. Sorunun tamami UYELIGIN kestirilebilirligi.
   Blok-disi siniflandirici SADECE soguk satirlarda egitilir/olculur."""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
ATIL = {"tuketim", "id", "_blok", "tanim", "tarih", "lokasyon", "p", "y", "r", "w",
        "sog_cat", "sog_xgb", "sog_lgbm", "ay", "hg", "tanim_num", "soguk_mu"}
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
           verbose=-1, num_threads=4, seed=5)


def rho_olc(d, u, alt=None):
    """alt: alt-kume maskesi; yon o kume disinda 0. rho TUM blok uzerinden."""
    w = d["w"].values; r = d["r"].values; W = w.sum()
    u = np.asarray(u, dtype=np.float64).copy()
    if alt is not None:
        u = np.where(alt, u, 0.0)
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
D = {}
for b in ("yaz25", "guz25", "kis26"):
    d = P.blok(b, soguk_harman="cat", son_islem=True).reset_index(drop=True)
    d["w"] = P.agirlik(d)
    D[b] = d
    print("kuruldu", b, len(d), "t=%.0f" % (time.time() - t0)); sys.stdout.flush()
kol = [c for c in D["yaz25"].columns if c not in ATIL and
       (pd.api.types.is_numeric_dtype(D["yaz25"][c]) or pd.api.types.is_bool_dtype(D["yaz25"][c]))]
print("oznitelik", len(kol))

OUT = {}
for hedef in ("yaz25", "guz25", "kis26"):
    dh = D[hedef]
    sg_h = dh.soguk_mu.values == 1
    tr = pd.concat([D[b][D[b].soguk_mu.values == 1] for b in D if b != hedef], ignore_index=True)
    Xtr = tr[kol].astype("float32")
    ytr = (tr.tuketim.values == 0).astype(np.int8)
    m = lgb.train(PAR, lgb.Dataset(Xtr, label=ytr, weight=tr["w"].values), num_boost_round=400)
    Xte = dh.loc[sg_h, kol].astype("float32")
    q = np.zeros(len(dh)); q[sg_h] = m.predict(Xte)
    yte = (dh.tuketim.values == 0).astype(np.int8)
    auc = float(roc_auc_score(yte[sg_h], q[sg_h], sample_weight=dh["w"].values[sg_h]))
    S = {"soguk_auc_sifir": auc, "soguk_sifir_orani": float(yte[sg_h].mean()),
         "soguk_n": int(sg_h.sum())}
    S["yumusak_q_soguk"] = rho_olc(dh, q, sg_h)
    qm = q - (q[sg_h].mean() if sg_h.any() else 0)
    S["yumusak_q_merkezli_soguk"] = rho_olc(dh, qm, sg_h)
    for e in (0.3, 0.5, 0.7, 0.9):
        S["bayrak_e%.1f" % e] = rho_olc(dh, (q > e).astype(float), sg_h)
    S["KAHIN soguk-sifir bayrak"] = rho_olc(dh, yte.astype(float), sg_h)
    S["KAHIN soguk-pozitif bayrak"] = rho_olc(dh, (1 - yte).astype(float), sg_h)
    S["KAHIN soguk +-1 (sifir/pozitif ayrimi)"] = rho_olc(dh, (1 - 2.0 * yte), sg_h)
    S["KAHIN soguk tam artik"] = rho_olc(dh, dh["r"].values, sg_h)
    OUT[hedef] = S
    print("[%s] soguk AUC=%.4f  yumusak oran=%+.4f  kahin(+-1)=%+.4f  t=%.0f" % (
        hedef, auc, S["yumusak_q_soguk"]["oran"], S["KAHIN soguk +-1 (sifir/pozitif ayrimi)"]["oran"],
        time.time() - t0)); sys.stdout.flush()
with open(os.path.join(CIK, "k09_soguk.json"), "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=1)
print("\n%-40s %9s %9s %9s" % ("SOGUK CEP", "yaz25", "guz25", "kis26"))
for k in OUT["yaz25"]:
    v = [OUT[b][k] for b in ("yaz25", "guz25", "kis26")]
    if isinstance(v[0], dict):
        print("%-40s %+9.4f %+9.4f %+9.4f" % (k, v[0]["oran"], v[1]["oran"], v[2]["oran"]))
    else:
        print("%-40s %9.4f %9.4f %9.4f" % (k, v[0], v[1], v[2]))
