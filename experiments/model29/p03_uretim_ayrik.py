"""p03 BELIRLEYICI DENEY: ayni ozellik kumesi ve ayni regresor ailesiyle
TEK ASAMALI mi IKI ASAMALI mi daha iyi?

p03_uretim_iki_asama.py, iki asamali kurulusu URETIM tahminine karsi olcuyor;
ama uretim tahmini 9 modelli bir cat/xgb/lgbm demeti, benim pozitif-altkume
regresyonum tek bir LightGBM. Bu karsilastirma YAPININ degil MODEL GUCUNUN
farkini olcer. Burada ikisini ESIT guce getiriyoruz:

  S1  tek asamali : LightGBM(huber), TUM satirlarda egitilir
  S2  iki asamali : (1-P0) x LightGBM(huber), YALNIZ pozitif satirlarda

Ikisi de guz25+kis26 bloklarinda, ayni 143 kimliksiz ozellikle, ayni
hiperparametre ve ayni 3 tohumla egitilir. Fark YALNIZCA yapidir.

SIZINTI: yaz25 hedefi hicbir yerde etiket degil; kimlik sutunlari elendi.
"""

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
BURA = os.path.dirname(os.path.abspath(__file__))
HEDEF_SOGUK = 0.222
KIMLIK = ["tanim_num", "tanim_on2", "tanim_on3", "tanim_on4", "tanim_on5"]
ATLA = ["tanim", "tarih", "tuketim", "lokasyon", "_blok"]
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
P = [np.load(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy")).astype(np.float64)
     for t in (1000, 1001, 1002) for aa in ("cat", "xgb", "lgbm")
     if os.path.exists(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy"))]
z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
idx = np.concatenate([sic.index.values, sog.index.values])
pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
bf = e.loc[idx].copy()
yv = np.log1p(bf.tuketim.to_numpy(dtype=np.float64))
sgm = bf.soguk_mu.to_numpy(dtype=np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()


def olc(p):
    r = np.asarray(p, dtype=np.float64) - yv
    s = sgm == 1
    return {"duz": float(np.sqrt(np.mean(r * r))),
            "test_agirlikli": float(np.sqrt(np.mean(ww * r * r))),
            "soguk": float(np.sqrt(np.mean(r[s] ** 2))),
            "sicak": float(np.sqrt(np.mean(r[~s] ** 2)))}


say = [c for c in e.columns
       if c not in ATLA and c not in KIMLIK and pd.api.types.is_numeric_dtype(e[c])]
egt = e[e._blok.isin(["guz25", "kis26"])]
ye = np.log1p(egt.tuketim.to_numpy(dtype=np.float64))
ze = (egt.tuketim.to_numpy() == 0).astype(int)
Xe = egt[say].astype(np.float32)
Xh = bf[say].astype(np.float32)
poz = ye > 0
del e, egt
log(f"egitim {Xe.shape}, yaz25 {Xh.shape}")

ORT = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           num_threads=8, verbose=-1)
HUB = dict(ORT, objective="huber", alpha=2.0, lambda_l2=20.0, metric="l2")
L2 = dict(ORT, objective="l2", metric="l2", lambda_l2=5.0)
PC = dict(ORT, objective="binary", metric="binary_logloss", lambda_l2=5.0)
TUR, TOHUM = 700, [7, 17, 27]


def cok(pk, X, y):
    return np.mean([lgb.train(dict(pk, seed=s), lgb.Dataset(X, y),
                              num_boost_round=TUR).predict(Xh) for s in TOHUM], axis=0)


R = {"uretim_tabani": olc(pb), "aciklama": __doc__.strip()}
P0 = cok(PC, Xe, ze)
log("P0 hazir")
for ad, pk in (("huber", HUB), ("l2", L2)):
    s1 = cok(pk, Xe, ye)
    s2p = cok(pk, Xe[poz], ye[poz])
    R[f"S1_tek_asamali_{ad}"] = olc(s1)
    R[f"S2_iki_asamali_{ad}"] = olc((1 - P0) * s2p)
    R[f"YAPI_KAZANCI_{ad}"] = {
        k: R[f"S1_tek_asamali_{ad}"][k] - R[f"S2_iki_asamali_{ad}"][k]
        for k in R["uretim_tabani"]}
    log(f"{ad} S1", json.dumps(R[f"S1_tek_asamali_{ad}"]))
    log(f"{ad} S2", json.dumps(R[f"S2_iki_asamali_{ad}"]))
    log(f"{ad} YAPI KAZANCI", json.dumps(R[f"YAPI_KAZANCI_{ad}"]))
    # uretim demetiyle harman: (1-P0)*(0.5*s2p + 0.5*pb/(1-P0)) yerine dogrudan
    R[f"H_uretim_ile_harman_{ad}"] = {
        f"w{w}": olc(w * ((1 - P0) * s2p) + (1 - w) * pb) for w in (0.25, 0.5)}
    log(f"{ad} harman", json.dumps(R[f"H_uretim_ile_harman_{ad}"]))

json.dump(R, open(os.path.join(BURA, "p03_uretim_ayrik.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
log("bitti")
