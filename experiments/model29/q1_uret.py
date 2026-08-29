"""Q1 -- HIZLI NISANLI YONLER (22 dakikalik pencere).

Iki bagimsiz aday, ayni onbellekli ozellik matrisi uzerinde:
  A) KAPASITE OFSETLI HEDEF: y = log1p(tuketim) - log_guc, L2 kaybi.
     Guc bandindaki U seklini (<=50 ve >1000 kVA'da fazla tahmin) dogrudan
     vurur: model artik "mutlak seviye" degil "kapasiteye gore yuklenme"
     ogrenir, guc bandi kirilimi hedefin icine gomulur.
  B) KUANTIL 0,38: dusuk seviye desilinde (+0,040 fazla tahmin) sisirmeyi
     vurur. Tek kuantil -- z2_kantil'in 7-kuantil integralinden FARKLI bir
     yon (o kosullu ORTALAMA kurar, bu dogrudan alt-kuantile kayar).
Amac KALITE degil FARKLI/DIK YON.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import z1_ortak as Z  # noqa: E402
import z2_tezgah as T  # noqa: E402
from m33_durust import VARSAYILAN  # noqa: E402

TUR = 300
TOHUM = 11


def main():
    t0 = time.time()
    Xtr, ytr, Xte, tr, te = T.matrisler()
    tr["L"] = tr.ly
    msk = Z.maskeler(tr, te)
    A6 = Z.taban()
    Ltr = np.log1p(ytr)
    otr = Xtr.log_guc.values.astype(float)
    ote = Xte.log_guc.values.astype(float)
    print(f"ofset log_guc: tr ort {otr.mean():.4f} te ort {ote.mean():.4f}", flush=True)
    rap = {}

    # --- A) kapasite ofsetli hedef
    p = dict(VARSAYILAN)
    p.update(
        objective="l2",
        metric="l2",
        learning_rate=0.06,
        lambda_l2=10.0,
        seed=TOHUM,
        bagging_seed=TOHUM + 1,
        feature_fraction_seed=TOHUM + 2,
    )
    dsA = lgb.Dataset(Xtr, Ltr - otr)
    mA = lgb.train(p, dsA, TUR)
    LA = np.clip(mA.predict(Xte) + ote, 0.0, 14.0)
    print(f"  A kapasite-ofset bitti, ort {LA.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(BURA, "q1_A.npy"), LA)
    rap["q1a_kapasite"] = Z.bitir(LA, te, msk, A6, "tuketim_q1a_kapasite.csv", kirp=2.0)
    del dsA, mA

    # --- B) kuantil 0,38
    p = dict(VARSAYILAN)
    p.update(
        objective="quantile",
        alpha=0.38,
        metric="quantile",
        learning_rate=0.06,
        lambda_l2=20.0,
        seed=TOHUM,
        bagging_seed=TOHUM + 1,
        feature_fraction_seed=TOHUM + 2,
    )
    dsB = lgb.Dataset(Xtr, Ltr)
    mB = lgb.train(p, dsB, TUR)
    LB_ = np.clip(mB.predict(Xte), 0.0, 14.0)
    print(f"  B kuantil0.38 bitti, ort {LB_.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(BURA, "q1_B.npy"), LB_)
    rap["q1b_kuantil38"] = Z.bitir(LB_, te, msk, A6, "tuketim_q1b_kuantil38.csv", kirp=2.0)

    json.dump(rap, open(os.path.join(BURA, "q1_uret.json"), "w"), indent=1)
    print(f"TAMAM ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
