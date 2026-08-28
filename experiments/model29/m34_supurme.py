"""Sizintisiz tezgahta konfig supurmesi. Iki dogrulama kesimi (tek kesime guvenme)."""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from m30_ozellik import yukle_ham
from m33_durust import VARSAYILAN, hizala, parca

tr, te = yukle_ham()
AY = [
    "2025-03-31",
    "2025-04-30",
    "2025-05-31",
    "2025-06-30",
    "2025-07-31",
    "2025-08-31",
    "2025-09-30",
    "2025-10-31",
    "2025-11-30",
    "2025-12-31",
]
OB = {}


def al(kesim, tavan):
    a = (kesim, tavan)
    if a not in OB:
        OB[a] = parca(tr, kesim, tavan=tavan)
    return OB[a]


def kos(dog, ofset=False, drop=(), tur=None, **pk):
    Xva, yva = al(dog, None)
    sog = Xva.soguk.values.astype(bool)
    Xs, ys = [], []
    for k in [x for x in AY if x < dog]:
        r = al(k, dog)
        if r is None:
            continue
        Xs.append(r[0])
        ys.append(r[1])
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    otr = np.log1p(Xtr.guc.values)
    ova = np.log1p(Xva.guc.values)
    if ofset:
        ytr = ytr - otr
    Xtr = Xtr.drop(columns=list(drop))
    Xva2 = Xva.drop(columns=list(drop))
    Xtr, Xva2 = hizala(Xtr.copy(), Xva2.copy())
    p = dict(VARSAYILAN)
    p.update(pk)
    ds = lgb.Dataset(Xtr, ytr)
    if tur is None:
        yv = (yva - ova) if ofset else yva
        m = lgb.train(
            p,
            ds,
            4000,
            valid_sets=[lgb.Dataset(Xva2, yv, reference=ds)],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        n = m.best_iteration
    else:
        m = lgb.train(p, ds, tur)
        n = tur
    pv = m.predict(Xva2, num_iteration=n)
    if ofset:
        pv = pv + ova
    L = (pv - yva) ** 2
    return dict(
        tur=int(n),
        rmsle=float(np.sqrt(L.mean())),
        soguk=float(np.sqrt(L[sog].mean())),
        sicak=float(np.sqrt(L[~sog].mean())),
        karisik=float(np.sqrt(0.7784 * L[~sog].mean() + 0.2216 * L[sog].mean())),
    )


if __name__ == "__main__":
    DENEY = (
        []
        if __name__ != "__main__"
        else [
            ("TABAN (tum aylik kesimler)", dict()),
            ("kapasite ofsetli hedef", dict(ofset=True)),
            ("yaprak 127", dict(num_leaves=127)),
            ("yaprak 31 lr.03", dict(num_leaves=31, learning_rate=0.03)),
            ("min_data 500", dict(min_data_in_leaf=500)),
            ("lambda_l2 20", dict(lambda_l2=20.0)),
            ("ff .6 bf .7", dict(feature_fraction=0.6, bagging_fraction=0.7)),
            ("huber", dict(objective="huber", alpha=2.0)),
        ]
    )
    sonuc = {}
    for dog in ["2025-11-30", "2025-09-30"]:
        print(f"\n########## DOGRULAMA {dog} ##########", flush=True)
        t0 = time.time()
        for ad, kw in DENEY:
            try:
                r = kos(dog, **kw)
                print(
                    f"  {ad:30s} tur {r['tur']:4d} RMSLE {r['rmsle']:.4f} soguk {r['soguk']:.4f} "
                    f"sicak {r['sicak']:.4f} | test-karisimi {r['karisik']:.4f}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                sonuc.setdefault(ad, {})[dog] = r
            except Exception as e:
                print(f"  {ad:30s} HATA {e}", flush=True)
    json.dump(sonuc, open("m34_supurme.json", "w"), indent=1)
