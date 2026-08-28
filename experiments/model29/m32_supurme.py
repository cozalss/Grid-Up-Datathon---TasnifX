"""Konfig supurmesi: kesim havuzu, idnum, agac boyutu. Iki ayri dogrulama kesimi."""

import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from m30_ozellik import yukle_ham
from m31_egit import veri

tr, te = yukle_ham()
ONBELLEK = {}


def al(k):
    if k not in ONBELLEK:
        ONBELLEK[k] = veri(tr, k)
    return ONBELLEK[k]


TUM = ["2025-05-31", "2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31"]


def kos(egit, dog, drop=(), **pk):
    Xs, ys = [], []
    for k in egit:
        X, y, _ = al(k)
        Xs.append(X)
        ys.append(y)
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    Xva, yva, _ = al(dog)
    Xtr = Xtr.drop(columns=list(drop))
    Xva = Xva.drop(columns=list(drop))
    for c in ("il", "bolge", "ilce"):
        cats = Xtr[c].cat.categories.union(Xva[c].cat.categories)
        Xtr[c] = Xtr[c].cat.set_categories(cats)
        Xva[c] = Xva[c].cat.set_categories(cats)
    p = dict(
        objective="l2",
        metric="l2",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        num_threads=14,
        verbose=-1,
        seed=7,
    )
    p.update(pk)
    ds = lgb.Dataset(Xtr, ytr)
    dv = lgb.Dataset(Xva, yva, reference=ds)
    m = lgb.train(
        p,
        ds,
        num_boost_round=4000,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    pv = m.predict(Xva, num_iteration=m.best_iteration)
    L = (pv - yva) ** 2
    s = Xva.soguk.values.astype(bool)
    return dict(
        tur=m.best_iteration,
        rmsle=float(np.sqrt(L.mean())),
        soguk=float(np.sqrt(L[s].mean())),
        sicak=float(np.sqrt(L[~s].mean())),
    )


DOG = sys.argv[1] if len(sys.argv) > 1 else "2025-11-30"
egit_tum = [k for k in TUM if k < DOG]
deneyler = [
    ("tum kesimler, temel", egit_tum, (), {}),
    ("idnum YOK", egit_tum, ("idnum",), {}),
    (
        "varlik ozellikleri YOK",
        egit_tum,
        ("v_n", "v_ilk", "v_son", "v_aralik", "v_yogunluk", "v_dalga"),
        {},
    ),
    ("yaprak 127", egit_tum, (), dict(num_leaves=127)),
    ("yaprak 31, lr .03", egit_tum, (), dict(num_leaves=31, learning_rate=0.03)),
    ("tek kesim (en yakin)", egit_tum[-1:], (), {}),
    ("son 3 kesim", egit_tum[-3:], (), {}),
]
print(f"DOGRULAMA KESIMI {DOG}   (taban: son28+guc)")
t0 = time.time()
for ad, e, d, pk in deneyler:
    r = kos(e, DOG, d, **pk)
    print(
        f"  {ad:28s} tur {r['tur']:4d}  RMSLE {r['rmsle']:.4f}  soguk {r['soguk']:.4f}  sicak {r['sicak']:.4f}   ({time.time() - t0:.0f}s)",
        flush=True,
    )
