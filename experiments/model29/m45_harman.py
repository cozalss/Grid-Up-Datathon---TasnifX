"""huber + l1 + catboost harmani geri-testte kazandiriyor mu? (seviye-yanliligi giderilmis olcum)"""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from m33_durust import VARSAYILAN, hizala
from m34_supurme import al

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


def veriler(dog):
    Xva, yva = al(dog, None)
    sog = Xva.soguk.values.astype(bool)
    Xs, ys = [], []
    for k in [x for x in AY if x < dog]:
        r = al(k, dog)
        if r:
            Xs.append(r[0])
            ys.append(r[1])
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    return hizala(Xtr.copy(), Xva.copy()), ytr, yva, sog


def lgbm(Xtr, ytr, Xva, yva, **pk):
    p = dict(VARSAYILAN)
    p.update(pk)
    m = lgb.train(
        p,
        lgb.Dataset(Xtr, ytr),
        4000,
        valid_sets=[lgb.Dataset(Xva, yva)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return m.predict(Xva, num_iteration=m.best_iteration)


def cb(Xtr, ytr, Xva, yva, kayip="RMSE"):
    kat = ["il", "bolge", "ilce"]
    A = Xtr.copy()
    B = Xva.copy()
    for c in kat:
        A[c] = A[c].astype(str).fillna("YOK").replace("nan", "YOK")
        B[c] = B[c].astype(str).fillna("YOK").replace("nan", "YOK")
    m = CatBoostRegressor(
        loss_function=kayip,
        depth=8,
        learning_rate=0.06,
        iterations=1200,
        l2_leaf_reg=5.0,
        random_seed=7,
        verbose=False,
        thread_count=14,
        early_stopping_rounds=120,
    )
    m.fit(Pool(A, ytr, cat_features=kat), eval_set=Pool(B, yva, cat_features=kat))
    return m.predict(B)


def skor(p, y, sog, capa=True):
    if capa:
        p = p.copy()
        p[sog] -= (p[sog] - y[sog]).mean()
        p[~sog] -= (p[~sog] - y[~sog]).mean()
    L = (p - y) ** 2
    return float(np.sqrt(0.7784 * L[~sog].mean() + 0.2216 * L[sog].mean()))


hepsi = {}
for dog in ["2025-11-30", "2025-09-30"]:
    t0 = time.time()
    (Xtr, Xva), ytr, yva, sog = veriler(dog)
    P = {}
    P["huber"] = lgbm(Xtr, ytr, Xva, yva, objective="huber", alpha=2.0, lambda_l2=20.0)
    print(f"  huber ({time.time() - t0:.0f}s)", flush=True)
    P["l1"] = lgbm(Xtr, ytr, Xva, yva, objective="l1")
    print(f"  l1 ({time.time() - t0:.0f}s)", flush=True)
    P["cb_mae"] = cb(Xtr, ytr, Xva, yva, "MAE")
    print(f"  catboost MAE ({time.time() - t0:.0f}s)", flush=True)
    P["h+l1"] = (P["huber"] + P["l1"]) / 2
    P["h+cb"] = (P["huber"] + P["cb_mae"]) / 2
    P["h+l1+cb"] = (P["huber"] + P["l1"] + P["cb_mae"]) / 3
    print(f"### {dog}  (capali=seviye yanliligi giderilmis, test-karisimi)")
    for ad, p in P.items():
        print(f"   {ad:10s} capali {skor(p, yva, sog):.4f}   ham {skor(p, yva, sog, False):.4f}")
        hepsi.setdefault(ad, {})[dog] = dict(capali=skor(p, yva, sog), ham=skor(p, yva, sog, False))
json.dump(hepsi, open("m45_harman.json", "w"), indent=1)
print("\n=== IKI KESIM ORTALAMASI (capali) ===")
for ad, d in sorted(hepsi.items(), key=lambda kv: np.mean([v["capali"] for v in kv[1].values()])):
    print(
        f"  {ad:10s} {np.mean([v['capali'] for v in d.values()]):.4f}   "
        + "  ".join(f"{k}:{v['capali']:.4f}" for k, v in d.items())
    )
