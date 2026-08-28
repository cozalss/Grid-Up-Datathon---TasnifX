"""Ileri-pencere dogrudan tahmin modeli: coklu kesimde egit, tutulan kesimde dogrula."""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from m30_ozellik import kur, yukle_ham


def pencere(tr, kesim, ay=4):
    k = pd.Timestamp(kesim)
    son = k + pd.DateOffset(months=ay)
    gec = tr[tr.tarih <= k]
    hed = tr[(tr.tarih > k) & (tr.tarih <= son)]
    return gec, hed, set(gec.tanim)


def veri(tr, kesim, ay=4):
    gec, hed, sicak = pencere(tr, kesim, ay)
    X = kur(gec, hed, kesim, sicak)
    y = np.log1p(hed.tuketim.values)
    return X, y, hed


if __name__ == "__main__":
    t0 = time.time()
    tr, te = yukle_ham()
    EGIT = ["2025-07-31", "2025-08-31", "2025-09-30"]
    DOG = "2025-11-30"
    Xs, ys = [], []
    for k in EGIT:
        X, y, _ = veri(tr, k)
        Xs.append(X)
        ys.append(y)
        print(f"egitim kesimi {k}: {len(X):,} satir  ({time.time() - t0:.0f}s)", flush=True)
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    Xva, yva, hva = veri(tr, DOG)
    print(f"dogrulama {DOG}: {len(Xva):,} satir  ({time.time() - t0:.0f}s)", flush=True)
    for c in ("il", "bolge", "ilce"):
        cats = Xtr[c].cat.categories.union(Xva[c].cat.categories)
        Xtr[c] = Xtr[c].cat.set_categories(cats)
        Xva[c] = Xva[c].cat.set_categories(cats)
    p = dict(
        objective="l2",
        metric="l2",
        learning_rate=0.05,
        num_leaves=255,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=14,
        verbose=-1,
        seed=7,
    )
    ds = lgb.Dataset(Xtr, ytr)
    dv = lgb.Dataset(Xva, yva, reference=ds)
    m = lgb.train(
        p,
        ds,
        num_boost_round=3000,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    pv = m.predict(Xva, num_iteration=m.best_iteration)
    L = (pv - yva) ** 2
    sog = Xva.soguk.values.astype(bool)
    print(f"\nEN IYI TUR {m.best_iteration}   sure {time.time() - t0:.0f}s")
    print(
        f"MODEL  RMSLE {np.sqrt(L.mean()):.4f} | soguk {np.sqrt(L[sog].mean()):.4f} "
        f"(kutle %{100 * L[sog].sum() / L.sum():.1f}) | sicak {np.sqrt(L[~sog].mean()):.4f}"
    )
    print("TABAN (son28+guc)  RMSLE 0.9650 | soguk 1.8400 | sicak 0.7319")
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(
        ascending=False
    )
    print("\nen onemli 22 ozellik:")
    print((imp / imp.sum() * 100).head(22).round(2).to_string())
    np.save("m31_pv.npy", pv)
    np.save("m31_yva.npy", yva)
    np.save("m31_sog.npy", sog)
    m.save_model("m31_model.txt")
