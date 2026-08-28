"""HAVA + h/l1 harmani + yakinlik agirligi: iki kesimde de dogrula."""

import json
import time

import lightgbm as lgb
import m61_hava as h
import numpy as np
import pandas as pd
from m33_durust import VARSAYILAN as V0
from m33_durust import hizala

V = dict(V0)
V["num_threads"] = 8
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
L1 = dict(objective="l1")
AILE = ("A", "C", "G", "E")


def veri(dog, aileler):
    Xva, yva, mva, egv = h.baz(dog, None)
    Xva = h.zenginlestir(Xva, mva, egv, aileler)
    sog = Xva.soguk.values.astype(bool)
    Xs, ys, ks = [], [], []
    for i, k in enumerate([x for x in h.AY if x < dog]):
        r = h.baz(k, dog)
        if r is None:
            continue
        Xs.append(h.zenginlestir(r[0], r[2], r[3], aileler))
        ys.append(r[1])
        ks.append(np.full(len(r[0]), i))
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    ki = np.concatenate(ks)
    del Xs
    Xtr, Xva2 = hizala(Xtr, Xva)
    return Xtr, ytr, ki, Xva2, yva, sog


def kar(p, y, sog):
    p = p.copy()
    p[sog] -= (p[sog] - y[sog]).mean()
    p[~sog] -= (p[~sog] - y[~sog]).mean()
    L = (p - y) ** 2
    return float(np.sqrt(0.7784 * L[~sog].mean() + 0.2216 * L[sog].mean()))


def eg(X, y, Xv, yv, w=None, **pk):
    p = dict(V)
    p.update(pk)
    m = lgb.train(
        p,
        lgb.Dataset(X, y, weight=w),
        4000,
        valid_sets=[lgb.Dataset(Xv, yv)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return m.predict(Xv, num_iteration=m.best_iteration), m.best_iteration


sonuc = {}
turlar = {}
for dog in ["2025-11-30", "2025-09-30"]:
    t0 = time.time()
    for etiket, aileler in [("HAVASIZ", ()), ("HAVALI", AILE)]:
        Xtr, ytr, ki, Xva, yva, sog = veri(dog, aileler)
        for wad, w in [("duz", None), ("yakinlik", 1.15**ki)]:
            a, na = eg(Xtr, ytr, Xva, yva, w=w, **HUB)
            b, nb = eg(Xtr, ytr, Xva, yva, w=w, **L1)
            s = kar((a + b) / 2, yva, sog)
            ad = f"{etiket} {wad}"
            sonuc.setdefault(ad, {})[dog] = s
            turlar.setdefault(ad, {})[dog] = [na, nb]
            print(
                f"  {dog} {ad:18s} {s:.4f}  (tur {na}/{nb}, ozellik {Xtr.shape[1]}, {time.time() - t0:.0f}s)",
                flush=True,
            )
        del Xtr, Xva
json.dump(dict(skor=sonuc, tur=turlar), open("m70_hava_harman.json", "w"), indent=1)
print("\n=== OZET (test-karisimi, capali) ===")
for ad, d in sorted(sonuc.items(), key=lambda kv: np.mean(list(kv[1].values()))):
    print(
        f"  {ad:18s} ort {np.mean(list(d.values())):.4f}   "
        + "  ".join(f"{k}:{v:.4f}" for k, v in d.items())
    )
