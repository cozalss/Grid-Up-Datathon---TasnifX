"""m62'de IKI kesimde de kazanan uc unsuru birlestir: yakinlik agirligi + rejim uzmani ort + yavas lr."""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from m33_durust import VARSAYILAN as V0
from m33_durust import hizala
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
V = dict(V0)
V["num_threads"] = 7
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
L1 = dict(objective="l1")


def veri(dog):
    Xva, yva = al(dog, None)
    sog = Xva.soguk.values.astype(bool)
    Xs, ys, ks = [], [], []
    for i, k in enumerate([x for x in AY if x < dog]):
        r = al(k, dog)
        if r:
            Xs.append(r[0])
            ys.append(r[1])
            ks.append(np.full(len(r[0]), i))
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    ki = np.concatenate(ks)
    Xtr, Xva2 = hizala(Xtr.copy(), Xva.copy())
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
    return m.predict(Xv, num_iteration=m.best_iteration)


def aday(Xtr, ytr, ki, Xva, yva, lr, yakin):
    w = (1.15**ki) if yakin else None
    hav = (
        eg(Xtr, ytr, Xva, yva, w=w, learning_rate=lr, **HUB)
        + eg(Xtr, ytr, Xva, yva, w=w, learning_rate=lr, **L1)
    ) / 2
    ms = Xtr.soguk.values.astype(bool)
    vs = Xva.soguk.values.astype(bool)
    rej = np.empty(len(yva))
    for a, b in [(ms, vs), (~ms, ~vs)]:
        if b.sum() == 0:
            continue
        ww = None if w is None else w[a]
        rej[b] = (
            eg(Xtr[a], ytr[a], Xva[b], yva[b], w=ww, learning_rate=lr, **HUB)
            + eg(Xtr[a], ytr[a], Xva[b], yva[b], w=ww, learning_rate=lr, **L1)
        ) / 2
    return (hav + rej) / 2, hav


sonuc = {}
for dog in ["2025-11-30", "2025-09-30"]:
    t0 = time.time()
    Xtr, ytr, ki, Xva, yva, sog = veri(dog)
    taban = (eg(Xtr, ytr, Xva, yva, **HUB) + eg(Xtr, ytr, Xva, yva, **L1)) / 2
    print(f"  taban ({time.time() - t0:.0f}s)", flush=True)
    b1, _ = aday(Xtr, ytr, ki, Xva, yva, 0.03, True)
    print(f"  birlesik lr.03 ({time.time() - t0:.0f}s)", flush=True)
    b2, _ = aday(Xtr, ytr, ki, Xva, yva, 0.04, True)
    print(f"  birlesik lr.04 ({time.time() - t0:.0f}s)", flush=True)
    R = {
        "TABAN h+l1": taban,
        "BIRLESIK lr.03": b1,
        "BIRLESIK lr.04": b2,
        "BIRLESIK ort": (b1 + b2) / 2,
        "TABAN+BIRLESIK": (taban + b1) / 2,
    }
    print(f"### {dog}")
    for ad, p in R.items():
        s = kar(p, yva, sog)
        sonuc.setdefault(ad, {})[dog] = s
        print(f"   {ad:18s} {s:.4f}")
json.dump(sonuc, open("m63_birlesik.json", "w"), indent=1)
print("\n=== TABANA GORE (- = iyilesme) ===")
tb = sonuc["TABAN h+l1"]
for ad, d in sonuc.items():
    dl = {k: d[k] - tb[k] for k in d}
    print(
        f"  {ad:18s} "
        + "  ".join(f"{k}:{v:+.4f}" for k, v in dl.items())
        + ("   KAZANDI" if all(v < -0.0005 for v in dl.values()) else "")
    )
