"""Kesim agirliklandirma, rejim uzmani, XGBoost uyesi. Iki kesimde de kazanmayan alinmaz."""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
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
VARSAYILAN = dict(VARSAYILAN)
VARSAYILAN["num_threads"] = 7


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
    kidx = np.concatenate(ks)
    Xtr, Xva2 = hizala(Xtr.copy(), Xva.copy())
    return Xtr, ytr, kidx, Xva2, yva, sog


def karisim(p, y, sog):
    p = p.copy()
    p[sog] -= (p[sog] - y[sog]).mean()
    p[~sog] -= (p[~sog] - y[~sog]).mean()
    L = (p - y) ** 2
    return float(np.sqrt(0.7784 * L[~sog].mean() + 0.2216 * L[sog].mean()))


def lg_egit(Xtr, ytr, Xva, yva, w=None, **pk):
    p = dict(VARSAYILAN)
    p.update(pk)
    ds = lgb.Dataset(Xtr, ytr, weight=w)
    m = lgb.train(
        p,
        ds,
        4000,
        valid_sets=[lgb.Dataset(Xva, yva)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return m.predict(Xva, num_iteration=m.best_iteration)


HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
L1 = dict(objective="l1")
sonuc = {}
for dog in ["2025-11-30", "2025-09-30"]:
    t0 = time.time()
    Xtr, ytr, kidx, Xva, yva, sog = veri(dog)
    nk = kidx.max() + 1
    R = {}
    ph = lg_egit(Xtr, ytr, Xva, yva, **HUB)
    pl = lg_egit(Xtr, ytr, Xva, yva, **L1)
    R["TABAN h+l1"] = (ph + pl) / 2
    print(f"  taban ({time.time() - t0:.0f}s)", flush=True)
    for ad, gam in [("yakinlik w=1.15^i", 1.15), ("yakinlik w=1.35^i", 1.35)]:
        w = gam**kidx
        a = lg_egit(Xtr, ytr, Xva, yva, w=w, **HUB)
        b = lg_egit(Xtr, ytr, Xva, yva, w=w, **L1)
        R[ad] = (a + b) / 2
        print(f"  {ad} ({time.time() - t0:.0f}s)", flush=True)
    # rejim uzmani: soguk ve sicak icin AYRI model
    ms = Xtr.soguk.values.astype(bool)
    vs = Xva.soguk.values.astype(bool)
    ps = np.empty(len(yva))
    for m_tr, m_va in [(ms, vs), (~ms, ~vs)]:
        if m_va.sum() == 0:
            continue
        a = lg_egit(Xtr[m_tr], ytr[m_tr], Xva[m_va], yva[m_va], **HUB)
        b = lg_egit(Xtr[m_tr], ytr[m_tr], Xva[m_va], yva[m_va], **L1)
        ps[m_va] = (a + b) / 2
    R["rejim uzmani"] = ps
    print(f"  rejim uzmani ({time.time() - t0:.0f}s)", flush=True)
    R["taban + rejim ort"] = (R["TABAN h+l1"] + ps) / 2
    # daha yavas/derin
    a = lg_egit(Xtr, ytr, Xva, yva, learning_rate=0.02, **HUB)
    b = lg_egit(Xtr, ytr, Xva, yva, learning_rate=0.02, **L1)
    R["lr 0.02"] = (a + b) / 2
    print(f"  lr0.02 ({time.time() - t0:.0f}s)", flush=True)
    R["taban + lr0.02"] = (R["TABAN h+l1"] + R["lr 0.02"]) / 2
    print(f"### {dog}")
    for ad, p in R.items():
        s = karisim(p, yva, sog)
        sonuc.setdefault(ad, {})[dog] = s
        print(f"   {ad:22s} {s:.4f}")
json.dump(sonuc, open("m62_cesitlilik.json", "w"), indent=1)
print("\n=== IKI KESIM (taban farki, - = iyilesme) ===")
tb = {k: v for k, v in sonuc["TABAN h+l1"].items()}
for ad, d in sonuc.items():
    dl = {k: d[k] - tb[k] for k in d}
    isaret = (
        "KAZANDI"
        if all(v < -0.0005 for v in dl.values())
        else ("kaybetti" if all(v > 0 for v in dl.values()) else "KARISIK")
    )
    print(f"  {ad:22s} " + "  ".join(f"{k}:{v:+.4f}" for k, v in dl.items()) + f"   {isaret}")
