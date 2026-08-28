"""m4 uzerine iki denenmemis eksen: (a) rejim uzmani + hava, (b) sik kesim (2 haftada bir).
Kural: IKI dogrulama kesiminde de kazanmayan ALINMAZ."""

import json
import time

import lightgbm as lgb
import m61_hava as h
import numpy as np
import pandas as pd
from m33_durust import VARSAYILAN as V0
from m33_durust import hizala

V = dict(V0)
V["num_threads"] = 14
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
L1 = dict(objective="l1")
AILE = ("A", "C", "G", "E")
SIK = [str(d.date()) for d in pd.date_range("2025-03-15", "2025-12-31", freq="SME")]


def veri(dog, kesimler):
    Xva, yva, mva, egv = h.baz(dog, None)
    Xva = h.zenginlestir(Xva, mva, egv, AILE)
    sog = Xva.soguk.values.astype(bool)
    Xs, ys = [], []
    for k in [x for x in kesimler if x < dog]:
        r = h.baz(k, dog)
        if r is None:
            continue
        Xs.append(h.zenginlestir(r[0], r[2], r[3], AILE))
        ys.append(r[1])
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    del Xs
    Xtr, Xva2 = hizala(Xtr, Xva)
    return Xtr, ytr, Xva2, yva, sog


def kar(p, y, sog):
    p = p.copy()
    p[sog] -= (p[sog] - y[sog]).mean()
    p[~sog] -= (p[~sog] - y[~sog]).mean()
    L = (p - y) ** 2
    return float(np.sqrt(0.7784 * L[~sog].mean() + 0.2216 * L[sog].mean()))


def eg(X, y, Xv, yv, **pk):
    p = dict(V)
    p.update(pk)
    m = lgb.train(
        p,
        lgb.Dataset(X, y),
        4000,
        valid_sets=[lgb.Dataset(Xv, yv)],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return m.predict(Xv, num_iteration=m.best_iteration)


def havuz(Xtr, ytr, Xva, yva):
    return (eg(Xtr, ytr, Xva, yva, **HUB) + eg(Xtr, ytr, Xva, yva, **L1)) / 2


def rejim(Xtr, ytr, Xva, yva):
    ms = Xtr.soguk.values.astype(bool)
    vs = Xva.soguk.values.astype(bool)
    p = np.empty(len(yva))
    for a, b in [(ms, vs), (~ms, ~vs)]:
        if b.sum() == 0:
            continue
        p[b] = (
            eg(Xtr[a], ytr[a], Xva[b], yva[b], **HUB) + eg(Xtr[a], ytr[a], Xva[b], yva[b], **L1)
        ) / 2
    return p


sonuc = {}
for dog in ["2025-11-30", "2025-09-30"]:
    t0 = time.time()
    Xtr, ytr, Xva, yva, sog = veri(dog, h.AY)
    R = {}
    R["m4 (aylik havuz)"] = havuz(Xtr, ytr, Xva, yva)
    print(f"  havuz ({time.time() - t0:.0f}s)", flush=True)
    rj = rejim(Xtr, ytr, Xva, yva)
    print(f"  rejim ({time.time() - t0:.0f}s)", flush=True)
    R["rejim uzmani"] = rj
    R["havuz+rejim ort"] = (R["m4 (aylik havuz)"] + rj) / 2
    n_ay = Xtr.shape[0]
    del Xtr, Xva
    Xtr, ytr, Xva, yva2, sog2 = veri(dog, SIK)
    R["SIK kesim havuz"] = havuz(Xtr, ytr, Xva, yva2)
    print(f"  sik ({time.time() - t0:.0f}s)", flush=True)
    n_sik = Xtr.shape[0]
    del Xtr, Xva
    print(f"### {dog}   aylik egitim {n_ay:,} | sik egitim {n_sik:,}")
    for ad, p in R.items():
        y = yva2 if ad.startswith("SIK") else yva
        s = kar(p, y, sog)
        sonuc.setdefault(ad, {})[dog] = s
        print(f"   {ad:20s} {s:.4f}")
json.dump(sonuc, open("m81_daha.json", "w"), indent=1)
print("\n=== m4'e GORE (- = iyilesme) ===")
tb = sonuc["m4 (aylik havuz)"]
for ad, d in sonuc.items():
    dl = {k: d[k] - tb[k] for k in d}
    hk = (
        "KAZANDI"
        if all(v < -0.0005 for v in dl.values())
        else ("kaybetti" if all(v > 0 for v in dl.values()) else "KARISIK")
    )
    print(f"  {ad:20s} " + "  ".join(f"{k}:{v:+.4f}" for k, v in dl.items()) + f"   {hk}")
