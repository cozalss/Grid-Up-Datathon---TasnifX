"""KRITIK BOSLUK: hava ozellikleri SOGUTMA hakimiyetli bir pencerede dogrulanmadi.
Dogrulama kesimlerim (11-30 -> Ara-Mar, 09-30 -> Eki-Oca) CDD22 ~ 0.
Test penceresi Nis-Tem = KLIMA sezonu.
Burada YAZ pencerelerinde ayni karsilastirmayi yapiyorum. Egitim verisi ince
kalacak (yalniz onceki kesimler) ama havali/havasiz KIYASI adil."""

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


def veri(dog, aileler):
    Xva, yva, mva, egv = h.baz(dog, None)
    Xva = h.zenginlestir(Xva, mva, egv, aileler)
    sog = Xva.soguk.values.astype(bool)
    Xs, ys = [], []
    for k in [x for x in h.AY if x < dog]:
        r = h.baz(k, dog)
        if r is None:
            continue
        Xs.append(h.zenginlestir(r[0], r[2], r[3], aileler))
        ys.append(r[1])
    if not Xs:
        return None
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


# hedef penceredeki gercek CDD22 yukunu de raporla
o = h.ortam()
gun = o["gun"].reset_index()


def cdd(dog):
    k = pd.Timestamp(dog)
    s = k + pd.DateOffset(months=4)
    g = gun[(gun.tarih > k) & (gun.tarih <= s)]
    return float(g["cdd22"].mean()) if "cdd22" in g.columns else float("nan")


sonuc = {}
for dog in ["2025-05-31", "2025-06-30", "2025-11-30"]:
    r0 = veri(dog, ())
    if r0 is None:
        print(f"{dog}: egitim verisi yok")
        continue
    t0 = time.time()
    p0 = (eg(*r0[:4], **HUB) + eg(*r0[:4], **L1)) / 2
    s0 = kar(p0, r0[3], r0[4])
    n0 = r0[0].shape[0]
    del r0
    r1 = veri(dog, AILE)
    p1 = (eg(*r1[:4], **HUB) + eg(*r1[:4], **L1)) / 2
    s1 = kar(p1, r1[3], r1[4])
    print(
        f"{dog} -> hedef {pd.Timestamp(dog) + pd.Timedelta(days=1):%b}..{pd.Timestamp(dog) + pd.DateOffset(months=4):%b}"
        f"  ort CDD22 {cdd(dog):.3f}  egitim {n0:,} satir"
    )
    print(
        f"    HAVASIZ {s0:.4f}   HAVALI {s1:.4f}   KAZANC {s0 - s1:+.4f}   ({time.time() - t0:.0f}s)",
        flush=True,
    )
    sonuc[dog] = dict(cdd=cdd(dog), havasiz=s0, havali=s1, kazanc=s0 - s1, n_egit=int(n0))
    del r1
json.dump(sonuc, open("m80_yaz.json", "w"), indent=1)
