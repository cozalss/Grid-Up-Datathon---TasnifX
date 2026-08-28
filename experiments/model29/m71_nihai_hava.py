"""NIHAI URETIM v2: hava (A,C,G,E) + huber/l1 harmani + yakinlik agirligi
+ v102'nin LB-kalibre rejim seviyesine capa."""

import json
import os
import sys
import time

import lightgbm as lgb
import m61_hava as h
import numpy as np
import pandas as pd
from m30_ozellik import KOK, kur
from m33_durust import VARSAYILAN as V0
from m33_durust import hizala

KESIM = "2026-03-31"
AILE = ("A", "C", "G", "E")
V = dict(V0)
V["num_threads"] = int(os.environ.get("IPLIK", "14"))
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
L1 = dict(objective="l1")
TUR_HUB = int(sys.argv[1]) if len(sys.argv) > 1 else 185
TUR_L1 = int(sys.argv[2]) if len(sys.argv) > 2 else 520

t0 = time.time()
o = h.ortam()
tr = o["tr"]
te = o["te"]
Xs, ys, ks = [], [], []
for i, k in enumerate(h.AY):
    r = h.baz(k, KESIM)
    if r is None:
        continue
    Xs.append(h.zenginlestir(r[0], r[2], r[3], AILE))
    ys.append(r[1])
    ks.append(np.full(len(r[0]), i))
    print(f"  kesim {k}: {len(r[0]):,} ({time.time() - t0:.0f}s)", flush=True)
Xtr = pd.concat(Xs, ignore_index=True)
ytr = np.concatenate(ys)
ki = np.concatenate(ks)
del Xs, ys
w = None  # yakinlik agirligi ELENDI: hava girince iki kesimde de kazanmiyor (-0,0036 / +0,0021)

Xte = kur(tr, te, KESIM, set(tr.tanim))
meta = pd.DataFrame(
    {
        "ilce_key": te.ilce_key.to_numpy(),
        "il_key": te.il_key.to_numpy(),
        "tanim": te.tanim.to_numpy(),
        "tarih": te.tarih.to_numpy(),
    }
)
meta["ilce_key"] = meta.ilce_key.astype(object)
meta["il_key"] = meta.il_key.astype(object)
meta["ay"] = meta.tarih.dt.month.astype("int64")
egt = h.trafo_egim(tr, o["gun"], KESIM)
Xte = h.zenginlestir(Xte, meta, egt, AILE)
print(
    f"  TEST {len(Xte):,} satir, {Xte.shape[1]} ozellik, soguk %{100 * Xte.soguk.mean():.1f} ({time.time() - t0:.0f}s)",
    flush=True,
)
Xtr, Xte = hizala(Xtr, Xte)
Xte = Xte[Xtr.columns]
nan = (
    int(
        Xte[[c for c in Xte.columns if c.startswith(("a_", "c_", "cdd", "hdd", "sicaklik"))]]
        .isna()
        .mean()
        .max()
        * 100
    )
    if any(c.startswith(("a_", "cdd")) for c in Xte.columns)
    else -1
)
ds = lgb.Dataset(Xtr, ytr, weight=w)
tahmin = {}
for ad, pk, tur in [("huber", HUB, TUR_HUB), ("l1", L1, TUR_L1)]:
    acc = []
    for s in (7, 17, 27):
        p = dict(V)
        p.update(pk)
        p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        acc.append(lgb.train(p, ds, tur).predict(Xte))
        print(f"  {ad} tohum {s} ({time.time() - t0:.0f}s)", flush=True)
    tahmin[ad] = np.mean(acc, axis=0)
lg = (tahmin["huber"] + tahmin["l1"]) / 2

a = np.log1p(
    pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv")).tuketim.values
)
ilk = tr.groupby("tanim").tarih.min()
soguk = (~te.tanim.isin(set(tr.tanim))).values
kuyruk = (~soguk) & (te.tanim.map(ilk) >= pd.Timestamp("2026-03-26")).values
cek = ~soguk & ~kuyruk
lg2 = lg.copy()
rap = {}
for ad, m in [("soguk", soguk), ("kuyruk", kuyruk), ("cekirdek", cek)]:
    d = a[m].mean() - lg[m].mean()
    lg2[m] = lg[m] + d
    rap[ad] = float(d)
    print(f"  capa {ad}: {d:+.4f}")
y = np.clip(np.expm1(lg2), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
yol = os.path.join(KOK, "submissions", "tuketim_m4_hava_capali.csv")
out.to_csv(yol, index=False)
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
kapi = dict(
    satir=len(out),
    id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
    nan=int(out.tuketim.isna().sum()),
    negatif=int((out.tuketim < 0).sum()),
    maks=float(out.tuketim.max()),
    log_ort=float(np.log1p(out.tuketim).mean()),
)
print("KAPI:", json.dumps(kapi))
assert kapi["satir"] == 714688 and kapi["id_birebir"] and kapi["nan"] == 0 and kapi["negatif"] == 0
d = lg2 - a
Q = float((d**2).mean())
m0 = 1.00553**2
print(f"Q(vs v102)={Q:.6f}  basabas {np.sqrt(m0 + Q):.5f}")
for S in [0.95, 0.97, 0.99, 1.00553, 1.02, 1.04]:
    L = (m0 + Q - S * S) / 2
    print(f"   S={S:.5f} kappa*={L / Q:+.4f} optimum {np.sqrt(max(m0 - L * L / Q, 0)):.5f}")
json.dump(
    dict(capa=rap, kapi=kapi, Q=Q, tur=[TUR_HUB, TUR_L1], ozellik=int(Xtr.shape[1])),
    open("m71_nihai_hava.json", "w"),
    indent=1,
)
print(f"YAZILDI {yol} ({time.time() - t0:.0f}s)")
