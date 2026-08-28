"""NIHAI: huber + l1 harmani, 3'er tohum, v102'nin LB-kalibre seviyesine rejim-bazli capa."""

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from m30_ozellik import KOK, kur, yukle_ham
from m33_durust import VARSAYILAN, hizala, parca

KESIM = "2026-03-31"
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
t0 = time.time()
tr, te = yukle_ham()
Xs, ys = [], []
for k in AY:
    r = parca(tr, k, tavan=KESIM)
    if r:
        Xs.append(r[0])
        ys.append(r[1])
        print(f"  {k}: {len(r[0]):,} ({time.time() - t0:.0f}s)", flush=True)
Xtr = pd.concat(Xs, ignore_index=True)
ytr = np.concatenate(ys)
del Xs, ys
Xte = kur(tr, te, KESIM, set(tr.tanim))
print(
    f"  TEST {len(Xte):,} soguk %{100 * Xte.soguk.mean():.1f} ({time.time() - t0:.0f}s)", flush=True
)
Xtr, Xte = hizala(Xtr, Xte)
Xte = Xte[Xtr.columns]
ds = lgb.Dataset(Xtr, ytr)
AILE = [
    ("huber", dict(objective="huber", alpha=2.0, lambda_l2=20.0), 170),
    ("l1", dict(objective="l1"), 400),
]
parcalar = {}
for ad, pk, tur in AILE:
    tp = []
    for s in (7, 17, 27):
        p = dict(VARSAYILAN)
        p.update(pk)
        p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        tp.append(lgb.train(p, ds, tur).predict(Xte))
        print(f"  {ad} tohum {s} ({time.time() - t0:.0f}s)", flush=True)
    parcalar[ad] = np.mean(tp, axis=0)
lg = (parcalar["huber"] + parcalar["l1"]) / 2

# --- rejim bazli seviye capasi (v102 LB-kalibre) ---
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
    rap[ad] = dict(satir=int(m.sum()), kaydirma=float(d))
    print(f"  capa {ad}: {d:+.4f}")
y = np.clip(np.expm1(lg2), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
yol = os.path.join(KOK, "submissions", "tuketim_m3_hl1_capali.csv")
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
print(f"Q(vs v102) = {Q:.6f}   basabas skor {np.sqrt(m0 + Q):.5f}")
for S in [0.95, 0.97, 0.99, 1.00553, 1.02, 1.04]:
    L = (m0 + Q - S * S) / 2
    print(f"   S={S:.5f}  kappa*={L / Q:+.4f}  optimum {np.sqrt(max(m0 - L * L / Q, 0)):.5f}")
json.dump(dict(capa=rap, kapi=kapi, Q=Q), open("m46_nihai.json", "w"), indent=1)
print(f"YAZILDI {yol}  ({time.time() - t0:.0f}s)")
