"""Hata SISTEMATIK mi RASGELE mi?  Ve trafo-seviyesi kaymasi TASINABILIR mi?"""

import json
import os

import numpy as np
import pandas as pd
from p02_duzeltme import blok, skor

BURA = os.path.dirname(os.path.abspath(__file__))
yaz = blok("yaz25")
t0, t0w = skor(yaz, yaz.p.values)
R = dict(taban=dict(rmsle=round(t0, 5), rmsle_test_bilesimi=round(t0w, 5)))

# --- A. Trafo kaymasi zaman icinde KALICI mi?  (yalnizca teshis; hedef kullanir)
yaz["yg"] = (pd.to_datetime(yaz.tarih) - pd.Timestamp("2025-04-01")).dt.days
ilk = yaz[yaz.yg < 30]
son = yaz[yaz.yg >= 30]
a = ilk.groupby("tanim", observed=True).r.mean()
b = son.groupby("tanim", observed=True).r.mean()
ort = a.index.intersection(b.index)
R["trafo_kaymasi_kaliciligi"] = dict(
    n_ortak=int(len(ort)),
    korelasyon_ilk30_son92=round(float(np.corrcoef(a[ort], b[ort])[0, 1]), 4),
)
# ilk 30 gunun kaymasini son 92 gune uygulasak (GERCEK TESTTE MUMKUN DEGIL)
off = yaz.tanim.map(a).fillna(0.0).values
s_, sw_ = skor(son, son.p.values + son.tanim.map(a).fillna(0.0).values)
b0, _ = skor(son, son.p.values)
R["ilk30_kayma_son92ye"] = dict(taban=round(b0, 5), duzeltilmis=round(s_, 5), kazanc=round(b0 - s_, 5))

# --- B. TASINABILIR mi: dis bloklarin trafo kaymasi yaz25'i duzeltir mi?
for ad, bl in (("kis26", ["kis26"]), ("guz25", ["guz25"]), ("ikisi", ["guz25", "kis26"])):
    d = pd.concat([blok(x) for x in bl], ignore_index=True)
    g = d.groupby("tanim", observed=True).r.agg(["mean", "size"])
    for lam in (0.3, 0.5, 1.0):
        # buzme: n/(n+k) ile ortalamaya cek
        k = 60.0
        sh = g["mean"] * g["size"] / (g["size"] + k)
        o = yaz.tanim.map(sh).fillna(0.0).values
        s, sw = skor(yaz, yaz.p.values + lam * o)
        R[f"dis_trafo_kaymasi_{ad}_lam{lam}"] = dict(
            rmsle=round(s, 5), kazanc=round(t0 - s, 5), kazanc_test_bilesimi=round(t0w - sw, 5)
        )

# --- C. Varyans ayristirmasi
tk = float((yaz.r**2).sum())
gt = yaz.groupby("tanim", observed=True).r.agg(["mean", "size"])
gd = yaz.groupby("tarih", observed=True).r.agg(["mean", "size"])
R["ayristirma"] = dict(
    kuresel_yanlilik=round(float(yaz.r.mean()) ** 2 * len(yaz) / tk, 4),
    trafo_sabiti=round(float((gt["size"] * gt["mean"] ** 2).sum()) / tk, 4),
    gun_sabiti=round(float((gd["size"] * gd["mean"] ** 2).sum()) / tk, 4),
)
# sifirsiz alt kumede ayni ayristirma
nz = yaz[yaz.tuketim > 0]
tnz = float((nz.r**2).sum())
gt2 = nz.groupby("tanim", observed=True).r.agg(["mean", "size"])
gd2 = nz.groupby("tarih", observed=True).r.agg(["mean", "size"])
R["ayristirma_sifirsiz"] = dict(
    n=int(len(nz)),
    rmsle=round(float(np.sqrt((nz.r**2).mean())), 5),
    kuresel_yanlilik=round(float(nz.r.mean()) ** 2 * len(nz) / tnz, 4),
    trafo_sabiti=round(float((gt2["size"] * gt2["mean"] ** 2).sum()) / tnz, 4),
    gun_sabiti=round(float((gd2["size"] * gd2["mean"] ** 2).sum()) / tnz, 4),
)
print(json.dumps(R, indent=1, ensure_ascii=False))
json.dump(R, open(os.path.join(BURA, "p06_sistematik.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
