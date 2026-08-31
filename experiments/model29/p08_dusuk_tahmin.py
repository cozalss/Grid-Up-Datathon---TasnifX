"""DUSUK TAHMIN bolgesi: p<4 satirlarin %11,8'i kare hatanin %36,8'ini
tasiyor ve HEPSI ayni yonde (model az tahmin ediyor). Bu kesit yalnizca
TAHMINE bakarak tanimlanir -> gercek testte de uygulanabilir.
Ofset dis bloklardan kestirilir."""

import json
import os

import numpy as np
import pandas as pd
from p02_duzeltme import blok, skor

BURA = os.path.dirname(os.path.abspath(__file__))
yaz = blok("yaz25")
dis = pd.concat([blok("guz25"), blok("kis26")], ignore_index=True)
t0, t0w = skor(yaz, yaz.p.values)
R = dict(taban=dict(rmsle=round(t0, 5), rmsle_test_bilesimi=round(t0w, 5)))

KEN = [0.7, 2, 3, 4, 5, 6, 7, 8, 9]


def tablo(d):
    b = np.clip(np.digitize(d.p.values, KEN), 0, len(KEN))
    return pd.DataFrame(dict(b=b, r=d.r.values)).groupby("b").r.agg(["mean", "size"])


R["p_kova_yanlilik"] = {}
for ad, d in (("yaz25", yaz), ("guz25", blok("guz25")), ("kis26", blok("kis26"))):
    t = tablo(d)
    R["p_kova_yanlilik"][ad] = {int(k): [round(float(v["mean"]), 3), int(v["size"])] for k, v in t.iterrows()}

# dis bloklarin BLOK-ICI ORTALAMASI CIKARILMIS kova ofseti (seviye kaymasi transfer etmiyor)
off = np.zeros(len(KEN) + 1)
say = np.zeros(len(KEN) + 1)
for b, g in dis.groupby("_blok", observed=True):
    t = tablo(g)
    m = float(g.r.mean())
    for k, v in t.iterrows():
        off[k] += (v["mean"] - m) * v["size"]
        say[k] += v["size"]
off = np.where(say > 0, off / np.maximum(say, 1), 0.0)
R["dis_ofset_merkezli"] = [round(float(v), 4) for v in off]
by = np.clip(np.digitize(yaz.p.values, KEN), 0, len(KEN))
D = {}
for lam in (0.25, 0.5, 1.0):
    D[f"P1_kova_ofset_lam{lam}"] = yaz.p.values + lam * off[by]
# yalnizca dusuk bolge (p<4) yukseltilir
o2 = off.copy()
o2[4:] = 0.0
for lam in (0.5, 1.0):
    D[f"P2_yalniz_dusuk_lam{lam}"] = yaz.p.values + lam * o2[by]

R["duzeltmeler"] = {}
for ad, x in D.items():
    s, sw = skor(yaz, x)
    R["duzeltmeler"][ad] = dict(rmsle=round(s, 5), kazanc=round(t0 - s, 5),
                                kazanc_test_bilesimi=round(t0w - sw, 5))
    print(f"{ad:24s} RMSLE={s:.5f} kazanc={t0 - s:+.5f} (agirlikli {t0w - sw:+.5f})")

# tavan: yaz25'in KENDI kova ofseti (sizintili)
ty = tablo(yaz)
oy = np.zeros(len(KEN) + 1)
for k, v in ty.iterrows():
    oy[k] = v["mean"]
s, _ = skor(yaz, yaz.p.values + oy[by])
R["tavan_kendi_kova_ofseti"] = dict(rmsle=round(s, 5), kazanc=round(t0 - s, 5))
print("tavan (sizintili):", R["tavan_kendi_kova_ofseti"])
print(json.dumps(R["p_kova_yanlilik"], ensure_ascii=False))
json.dump(R, open(os.path.join(BURA, "p08_dusuk_tahmin.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
