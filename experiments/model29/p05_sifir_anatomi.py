"""Sifir satirlarin anatomisi: model onlari zaten biliyor mu, bilmiyor mu?
Ve dis-blok kalibrasyonu neden transfer etmiyor?"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from p02_duzeltme import blok, skor
from p04_sifir import OZ

BURA = os.path.dirname(os.path.abspath(__file__))

yaz = blok("yaz25")
dis = pd.concat([blok("guz25"), blok("kis26")], ignore_index=True)
z = (dis.tuketim.values <= 0).astype(int)
m = lgb.train(
    dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7, num_threads=8),
    lgb.Dataset(dis[OZ], z), 500)
qd, qy = m.predict(dis[OZ]), m.predict(yaz[OZ])
yaz["q"], dis["q"] = qy, qd
R = {}

# --- sifir satirlarin tahmin dagilimi: model zaten dusuk mu tahmin ediyor?
s = yaz[yaz.tuketim <= 0]
tk = float((yaz.r**2).sum())
R["yaz25_sifir_satirlar"] = dict(
    n=int(len(s)),
    kare_pay=round(float((s.r**2).sum() / tk), 4),
    p_medyan=round(float(s.p.median()), 3),
    p_ort=round(float(s.p.mean()), 3),
    p_kucuk_1_orani=round(float((s.p < 1).mean()), 4),
    # tahmini zaten <1 olan sifirlarin kare paya katkisi
    kare_pay_p_kucuk_1=round(float((s.loc[s.p < 1, "r"] ** 2).sum() / tk), 4),
    kare_pay_p_buyuk_4=round(float((s.loc[s.p > 4, "r"] ** 2).sum() / tk), 4),
    n_p_buyuk_4=int((s.p > 4).sum()),
    q_medyan_p_buyuk_4=round(float(s.loc[s.p > 4, "q"].median()), 4),
)
# --- yuksek q kovalarinda GERCEKTEN ne var
QK = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]
tab = []
for i, (lo, hi) in enumerate(zip([0] + QK, QK + [1.0])):
    for ad, d in (("yaz25", yaz), ("dis", dis)):
        mk = (d.q >= lo) & (d.q < hi)
        if mk.sum() == 0:
            continue
        dd = d[mk]
        tab.append(dict(
            kova=f"[{lo:.2f},{hi:.2f})", blok=ad, n=int(len(dd)),
            gercek_sifir_orani=round(float((dd.tuketim <= 0).mean()), 4),
            p_ort=round(float(dd.p.mean()), 3),
            y_ort=round(float(dd.y.mean()), 3),
            en_iyi_ofset=round(float(dd.r.mean()), 3),
            kare_pay=round(float((dd.r**2).sum() / float((d.r**2).sum())), 4),
        ))
R["q_kovalari"] = tab

# --- yanlis pozitif maliyeti: q yuksek ama tuketim buyuk olan satirlar
hi = yaz[yaz.q > 0.5]
R["yaz25_q_ust_0.5"] = dict(
    n=int(len(hi)),
    gercek_sifir=round(float((hi.tuketim <= 0).mean()), 4),
    canli_ort_y=round(float(hi.loc[hi.tuketim > 0, "y"].mean()), 3),
    n_canli=int((hi.tuketim > 0).sum()),
)

# --- sifir satirlar trafo bazinda mi gun bazinda mi?
sg = yaz.groupby("tanim", observed=True).apply(lambda d: (d.tuketim <= 0).mean(), include_groups=False)
R["sifir_trafo_profili"] = dict(
    tam_olu_trafo=int((sg > 0.99).sum()),
    kismi_sifir_trafo=int(((sg > 0.01) & (sg <= 0.99)).sum()),
    hic_sifir_yok=int((sg <= 0.01).sum()),
)
# kismi sifirlarin (kesintilerin) kare hatadaki payi
kis = set(sg[(sg > 0.01) & (sg <= 0.99)].index)
mk = yaz.tanim.isin(kis) & (yaz.tuketim <= 0)
R["kismi_sifir_kare_pay"] = round(float((yaz.loc[mk, "r"] ** 2).sum() / tk), 4)
tam = set(sg[sg > 0.99].index)
mk2 = yaz.tanim.isin(tam) & (yaz.tuketim <= 0)
R["tam_olu_kare_pay"] = round(float((yaz.loc[mk2, "r"] ** 2).sum() / tk), 4)

print(json.dumps(R, indent=1, ensure_ascii=False))
json.dump(R, open(os.path.join(BURA, "p05_sifir_anatomi.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
