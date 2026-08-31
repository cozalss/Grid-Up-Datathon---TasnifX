"""Neden dis-bloktan kestirilen duzeltmeler yaz25'te ISE YARAMIYOR?
Bloklar arasi artik yapisini yan yana koyar."""

import json
import os

import numpy as np
import pandas as pd
from p02_duzeltme import blok

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))

B = {b: blok(b) for b in ("yaz25", "guz25", "kis26")}
R = {}
for b, d in B.items():
    uf = pd.cut(d.ufuk_gun, [0, 15, 30, 45, 60, 75, 90, 105, 122])
    R[b] = dict(
        n=int(len(d)),
        rmsle=round(float(np.sqrt((d.r**2).mean())), 5),
        yanlilik=round(float(d.r.mean()), 4),
        sifir_orani=round(float((d.tuketim <= 0).mean()), 4),
        soguk_orani=round(float(d.soguk_mu.mean()), 4),
        ufuk_yanlilik={str(k): round(float(v), 4) for k, v in d.groupby(uf, observed=True).r.mean().items()},
        ay_yanlilik={str(k): round(float(v), 4) for k, v in d.groupby(d.ay, observed=True).r.mean().items()},
        sifir_satir_yanlilik=round(float(d.loc[d.tuketim <= 0, "r"].mean()), 4),
        sifir_satir_kare_pay=round(
            float((d.loc[d.tuketim <= 0, "r"] ** 2).sum() / (d.r**2).sum()), 4
        ),
        sifirsiz_yanlilik=round(float(d.loc[d.tuketim > 0, "r"].mean()), 4),
    )
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), usecols=["tarih"], parse_dates=["tarih"])
R["ham_train_araligi"] = [str(tr.tarih.min().date()), str(tr.tarih.max().date())]
print(json.dumps(R, indent=1, ensure_ascii=False))
json.dump(R, open(os.path.join(BURA, "p03_bloklar.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
