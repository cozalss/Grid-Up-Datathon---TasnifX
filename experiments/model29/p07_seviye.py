"""Tek pozitif yon: SEVIYE (yayilma) kalibrasyonu. Buyuklugunu ve
tasinabilirligini olcer. Parametreler dis bloklardan; yaz25 yalnizca olcum."""

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
D = {}


def beta_dis(d, maske=None):
    """Blok icinde ORTALANMIS artigi ortalanmis tahmine regresle -> saf yayilma."""
    if maske is not None:
        d = d[maske]
    out = {}
    for b, g in d.groupby("_blok", observed=True):
        x = g.p.values - g.p.mean()
        r = g.r.values - g.r.mean()
        out[b] = float((x * r).sum() / (x * x).sum())
    return out


bd = beta_dis(dis)
R["beta_blok_basina"] = {k: round(v, 5) for k, v in bd.items()}
beta = float(np.mean(list(bd.values())))
R["beta_dis_ortalama"] = round(beta, 5)
# yaz25'in KENDI optimumu (sizintili referans)
xz = yaz.p.values - yaz.p.mean()
rz = yaz.r.values - yaz.r.mean()
R["beta_yaz25_optimum_sizintili"] = round(float((xz * rz).sum() / (xz * xz).sum()), 5)

pm = float(yaz.p.mean())
for lam in (0.5, 1.0):
    D[f"B1_yayilma_beta*{lam}"] = yaz.p.values + lam * beta * (yaz.p.values - pm)
# sicak/soguk ayri
bh = beta_dis(dis, dis.soguk_mu == 0)
bc = beta_dis(dis, dis.soguk_mu == 1)
bhm, bcm = float(np.mean(list(bh.values()))), float(np.mean(list(bc.values())))
R["beta_sicak"], R["beta_soguk"] = round(bhm, 5), round(bcm, 5)
sg = yaz.soguk_mu.values == 1
pms = yaz.p.values[~sg].mean()
pmc = yaz.p.values[sg].mean()
pp = yaz.p.values.copy()
pp[~sg] += bhm * (yaz.p.values[~sg] - pms)
pp[sg] += bcm * (yaz.p.values[sg] - pmc)
D["B2_yayilma_rejim_ayri"] = pp
# t_log_ort'a harmanlama (trafonun kendi gecmis seviyesi)
h = yaz.t_log_ort.values.copy()
h = np.where(np.isfinite(h), h, yaz.p.values)
for w in (0.05, 0.1, 0.2):
    D[f"B3_gecmis_ort_harman_w={w}"] = (1 - w) * yaz.p.values + w * h

R["duzeltmeler"] = {}
for ad, x in D.items():
    s, sw = skor(yaz, x)
    R["duzeltmeler"][ad] = dict(
        rmsle=round(s, 5), kazanc=round(t0 - s, 5),
        rmsle_test_bilesimi=round(sw, 5), kazanc_test_bilesimi=round(t0w - sw, 5),
    )
    print(f"{ad:26s} RMSLE={s:.5f} kazanc={t0 - s:+.5f}  (agirlikli {sw:.5f} {t0w - sw:+.5f})")

# yaz25-optimal beta taramasi (tavan)
best = min(((skor(yaz, yaz.p.values + b * (yaz.p.values - pm))[0], b) for b in np.arange(-0.20, 0.06, 0.01)))
R["beta_taramasi_tavan"] = dict(en_iyi_beta=round(float(best[1]), 3), rmsle=round(best[0], 5), kazanc=round(t0 - best[0], 5))
print("tavan:", R["beta_taramasi_tavan"])
json.dump(R, open(os.path.join(BURA, "p07_seviye.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
