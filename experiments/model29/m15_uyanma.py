"""EKSEN 4: sifir/olu gecmisli trafolar - uyanma olasiligi ve optimal sabit."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *

tr = yukle()
res = {}
for kesim in KESIMLER:
    k = pd.Timestamp(kesim)
    gec, hed = hazirla(tr, kesim)
    tam = pencere_seviye(gec, kesim, None, "mean")
    kok = float(gec.ly.mean())
    s7 = pencere_seviye(gec, kesim, 7, "mean")
    p = geri_dolgu(hed, s7, tam, kok=kok)
    e2 = (p - hed.ly.values) ** 2
    tot = e2.sum()
    # trafo bazli gecmis ozellikleri
    gg = gec.groupby("tanim")
    ozet = pd.DataFrame(
        {
            "nsat": gg.size(),
            "maxt": gg.tuketim.max(),
            "sonT": gg.tarih.max(),
            "tumly": gg.ly.mean(),
            "ilkT": gg.tarih.min(),
        }
    )
    ozet["bosluk"] = (k - ozet.sonT).dt.days
    # son N gun tamamen sifir mi (mevcut satirlar arasinda)
    for W in [7, 28, 56]:
        gw = gec[gec.tarih > k - pd.Timedelta(days=W)]
        ozet[f"max{W}"] = gw.groupby("tanim").tuketim.max()
    hed2 = hed.join(ozet, on="tanim")

    # gruplar
    def grup(r):
        if r.maxt < 1:
            return "A_tum_gecmis_sifir"
        if r.bosluk > 14:
            return "B_bayat(>14g veri yok)"
        if r.max28 < 1:
            return "C_son28_sifir"
        if r.max7 < 1:
            return "D_son7_sifir"
        return "E_normal"

    hed2["grup"] = hed2.apply(grup, axis=1)
    print(f"\n===== {kesim} | taban(mean7) sicak RMSLE {np.sqrt(e2.mean()):.4f} =====")
    print(
        f"{'grup':26s} {'satir':>8s} {'trafo':>6s} {'kutle%':>7s} {'RMSLE':>7s} {'hedef_ly_ort':>12s} {'hedefte<1%':>10s} {'opt_sbt':>8s} {'opt_RMSLE':>9s}"
    )
    res[kesim] = {}
    for gr, sub in hed2.groupby("grup"):
        m = hed2.grup.values == gr
        opt = float(sub.ly.mean())
        optr = float(np.sqrt(((sub.ly - opt) ** 2).mean()))
        d = {
            "n": int(m.sum()),
            "trafo": int(sub.tanim.nunique()),
            "kutle": float(e2[m].sum() / tot),
            "rmsle": float(np.sqrt(e2[m].mean())),
            "hedef_ly_ort": opt,
            "hedef_sifir_oran": float((sub.tuketim < 1).mean()),
            "opt_sabit": opt,
            "opt_rmsle": optr,
        }
        res[kesim][gr] = d
        print(
            f"{gr:26s} {d['n']:8,d} {d['trafo']:6,d} {100 * d['kutle']:6.1f}% {d['rmsle']:7.3f} {opt:12.3f} {100 * d['hedef_sifir_oran']:9.1f}% {opt:8.2f} {optr:9.3f}"
        )
json_yaz("eksen4_gruplar", res)
