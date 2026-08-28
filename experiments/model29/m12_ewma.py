"""EKSEN 1b: cok kisa pencereler, EWMA yari-omru, pencere HARMANI."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *

tr = yukle()
KISA = [1, 2, 3, 4, 5, 7, 10, 14, 21, 28]
HL = [3, 5, 7, 10, 14, 21, 28, 45, 60, 90]
res = {"kisa": {}, "ewma": {}, "harman": {}}
for kesim in KESIMLER:
    k = pd.Timestamp(kesim)
    gec, hed = hazirla(tr, kesim)
    tam = pencere_seviye(gec, kesim, None, "mean")
    kok = float(gec.ly.mean())
    print(f"\n===== {kesim} =====")
    res["kisa"][kesim] = {}
    for g in KISA:
        s = pencere_seviye(gec, kesim, g, "mean")
        r = puan(hed, geri_dolgu(hed, s, tam, kok=kok))
        res["kisa"][kesim][f"g{g}"] = r
    print(
        "kisa pencere:", " ".join(f"{g}:{res['kisa'][kesim][f'g{g}']['rmsle']:.4f}" for g in KISA)
    )

    # EWMA: yas = kesim - tarih (gun), agirlik 0.5^(yas/hl)
    gec2 = gec.assign(yas=(k - gec.tarih).dt.days)
    res["ewma"][kesim] = {}
    for hl in HL:
        w = 0.5 ** (gec2.yas.values / hl)
        num = pd.Series(w * gec2.ly.values, index=gec2.index).groupby(gec2.tanim).sum()
        den = pd.Series(w, index=gec2.index).groupby(gec2.tanim).sum()
        s = num / den
        r = puan(hed, geri_dolgu(hed, s, tam, kok=kok))
        res["ewma"][kesim][f"hl{hl}"] = r
    print(
        "ewma hl   :", " ".join(f"{hl}:{res['ewma'][kesim][f'hl{hl}']['rmsle']:.4f}" for hl in HL)
    )

    # harman: a*mean7 + (1-a)*meanW
    s7 = pencere_seviye(gec, kesim, 7, "mean")
    res["harman"][kesim] = {}
    for W in [28, 91, 182, None]:
        sW = pencere_seviye(gec, kesim, W, "mean")
        best = (9, None)
        for a in np.arange(0, 1.01, 0.1):
            p = a * geri_dolgu(hed, s7, tam, kok=kok) + (1 - a) * geri_dolgu(hed, sW, tam, kok=kok)
            r = puan(hed, p)
            res["harman"][kesim][f"7+{W}@{a:.1f}"] = r["rmsle"]
            if r["rmsle"] < best[0]:
                best = (r["rmsle"], a)
        print(f"  harman 7+{str(W):>4s}: en iyi a={best[1]:.1f} RMSLE {best[0]:.4f}")
json_yaz("eksen1b_kisa_ewma_harman", res)

print("\n== 4-kesim ortalamasi ==")
for tip in ["kisa", "ewma"]:
    ad = list(res[tip][KESIMLER[0]].keys())
    o = {a: np.mean([res[tip][k][a]["rmsle"] for k in KESIMLER]) for a in ad}
    print(tip, " ".join(f"{a}:{v:.4f}" for a, v in o.items()))
    json_yaz(f"eksen1b_{tip}_ort", o)
ad = list(res["harman"][KESIMLER[0]].keys())
o = {a: np.mean([res["harman"][k][a] for k in KESIMLER]) for a in ad}
print("harman en iyi 8:", sorted(o.items(), key=lambda x: x[1])[:8])
json_yaz("eksen1b_harman_ort", o)
