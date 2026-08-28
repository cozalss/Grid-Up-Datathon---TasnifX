"""EKSEN 1: pencere secimi + istatistik (ort/medyan/trim)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *

tr = yukle()
GUNLER = [7, 14, 28, 56, 91, 182, 365, None]
sonuc = {}
for kesim in KESIMLER:
    gec, hed = hazirla(tr, kesim)
    tam = pencere_seviye(gec, kesim, None, "mean")
    kok = float(gec.ly.mean())
    print(f"\n===== KESIM {kesim} | sicak n={len(hed):,} trafo={hed.tanim.nunique():,} =====")
    print(f"{'kestirimci':22s} {'RMSLE':>8s} {'ay1':>7s} {'ay2':>7s} {'ay3':>7s} {'ay4':>7s}")
    sonuc[kesim] = {}
    for stat in ["mean", "median", "trim"]:
        for gun in GUNLER:
            s = pencere_seviye(gec, kesim, gun, stat)
            p = geri_dolgu(hed, s, tam, kok=kok)
            r = puan(hed, p)
            ad = f"{stat}{gun if gun else 'ALL'}"
            sonuc[kesim][ad] = r
            print(
                f"{ad:22s} {r['rmsle']:8.4f} "
                + " ".join(f"{r.get(f'ay{a}', np.nan):7.4f}" for a in [1, 2, 3, 4])
            )
json_yaz("eksen1_pencere", sonuc)

# ortalama sirali ozet
print("\n===== 4 KESIM ORTALAMASI (RMSLE) =====")
adlar = list(sonuc[KESIMLER[0]].keys())
ort = {a: np.mean([sonuc[k][a]["rmsle"] for k in KESIMLER]) for a in adlar}
for a, v in sorted(ort.items(), key=lambda x: x[1]):
    print(f"{a:22s} {v:.4f}")
json_yaz("eksen1_ortalama", ort)
