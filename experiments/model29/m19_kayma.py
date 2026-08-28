"""EKSEN 2b/3: global kayma (drift), takvim-ayi profili, trend."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()
res = {}
print("=== A) optimal GLOBAL kayma (E_normal uzerinde, taban 0.8*k7+0.2*all) ===")
for kesim in KESIMLER + ["2025-12-31"]:
    uf = 4 if kesim != "2025-12-31" else 3
    gec, hed = hazirla(tr, kesim, uf)
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    kok = float(gec.ly.mean())
    base = 0.8 * geri_dolgu(hed, oz.k7, oz.ly_all, kok=kok) + 0.2 * geri_dolgu(
        hed, oz.ly_all, kok=kok
    )
    m = hed.tanim.map(gr).values == "E_normal"
    r = hed.ly.values[m] - base[m]
    print(
        f"{kesim}: E_normal n={m.sum():,} artik ort {r.mean():+.4f} -> RMSLE {np.sqrt((r**2).mean()):.4f} "
        f"kaymali {np.sqrt(((r - r.mean()) ** 2).mean()):.4f} (kazanc {np.sqrt((r**2).mean()) - np.sqrt(((r - r.mean()) ** 2).mean()):.4f})"
    )
    # ay ofsetine gore artik
    for a in sorted(hed.ay_ofset.unique()):
        mm = m & (hed.ay_ofset.values == a)
        print(f"    ay{a}: artik ort {(hed.ly.values[mm] - base[mm]).mean():+.4f} n={mm.sum():,}")
    res[kesim] = {
        "artik_ort": float(r.mean()),
        "rmsle": float(np.sqrt((r**2).mean())),
        "kaymali_rmsle": float(np.sqrt(((r - r.mean()) ** 2).mean())),
        "ay_artik": {
            int(a): float(
                (
                    hed.ly.values[m & (hed.ay_ofset.values == a)]
                    - base[m & (hed.ay_ofset.values == a)]
                ).mean()
            )
            for a in sorted(hed.ay_ofset.unique())
        },
    }
json_yaz("eksen2b_global_kayma", res)
