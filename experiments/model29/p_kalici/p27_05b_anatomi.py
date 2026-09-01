"""p27-05b: sifir anatomisi + kahin ayrisimi (egitim YOK, yalniz groupby)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import HEDEF_SOGUK, blok, rmsle  # noqa: E402

CIK = os.path.dirname(os.path.abspath(__file__))
BLOKLAR = ("yaz25", "guz25", "kis26")
D = {b: blok(b) for b in BLOKLAR}
R = {}


def bilesik(sic, sog):
    return float(np.sqrt(HEDEF_SOGUK * sog**2 + (1 - HEDEF_SOGUK) * sic**2))


def bil_of(d, p):
    sg = d.soguk_mu.values == 1
    return bilesik(rmsle(d.y.values[~sg], p[~sg]), rmsle(d.y.values[sg], p[sg]))


anat, kt = [], []
for b in BLOKLAR:
    d = D[b]
    z = (d.tuketim.values <= 0)
    g = pd.DataFrame(dict(t=d.tanim.values, z=z)).groupby("t").z.agg(
        ["mean", "sum", "size"])
    tam = g[g["mean"] == 1.0]
    ara = g[(g["mean"] > 0) & (g["mean"] < 1)]
    anat.append(dict(
        blok=b, n_trafo=int(len(g)), n_sifir=int(z.sum()),
        trafo_TAM_sifir=int(len(tam)),
        trafo_HIC_sifir=int((g["mean"] == 0.0).sum()),
        trafo_ARALIKLI=int(len(ara)),
        sifirlarin_TAM_sifir_trafodan_payi=round(float(tam["sum"].sum() / z.sum()), 4),
        sifirlarin_ARALIKLI_trafodan_payi=round(float(ara["sum"].sum() / z.sum()), 4),
        aralikli_trafolarda_ort_sifir_orani=round(float(ara["mean"].mean()), 4)))
    m = np.isin(d.tanim.values, list(tam.index))
    taban = bil_of(d, d.p.values)
    p2 = d.p.values.copy(); p2[m] = 0.0
    p3 = d.p.values.copy(); p3[z] = 0.0
    kt.append(dict(blok=b, taban=round(taban, 5),
                   kahin_TAM_sifir_trafo=round(taban - bil_of(d, p2), 5),
                   kahin_TUM_sifir_satir=round(taban - bil_of(d, p3), 5),
                   n_satir_tam=int(m.sum())))
R["01_sifir_anatomisi"] = anat
R["02_kahin_ayrisim"] = kt
for x in anat + kt:
    print(x)
with open(os.path.join(CIK, "p27_05.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("yazildi p27_05.json")
