"""p25-E KIRMIZI TAKIM: AGIRLIKLI merkezlemeyle kirpma merdiveni.

p20'nin kirpma merdiveni (K uctan trafo kirp) AGIRLIKSIZ merkezle
olculmustu. Duzeltilmis (agirlikli merkez) kazanc icin ayni merdiven.
Kirpma: trafo bazinda dMSE katkisi hesaplanir, |katki| en buyuk K trafo
(iki uctan K'ser DEGIL -- p20b'deki gibi en buyuk pozitif K/2 + negatif K/2
yerine burada iki uctan K'ser trafo dusurulur) disarida birakilir.
p20b usulu: katkiya gore siralayip alttan ve ustten K trafo at.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p20_harman import AILE, BLOKLAR, HARMANLAR, buz, veri_kur  # noqa: E402

BETA = 0.60
W311 = HARMANLAR["ESKI_3_1_1"]
R = {}
B, _ = veri_kur()

for merkez in ("agirliksiz", "agirlikli"):
    R[merkez] = {}
    for b in BLOKLAR:
        bb = B[b]
        r_t = bb["P"]["cat"].mean(axis=0) - bb["lguc"]
        r_a = sum(w * bb["P"][a].mean(axis=0) for w, a in zip(W311, AILE)) - bb["lguc"]
        d = r_a - r_t
        d0 = d - (d.mean() if merkez == "agirliksiz" else float((bb["w"] * d).mean()))
        e_t = bb["lgy"] - np.maximum(buz(r_t, BETA) + bb["lguc"], 0.0)
        e_a = bb["lgy"] - np.maximum(buz(r_t + d0, BETA) + bb["lguc"], 0.0)
        katki = bb["w"] * (e_t * e_t - e_a * e_a)  # satir basi agirlikli dMSE katkisi
        df = pd.DataFrame({"k": katki, "t": bb["tanim"]})
        tr = df.groupby("t")["k"].sum().sort_values()
        n = len(katki)
        R[merkez][b] = {}
        for K in (0, 5, 10, 25, 50):
            if K == 0:
                kal = tr
            else:
                kal = tr.iloc[K:-K]
            R[merkez][b][f"K={K}"] = round(float(kal.sum() / n), 6)
    for K in (0, 5, 10, 25, 50):
        R[merkez][f"ORT_K={K}"] = round(
            float(np.mean([R[merkez][b][f"K={K}"] for b in BLOKLAR])), 6
        )
        R[merkez][f"isaret_K={K}"] = f"{sum(R[merkez][b][f'K={K}'] > 0 for b in BLOKLAR)}/3"

# LB karsiliklari (agirlikli, K=25, oranlar)
g = R["agirlikli"]["ORT_K=25"]
R["DUZELTILMIS_MUHAFAZAKAR"] = {
    "K25_agirlikli_ORT_dMSE": g,
    "K25_test_dMSE": round(g * 0.2216, 6),
    "K25_LB_oran0.5": round(1.00115 - 0.5 * g * 0.2216 / (2 * 1.00115), 5),
    "K25_LB_oran1.0": round(1.00115 - g * 0.2216 / (2 * 1.00115), 5),
}

yol = os.path.join(BURA, "p_kalici", "p25_kirmizi.json")
mevcut = {}
if os.path.exists(yol):
    with open(yol, encoding="utf-8") as fh:
        mevcut = json.load(fh)
mevcut["E_kirpma_agirlikli_merkez"] = R
with open(yol, "w", encoding="utf-8") as fh:
    json.dump(mevcut, fh, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
