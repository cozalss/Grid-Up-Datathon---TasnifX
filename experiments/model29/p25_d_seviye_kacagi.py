"""p25-D KIRMIZI TAKIM: merkezleme uyusmazliginin KAZANCA etkisi.

Testte delta, TEST soguk dagilimi uzerinde merkezlenir (ort=0). Blok
CV'sinde ise BLOK dagilimi uzerinde merkezlendi ama olcut TEST-agirlikli
(agr). Test-benzeri kohortlar blokta +0.02..+0.05 seviye aldi -- test
uygulamasinda bu YOK. Dogru CV taklidi: deltayi AGIRLIKLI merkezle.

Olculen: agr kazanci (p20 usulu, agirliksiz merkez) vs agr kazanci
(agirlikli merkez). Fark = CV sayisindaki kacak seviye katkisi.
"""

import json
import os
import sys

import numpy as np

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p20_harman import AILE, BLOKLAR, HARMANLAR, buz, olcutler, veri_kur  # noqa: E402

BETA = 0.60
R = {}
B, _ = veri_kur()


def agr_mse(bb, r):
    e = bb["lgy"] - np.maximum(buz(r, BETA) + bb["lguc"], 0.0)
    return olcutler(bb, e)


for harman in ("ESKI_3_1_1", "ESIT"):
    W = HARMANLAR[harman]
    T = {}
    for b in BLOKLAR:
        bb = B[b]
        r_t = bb["P"]["cat"].mean(axis=0) - bb["lguc"]
        r_a = sum(w * bb["P"][a].mean(axis=0) for w, a in zip(W, AILE)) - bb["lguc"]
        d = r_a - r_t
        d0 = d - d.mean()  # p20 usulu (agirliksiz)
        d0w = d - float((bb["w"] * d).mean())  # test-agirlikli merkez
        d0pg = d - float(d[bb["pgm"]].mean())  # pg altkumesinde merkez
        m_t = agr_mse(bb, r_t)
        m_0 = agr_mse(bb, r_t + d0)
        m_w = agr_mse(bb, r_t + d0w)
        m_pg = agr_mse(bb, r_t + d0pg)
        T[b] = {
            "agr_kazanc_AGIRLIKSIZ_merkez(p20)": round(m_t["agr"] - m_0["agr"], 6),
            "agr_kazanc_AGIRLIKLI_merkez": round(m_t["agr"] - m_w["agr"], 6),
            "kacak_seviye_katkisi": round(m_w["agr"] - m_0["agr"], 6),
            "pg_kazanc_AGIRLIKSIZ_merkez": round(m_t["pg"] - m_0["pg"], 6),
            "pg_kazanc_PG_merkez": round(m_t["pg"] - m_pg["pg"], 6),
        }
    for k in (
        "agr_kazanc_AGIRLIKSIZ_merkez(p20)",
        "agr_kazanc_AGIRLIKLI_merkez",
        "kacak_seviye_katkisi",
        "pg_kazanc_AGIRLIKSIZ_merkez",
        "pg_kazanc_PG_merkez",
    ):
        v = [T[b][k] for b in BLOKLAR]
        T[f"ORT_{k}"] = round(float(np.mean(v)), 6)
        T[f"isaret_{k}"] = f"{sum(x > 0 for x in v)}/3"
    R[harman] = T

# test_dMSE ve LB karsiligi (agirlikli merkez ile)
for harman in ("ESKI_3_1_1", "ESIT"):
    g = R[harman]["ORT_agr_kazanc_AGIRLIKLI_merkez"]
    test_dmse = g * 0.2216
    R[harman]["DUZELTILMIS"] = {
        "ORT_dMSE_agr": g,
        "test_dMSE": round(test_dmse, 6),
        "dRMSLE_oran1.0": round(test_dmse / (2 * 1.00115), 5),
        "beklenen_LB_oran1.0": round(1.00115 - test_dmse / (2 * 1.00115), 5),
        "beklenen_LB_oran0.5": round(1.00115 - 0.5 * test_dmse / (2 * 1.00115), 5),
    }

yol = os.path.join(BURA, "p_kalici", "p25_kirmizi.json")
mevcut = {}
if os.path.exists(yol):
    with open(yol, encoding="utf-8") as fh:
        mevcut = json.load(fh)
mevcut["D_merkezleme_kacagi"] = R
with open(yol, "w", encoding="utf-8") as fh:
    json.dump(mevcut, fh, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
