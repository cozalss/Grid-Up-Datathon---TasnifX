"""p25-C KIRMIZI TAKIM: blok CV tarafinda uc sinama.

Hat 4: YAPI kazanci ham / agr / pg olcutlerinde -- kohort agirligi kazanci
sisiriyor mu, kucultuyor mu?
Hat 5: seviyesiz merkezleme blok icinde AGIRLIKSIZ yapildi ama olcut
AGIRLIKLI -- agirlikli uzayda kacak seviye ne kadar?
Hat 1 (olcek): uygulanan delta olcegi m kat yanlissa (m=s_uyg/s_gercek)
kazanc nasil degisir? m in {0.7..1.3}.
Hat 2: yaz25/guz25'te 5 tohum var -- kazanc 5 tohumla ayakta mi?
"""

import json
import os
import sys

import numpy as np

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p20_harman import (  # noqa: E402
    AILE,
    BLOKLAR,
    HARMANLAR,
    buz,
    olcutler,
    veri_kur,
)

BETA = 0.60
W311 = HARMANLAR["ESKI_3_1_1"]
R = {}

B, _ = veri_kur()


def kazanc(bb, r_t, r_a, olcut):
    e_t = bb["lgy"] - np.maximum(buz(r_t, BETA) + bb["lguc"], 0.0)
    e_a = bb["lgy"] - np.maximum(buz(r_a, BETA) + bb["lguc"], 0.0)
    return olcutler(bb, e_t)[olcut] - olcutler(bb, e_a)[olcut]


# ---------- Hat 4 + Hat 5 + olcek duyarliligi (3 tohum, p20 ile ayni kurulum)
h4, h5, olcek = {}, {}, {}
for b in BLOKLAR:
    bb = B[b]
    p_t = bb["P"]["cat"].mean(axis=0)
    p_a = sum(w * bb["P"][a].mean(axis=0) for w, a in zip(W311, AILE))
    r_t = p_t - bb["lguc"]
    r_a = p_a - bb["lguc"]
    d = r_a - r_t
    d0 = d - d.mean()  # V1 merkezleme (AGIRLIKSIZ)
    r_a0 = r_t + d0
    h4[b] = {o: round(kazanc(bb, r_t, r_a0, o), 6) for o in ("ham", "agr", "pg")}
    # agirlikli uzayda kacak seviye
    h5[b] = {
        "agirliksiz_ort_d0": round(float(d0.mean()), 8),
        "agr_agirlikli_ort_d0": round(float((bb["w"] * d0).mean()), 6),
        "pg_altkume_ort_d0": round(float(d0[bb["pgm"]].mean()), 6),
        "d0_std": round(float(d0.std()), 5),
    }
    # olcek duyarliligi (agr)
    olcek[b] = {}
    for m in (0.7, 0.85, 1.0, 1.15, 1.3):
        olcek[b][str(m)] = round(kazanc(bb, r_t, r_t + m * d0, "agr"), 6)

for tablo, ad in ((h4, "hat4_YAPI_kazanci_olcutlere_gore"),):
    ortalar = {
        o: round(float(np.mean([tablo[b][o] for b in BLOKLAR])), 6) for o in ("ham", "agr", "pg")
    }
    R[ad] = {
        **tablo,
        "ORT": ortalar,
        "isaret": {o: f"{sum(tablo[b][o] > 0 for b in BLOKLAR)}/3" for o in ("ham", "agr", "pg")},
    }
R["hat5_merkezleme_kacagi"] = h5
R["olcek_duyarliligi_agr"] = {
    **olcek,
    "ORT": {
        str(m): round(float(np.mean([olcek[b][str(m)] for b in BLOKLAR])), 6)
        for m in (0.7, 0.85, 1.0, 1.15, 1.3)
    },
}

# ---------- Hat 2: 5 tohum (yaz25 / guz25)
DN = os.path.join(os.path.dirname(os.path.dirname(BURA)), "data", "interim", "deney")
h2 = {}
for b in ("yaz25", "guz25"):
    bb = B[b]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
    for etiket, toh in (("3tohum", (1000, 1001, 1002)), ("5tohum", (1000, 1001, 1002, 1003, 1004))):
        P = {a: np.mean([z[f"{t}_{a}"].astype("float64") for t in toh], axis=0) for a in AILE}
        r_t = P["cat"] - bb["lguc"]
        r_a = sum(w * P[a] for w, a in zip(W311, AILE)) - bb["lguc"]
        d0 = (r_a - r_t) - (r_a - r_t).mean()
        h2.setdefault(b, {})[etiket] = {
            o: round(kazanc(bb, r_t, r_t + d0, o), 6) for o in ("ham", "agr", "pg")
        }
    # capraz: taban 5 tohum cat (uretime yakin), delta 3 tohumdan
    P5 = {
        a: np.mean(
            [z[f"{t}_{a}"].astype("float64") for t in (1000, 1001, 1002, 1003, 1004)], axis=0
        )
        for a in AILE
    }
    P3 = {
        a: np.mean([z[f"{t}_{a}"].astype("float64") for t in (1000, 1001, 1002)], axis=0)
        for a in AILE
    }
    r_t5 = P5["cat"] - bb["lguc"]
    d3 = sum(w * P3[a] for w, a in zip(W311, AILE)) - P3["cat"]
    d3 = d3 - d3.mean()
    h2[b]["taban5_delta3(capraz)"] = {
        o: round(kazanc(bb, r_t5, r_t5 + d3, o), 6) for o in ("ham", "agr", "pg")
    }
R["hat2_tohum_kumesi"] = h2

yol = os.path.join(BURA, "p_kalici", "p25_kirmizi.json")
mevcut = {}
if os.path.exists(yol):
    with open(yol, encoding="utf-8") as fh:
        mevcut = json.load(fh)
mevcut["C_cv_sinamalari"] = R
with open(yol, "w", encoding="utf-8") as fh:
    json.dump(mevcut, fh, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
