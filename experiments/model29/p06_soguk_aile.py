"""p06: SOGUK harmanin AILE bilesimi. cat/xgb/lgbm uyeleri esit agirlikli --
soguk satirlarda BU DOGRU MU?

Ipucu: p06_soguk_yeniden.py'de yalniz-lgbm 3 tohumlu bir soguk uzman yaz25
soguk RMSLE 1,40520 verdi; uretimin 15 uyeli (5 tohum x 3 aile) harmani
1,43592. Yani cat/xgb uyeleri soguk tarafta ZARAR veriyor olabilir.

Karar SIZINTISIZ olmali: aile agirligi UC BLOKTA DA ayni yone isaret
ediyorsa yapisaldir; yalnizca yaz25'te iyiyse p01'in tuzagidir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN  # noqa: E402

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
AO = os.path.join(os.path.dirname(os.path.dirname(BURA)), "data/interim/aile_onbellek")
AILE = ("cat", "xgb", "lgbm")


def rmsle(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def blok_soguk(bad):
    blk = E[E._blok == bad]
    sog = blk[blk.soguk_mu == 1]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{bad}.npz"))
    y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
    return sog, y, {k: z[k] for k in z.files}


def main():
    R = {}
    W = {}
    for bad in ("yaz25", "guz25", "kis26"):
        sog, y, z = blok_soguk(bad)
        aile_p = {
            a: np.mean([v for k, v in z.items() if k.endswith("_" + a)], axis=0) for a in AILE
        }
        hepsi = np.mean(list(z.values()), axis=0)
        r = dict(n=int(len(y)), uye=len(z), hepsi=round(rmsle(y - hepsi), 5))
        for a in AILE:
            r[a] = round(rmsle(y - aile_p[a]), 5)
        # ikili harmanlar
        r["cat+lgbm"] = round(rmsle(y - (aile_p["cat"] + aile_p["lgbm"]) / 2), 5)
        r["xgb+lgbm"] = round(rmsle(y - (aile_p["xgb"] + aile_p["lgbm"]) / 2), 5)
        r["cat+xgb"] = round(rmsle(y - (aile_p["cat"] + aile_p["xgb"]) / 2), 5)
        # dogrusal en iyi (blok ici KAHIN -- yalniz karsilastirma icin)
        A = np.c_[[aile_p[a] for a in AILE]].T
        X = np.c_[np.ones(len(y)), A]
        cf = np.linalg.lstsq(X, y, rcond=None)[0]
        r["kahin_dogrusal"] = round(rmsle(y - X @ cf), 5)
        r["kahin_katsayi"] = [round(float(v), 3) for v in cf]
        # basit simpleks agirligi (sabitsiz, toplami 1) -- blok ici kahin
        from itertools import product

        en = None
        for w in product(np.arange(0, 1.05, 0.05), repeat=2):
            if sum(w) > 1:
                continue
            ww = (w[0], w[1], 1 - w[0] - w[1])
            v = rmsle(y - A @ np.array(ww))
            if en is None or v < en[0]:
                en = (v, ww)
        r["kahin_simpleks"] = dict(rmsle=round(en[0], 5), w=[round(float(x), 2) for x in en[1]])
        W[bad] = np.array(en[1])
        R[bad] = r
        print(bad, json.dumps(r, indent=1), flush=True)

    # --- SIZINTISIZ: agirligi DIS bloklardan (guz25+kis26 birlestirilmis) sec, yaz25'te uygula
    Ys, As = [], []
    for bad in ("guz25", "kis26"):
        sog, y, z = blok_soguk(bad)
        Ys.append(y)
        As.append(
            np.c_[
                [np.mean([v for k, v in z.items() if k.endswith("_" + a)], axis=0) for a in AILE]
            ].T
        )
    yd, Ad = np.concatenate(Ys), np.vstack(As)
    from itertools import product

    en = None
    for w in product(np.arange(0, 1.05, 0.05), repeat=2):
        if sum(w) > 1:
            continue
        ww = np.array([w[0], w[1], 1 - w[0] - w[1]])
        v = rmsle(yd - Ad @ ww)
        if en is None or v < en[0]:
            en = (v, ww)
    sog, y, z = blok_soguk("yaz25")
    Ay = np.c_[[np.mean([v for k, v in z.items() if k.endswith("_" + a)], axis=0) for a in AILE]].T
    R["SIZINTISIZ"] = dict(
        aile=list(AILE),
        dis_agirlik=[round(float(x), 2) for x in en[1]],
        dis_rmsle=round(en[0], 5),
        yaz25_taban=round(rmsle(y - Ay.mean(axis=1)), 5),
        yaz25_dis_agirlikla=round(rmsle(y - Ay @ en[1]), 5),
    )
    print(json.dumps(R["SIZINTISIZ"], indent=1))

    with open(os.path.join(BURA, "p06_soguk_aile.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
