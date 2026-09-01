"""p06: aile ustunlugu SOGUGA MI OZGU yoksa BLOK artifakti mi?

p06_soguk_aile.json: yaz25 soguk lgbm 1,400 / cat 1,576; kis26 soguk cat
1,839 / lgbm 2,005. Isaret blok degistiriyor. Iki ayirt edici sinav:
  (a) AYNI blokta SICAK satirlarda siralama ne? Sicakta da ayni yonde
      donuyorsa bu bir BLOK artifakti, soguk hakkinda bir sey soylemiyor.
  (b) yaz25 soguk icindeki ALT KESITLERDE (ay, ufuk, guc, ilce) lgbm
      ustunlugu tekduze mi, yoksa birkac kesitten mi geliyor?
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import AO, DN  # noqa: E402

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
AILE = ("cat", "xgb", "lgbm")


def rmsle(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    R = {}
    for bad in ("yaz25", "guz25", "kis26"):
        blk = E[E._blok == bad]
        sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
        z = np.load(os.path.join(DN, f"soguk_tahmin_{bad}.npz"))
        ysog = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
        ysic = np.log1p(sic.tuketim.to_numpy(dtype="float64").clip(0))
        r = {}
        for a in AILE:
            ps = np.mean([z[k] for k in z.files if k.endswith("_" + a)], axis=0)
            r[f"soguk_{a}"] = round(rmsle(ysog - ps), 5)
            dosya = [os.path.join(AO, f"{bad}_{t}_{a}_uretim.npy") for t in (1000, 1001, 1002)]
            dosya = [f for f in dosya if os.path.exists(f)]
            if dosya:
                ph = np.mean([np.load(f).astype("float64") for f in dosya], axis=0)
                r[f"sicak_{a}"] = round(rmsle(ysic - ph), 5)
        R[bad] = r
        print(bad, json.dumps(r, indent=1), flush=True)

    # --- (b) yaz25 soguk alt kesitleri
    blk = E[E._blok == "yaz25"]
    sog = blk[blk.soguk_mu == 1]
    z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
    y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
    P = {a: np.mean([z[k] for k in z.files if k.endswith("_" + a)], axis=0) for a in AILE}
    hep = np.mean([z[k] for k in z.files], axis=0)
    d = pd.DataFrame(dict(y=y, hep=hep, **P))
    d["ay"] = pd.to_datetime(sog.tarih.values).month
    d["ufuk"] = pd.cut(sog.ufuk_gun.values, [0, 30, 60, 90, 130])
    d["guc"] = pd.qcut(sog.guc.values.astype(float), 4, duplicates="drop")
    d["ilce"] = sog.ilce_key.values
    kes = {}
    for kol in ("ay", "ufuk", "guc"):
        rows = []
        for k, g in d.groupby(kol, observed=True):
            rows.append(
                dict(
                    seviye=str(k),
                    n=int(len(g)),
                    **{a: round(rmsle(g.y - g[a]), 4) for a in AILE},
                    hepsi=round(rmsle(g.y - g.hep), 4),
                )
            )
        kes[kol] = rows
        print("---", kol)
        print(pd.DataFrame(rows).to_string(index=False))
    # ilce bazinda kac ilcede lgbm en iyi
    kaz = []
    for k, g in d.groupby("ilce", observed=True):
        if len(g) < 100:
            continue
        v = {a: rmsle(g.y - g[a]) for a in AILE}
        kaz.append(min(v, key=v.get))
    kes["ilce_kazanan_sayisi"] = {a: int(kaz.count(a)) for a in AILE}
    kes["ilce_n"] = len(kaz)
    print("ilce kazananlari:", kes["ilce_kazanan_sayisi"], "/", len(kaz))
    R["yaz25_soguk_kesitler"] = kes

    with open(os.path.join(BURA, "p06_soguk_aile2.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
