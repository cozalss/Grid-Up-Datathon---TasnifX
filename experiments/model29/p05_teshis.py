"""p05 teshis: URETIM tabani, p03 tezgahinin tabaninin yaptigi hatayi yapiyor mu?

p03'un iki-asamali kazanci "sifirlarin ortak regresyona girmesi ust kovalari
ASAGI cekiyor" mekanizmasina dayaniyordu (tezgah tabani: (1e3,5e3] sapma
-0.257, (5e3,1e5] sapma -0.419). Ayni kova ayristirmasi URETIM tahmini
uzerinde hesaplanir; sapmalar kucukse duzeltilecek hata YOKTUR.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
BURA = os.path.dirname(os.path.abspath(__file__))
KOVA = [
    (-1, 0),
    (0, 1),
    (1, 10),
    (10, 50),
    (50, 100),
    (100, 500),
    (500, 1e3),
    (1e3, 5e3),
    (5e3, 1e5),
    (1e5, np.inf),
]
AD = [
    "=0",
    "(0,1]",
    "(1,10]",
    "(10,50]",
    "(50,100]",
    "(100,500]",
    "(500,1e3]",
    "(1e3,5e3]",
    "(5e3,1e5]",
    ">1e5",
]

e = pd.read_parquet(os.path.join(DN, "egitim.parquet"), columns=["tuketim", "soguk_mu", "_blok"])
R = {"aciklama": __doc__.strip(), "bloklar": {}}
for blok in ("yaz25", "guz25", "kis26"):
    blk = e[e._blok == blok]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for aa in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    bf = e.loc[idx]
    y = bf.tuketim.to_numpy(dtype=np.float64)
    yv = np.log1p(y)
    r = pb - yv
    tot = float((r * r).sum())
    kv = []
    for (lo, hi), ad in zip(KOVA, AD):
        m = (y > lo) & (y <= hi) if lo >= 0 else (y == 0)
        if not m.any():
            continue
        kv.append(
            {
                "kova": ad,
                "n": int(m.sum()),
                "pay_satir": float(m.mean()),
                "e2_ort": float(np.mean(r[m] ** 2)),
                "pay_hata": float((r[m] ** 2).sum() / tot),
                "sapma": float(r[m].mean()),
            }
        )
    sg = bf.soguk_mu.to_numpy() == 1
    R["bloklar"][blok] = {
        "rmsle": float(np.sqrt(np.mean(r * r))),
        "genel_sapma": float(r.mean()),
        "sifir_satirda_ortalama_tahmin": float(pb[y == 0].mean()),
        "sicak": {"rmsle": float(np.sqrt(np.mean(r[~sg] ** 2))), "sapma": float(r[~sg].mean())},
        "soguk": {"rmsle": float(np.sqrt(np.mean(r[sg] ** 2))), "sapma": float(r[sg].mean())},
        "kova": kv,
    }
    print(blok, json.dumps(R["bloklar"][blok]["kova"], ensure_ascii=False, indent=1))

json.dump(
    R,
    open(os.path.join(BURA, "p05_teshis.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
print("yazildi")
