from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d, deney_ileri as di, tuketim_model as tm  # noqa

ONB = KOK / "data" / "interim" / "deney"
egitim = pd.read_parquet(ONB / "egitim.parquet")
test = pd.read_parquet(ONB / "test.parquet")
tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
tm.kategorik_kodla(egitim, test)
b = "yaz25"
dog = egitim[egitim["_blok"] == b]
kalan = egitim[egitim["_blok"] != b]
soguk_t = set(dog.loc[dog["soguk_mu"] == 1, tm.GRUP].unique())
onar = kalan[~kalan[tm.GRUP].isin(soguk_t)]
soguk = (dog["soguk_mu"] == 1).to_numpy()
gercek = dog[tm.HEDEF].to_numpy()
for ad, kaynak in (("KIRLI", kalan), ("ONARILMIS", onar)):
    t0 = time.time()
    mask = d.soguk_maskele(kaynak, uretim, 1.00, 1000)
    t1 = time.time()
    lg = di.egit_tahmin("cat", mask, dog, uretim, 1000, depth=7)
    t2 = time.time()
    tah = np.clip(np.expm1(lg), 0.0, None)
    print(
        f"{ad:10} n={len(kaynak):,}  maske {t1 - t0:.0f}s  cat-d7 fit+pred {t2 - t1:.0f}s  "
        f"soguk RMSLE {tm.rmsle(gercek[soguk], tah[soguk]):.5f}"
    )
    del mask
