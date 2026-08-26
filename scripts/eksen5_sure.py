# ruff: noqa
"""EKSEN 5 -- tek fit suresi olcumu (butce planlamasi)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

egitim, test = d.cerceveleri_kur()
kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
tm.kategorik_kodla(egitim, test)
print(f"  egitim {egitim.shape}  kolon {len(kolonlar)}")
kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, "yaz25")
sicak = ~soguk
kalan_s = kalan[(kalan["soguk_mu"] == 0)].reset_index(drop=True)
dog_s = dogrulama[sicak].reset_index(drop=True)
print(f"  kalan sicak {len(kalan_s):,}  dogrulama sicak {len(dog_s):,}")
for aile in ("lgbm", "cat"):
    t0 = time.time()
    lt = di.egit_tahmin(aile, kalan_s, dog_s, kolonlar, 1000)
    print(f"  {aile}: {time.time() - t0:.1f} sn  RMSLE {tm.rmsle(gercek[sicak], np.expm1(lt)):.5f}")
