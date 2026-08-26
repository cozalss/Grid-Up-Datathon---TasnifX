"""Tek fit suresini olcer -- bagimsiz pencere kosusunun butcesi icin."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

t0 = time.time()
egitim, test = d.cerceveleri_kur()
ek = d._ek_kokenler_kur(False)
ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
ortak = [k for k in egitim.columns if k in ek.columns]
genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
print(f"yukleme {time.time() - t0:.0f} sn  genis {len(genis):,}")
tm.kategorik_kodla(genis, test)
kolonlar = tm.oznitelikler(genis)
print(f"kolon {len(kolonlar)}")

PENCERE = {
    "sub25": ("2025-02-01", "2025-03-31"),
    "bah25": ("2025-05-01", "2025-08-31"),
    "yaz25b": ("2025-07-01", "2025-10-31"),
    "guz25b": ("2025-09-01", "2025-12-31"),
    "kis26b": ("2025-11-01", "2026-02-28"),
    "bah26": ("2026-01-01", "2026-03-31"),
    "yaz25": ("2025-04-01", "2025-07-31"),
    "guz25": ("2025-08-01", "2025-11-30"),
    "kis26": ("2025-12-01", "2026-03-31"),
}

hedef = "bah25"
hb, hs = pd.Timestamp(PENCERE[hedef][0]), pd.Timestamp(PENCERE[hedef][1])
tut = [a for a, (b, s) in PENCERE.items() if pd.Timestamp(s) < hb or pd.Timestamp(b) > hs]
print(f"hedef {hedef}  egitim kokenleri {tut}")
dog = genis[genis["_blok"] == hedef]
kalan = genis[genis["_blok"].isin(tut)]
print(f"dogrulama {len(dog):,}  egitim {len(kalan):,}")

t0 = time.time()
maskeli = d.soguk_maskele(kalan, kolonlar, tm.SOGUK_MASKE_ORANI, 1000)
print(f"maskeleme {time.time() - t0:.0f} sn")
t0 = time.time()
p = di.egit_tahmin("cat", maskeli, dog, kolonlar, 1000)
print(f"cat fit+tahmin {time.time() - t0:.0f} sn  shape {p.shape}")
t0 = time.time()
p = di.egit_tahmin("lgbm", maskeli, dog, kolonlar, 1000)
print(f"lgbm fit+tahmin {time.time() - t0:.0f} sn")
t0 = time.time()
p = di.egit_tahmin("xgb", maskeli, dog, kolonlar, 1000)
print(f"xgb fit+tahmin {time.time() - t0:.0f} sn")
